# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
import numpy as np
import copy

from gt_sp.comm_profiler import profile_call
from gt_sp.initialize import (
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
)
from gt_sp.utils import (
    split_tensor_along_second_dim, 
    merge_global_token, 
    extend_global_token,
    extend_global_token0,
    copy_global_token0,
)


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def _run_all_to_all(group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int, tag: str) -> Tensor:
        seq_world_size = get_sequence_parallel_world_size()

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        payload_bytes = int(input.numel() * input.element_size())

        def _collective():
            torch.distributed.all_to_all(output_list, input_list, group=group)
            return torch.cat(output_list, dim=gather_idx).contiguous()

        return profile_call(tag, _collective, device=input.device, payload_bytes=payload_bytes)

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _SeqAllToAll._run_all_to_all(group, input, scatter_idx, gather_idx, "seq_all_to_all_fwd")

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        grad_input = _SeqAllToAll._run_all_to_all(
            ctx.group,
            grad_output[0],
            ctx.gather_idx,
            ctx.scatter_idx,
            "seq_all_to_all_bwd",
        )
        return (None, grad_input, None, None)


class _SeqGather(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate.
    Forward: all-gather
    Backward: split
    """ 

    @staticmethod
    def forward(ctx, input_, gather_idx):
        """Gather tensors and concatinate along the second dimension."""
        
        seq_world_size = get_sequence_parallel_world_size()
        rank = get_sequence_parallel_rank()
        ctx.gather_idx = gather_idx
        ctx.seq_world_size = seq_world_size
        ctx.rank = rank
        
        # Bypass the function if we are using only 1 GPU.
        if seq_world_size == 1:
            return input_

        # Size and dimension.
        tensor_list = [torch.empty_like(input_) for _ in range(seq_world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_, group=get_sequence_parallel_group()) # Note: can only on same size tensor

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=gather_idx).contiguous()
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Split the tensor along its second dimension and keep the
        corresponding slice."""
        # Bypass the function if we are using only 1 GPU.
        if ctx.seq_world_size == 1:
            return (grad_output, None)

        # Split along second dimension.
        input_list = split_tensor_along_second_dim(grad_output, ctx.seq_world_size)

        # Note: torch.split does not create contiguous tensors by default.
        output = input_list[ctx.rank].contiguous()

        return (output, None)


class _SeqScatter(torch.autograd.Function):
    """Split the input in head dim and keep only the corresponding chunk to the rank.
    Forward: split
    Backward: all-gather
    """ 

    @staticmethod
    def forward(ctx, input_):
        # input_: [b, n_head, s+1, s+1]
        seq_world_size = get_sequence_parallel_world_size()
        seq_parallel_world_rank = get_sequence_parallel_rank()
        
        assert input_.size()[1] % seq_world_size == 0
        dim_size = input_.size()[1] // seq_world_size
        input_list = [t.contiguous() for t in torch.split(input_, dim_size, dim=1)]
        output = input_list[seq_parallel_world_rank]
        
        return output


    @staticmethod
    def backward(ctx, grad_output):
        seq_world_size = get_sequence_parallel_world_size()
        seq_parallel_world_rank = get_sequence_parallel_rank()
        
        # Bypass the function if we are using only 1 GPU.
        if seq_world_size == 1:
            return grad_output

        # print(f'rank {seq_parallel_world_rank} {grad_output.shape}')
       
        tensor_list = [torch.empty_like(grad_output) for _ in range(seq_world_size)]
        tensor_list[seq_parallel_world_rank] = grad_output
        torch.distributed.all_gather(tensor_list, grad_output, group=get_sequence_parallel_group()) 

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=1).contiguous()
        # print(f'after rank {seq_parallel_world_rank} {output[0, :, :6, :6]}')
        # exit(0)

        return output


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2, # head
        gather_idx: int = 1, # s
    ) -> None:

        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias: Tensor, *args: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        if self.training:
            # in shape : [b, s/p+1, n_head, hn]
            query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
            key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
            value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)
            # out shape : [b, s+p, np, hn]
            
            # Merge each rank's global token embedding into one 
            # q, k, v: [b, s+p, np, hn] -> [b, s+1, np, hn]
            query_layer = merge_global_token(query_layer, merge_dim=1)
            key_layer = merge_global_token(key_layer, merge_dim=1)
            value_layer = merge_global_token(value_layer, merge_dim=1)

            # allgather attention bias
            # in shape : [b, s/p+1, s+1, attn_bias_dim]
            attn_bias_layer = _SeqGather.apply(attn_bias, self.gather_idx)
            # out shape : [b, s+p, s+1, attn_bias_dim]

            # Merge each rank's global token embedding into one 
            # [b, s+p, s+1, attn_bias_dim] -> [b, s+1, s+1, attn_bias_dim]
            attn_bias_layer = merge_global_token(attn_bias_layer, merge_dim=1)
        else:
            query_layer = query
            key_layer = key
            value_layer = value
            attn_bias_layer = attn_bias
    
        # [b, s+1, hp]
        context_layer = self.local_attn(query_layer, key_layer, value_layer, attn_bias_layer, *args)
        
        if self.training:
            # [b, s+1, hp] -> [b, s+p, hp]
            context_layer = extend_global_token(context_layer, extend_dim=1)

            # [b, s+p, hp] -> [b, s/p+1, h]
            output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        else:
            output = context_layer

        # out e.g., [b, s/p+1, h]
        return output


class DistributedAttentionNoMerge(torch.nn.Module):
    """Distributed attention for node-level tasks (no global-token merge).

    When ``overlap_comm=True`` and the following conditions hold:
      - ``self.training`` is True
      - ``attn_type == "sparse"``
      - ``edge_index`` is a plain 2-D ``Tensor`` (not a list or dict)
      - ``attn_bias`` is ``None``
      - CUDA is available
      - ``local_attn`` exposes ``sparse_score_phase`` / ``sparse_aggregate_phase``

    the layer overlaps the Value tensor's AllToAll communication with the
    Q·K score computation on a separate CUDA stream, hiding communication
    latency behind useful GPU compute.

    In all other situations the original synchronous path is used.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        overlap_comm (bool): enable computation-communication overlap
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,  # head
        gather_idx: int = 1,  # s
        overlap_comm: bool = False,
    ) -> None:

        super(DistributedAttentionNoMerge, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        # Overlap is only useful on CUDA and when the local_attn supports
        # the two-phase sparse attention interface.
        self.overlap_comm = (
            bool(overlap_comm)
            and torch.cuda.is_available()
            and hasattr(local_attention, "sparse_score_phase")
            and hasattr(local_attention, "sparse_aggregate_phase")
        )
        self._comm_stream: torch.cuda.Stream | None = None

    def _get_comm_stream(self) -> torch.cuda.Stream:
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream()
        return self._comm_stream

    def _can_overlap(self, attn_bias, edge_index, attn_type) -> bool:
        """Check whether the overlap path is applicable for this call."""
        if not self.overlap_comm:
            return False
        if not self.training:
            return False
        if str(attn_type).lower() != "sparse":
            return False
        # Only the plain-tensor edge_index path supports overlap; list and
        # dict cases have head-group or streaming logic that is harder to
        # split.
        if not isinstance(edge_index, Tensor) or edge_index.dim() != 2:
            return False
        # attn_bias AllToAll would also need to go on comm_stream; skip
        # overlap when it is present (uncommon in the fullgraph_sp path).
        if attn_bias is not None:
            return False
        return True

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias: Tensor, edge_index: Tensor, attn_type, *args: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        if not self.training:
            # Evaluation path – no communication.
            return self.local_attn(query, key, value, attn_bias, edge_index, attn_type, *args)

        # ---- AllToAll Q, K (common to both paths) ----
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)

        if self._can_overlap(attn_bias, edge_index, attn_type):
            # === Overlapped path ===
            # Phase 1: Launch V AllToAll on a dedicated comm stream so it
            #          overlaps with the score computation below.
            comm_stream = self._get_comm_stream()
            default_stream = torch.cuda.current_stream()

            # Record that Q/K AllToAlls (on default_stream) are done.
            qk_done = torch.cuda.Event()
            qk_done.record(default_stream)

            with torch.cuda.stream(comm_stream):
                # Make sure Q/K AllToAlls finished (NCCL serializes on the
                # same communicator anyway, but this event ensures CUDA
                # ordering correctness for the producer-consumer chain).
                comm_stream.wait_event(qk_done)
                value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)
                v_done = torch.cuda.Event()
                v_done.record(comm_stream)

            # Phase 2: Score computation on the default stream (overlaps
            #          with V AllToAll on comm_stream).
            score_ctx = self.local_attn.sparse_score_phase(
                key_layer, query_layer, edge_index, None,
            )

            # Phase 3: Wait for V to arrive, then aggregate.
            default_stream.wait_event(v_done)
            value_layer.record_stream(default_stream)
            context_layer = self.local_attn.sparse_aggregate_phase(
                score_ctx, value_layer,
            )
            # Re-wrap to [b, s, hp] to match the non-overlap output shape.
            batch_size = query_layer.size(0)
            s_len = query_layer.size(1)
            context_layer = context_layer.view(batch_size, s_len, -1)
        else:
            # === Original synchronous path ===
            value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

            if attn_bias is not None:
                attn_bias_layer = _SeqAllToAll.apply(self.spg, attn_bias, 3, 1)
                attn_bias_layer = extend_global_token0(attn_bias_layer, extend_dim=2)
            else:
                attn_bias_layer = attn_bias

            context_layer = self.local_attn(
                query_layer, key_layer, value_layer,
                attn_bias_layer, edge_index, attn_type, *args,
            )

        context_layer = copy_global_token0(context_layer, extend_dim=1)
        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        return output
