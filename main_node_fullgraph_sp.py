"""Full-Graph Node-Level Training with Sequence Parallel (SP).

Architecture:
  Each SP rank holds a slice of the N graph nodes (indices [rank_start, rank_end)).
  Inside DistributedAttentionNodeLevel, _SeqAllToAll scatter-heads/gather-seq
  transforms each rank's [b, N/P, H, hn] into [b, N, H/P, hn] so CoreAttention
  sees ALL N nodes with H/P heads per head partition.  After the second all-to-all
  the result returns to [b, N/P, H, hn].  This means sparse attention across the
  full edge_index is correct without any additional change to the attention code.

Training:
  - edge_index (full graph) is built on rank-0 and broadcast.
  - Each rank inputs x_local = feature[rank_start:rank_end].
  - Loss is computed only on this rank's train nodes (train_idx ∩ [rank_start, rank_end)).
  - Gradients are all-reduced across ranks.

Evaluation:
  - Full-graph eval runs model.train() so the all-to-all path is active.
  - Each rank computes predictions for its local nodes.
  - Predictions are all-gathered on rank-0 for metric computation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
import os
import resource
import sys
import time
import random
import contextlib
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist
from torch_geometric.utils import coalesce

from models.gt_dist_node_level import GT
from models.graphormer_dist_node_level import Graphormer
from utils.lr import PolynomialDecayLR
import argparse

from utils.parser_node_level import (
    add_node_common_args,
    add_node_fullgraph_sp_args,
    normalize_main_node_fullgraph_sp_args,
)
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import sync_params_and_buffers
from gt_sp.utils import (
    random_split_idx,
    fixed_random_seed,
    build_head_hop_edges,
    _merge_edge_index_list,
    fix_edge_index,
    adjust_edge_index_nomerge,
    resolve_random_walk_device,
)
from utils.split_utils import load_default_split


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def _resolve_amp_dtype(args):
    amp_mode = getattr(args, "amp_dtype", "none")
    if amp_mode == "none" or not torch.cuda.is_available():
        return None
    if amp_mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if args.rank == 0:
            print("[amp] bf16 is not supported on this GPU; falling back to fp16.")
        return torch.float16
    if amp_mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_mode}")


def _autocast_context(device: str, amp_dtype):
    if amp_dtype is None or not device.startswith("cuda"):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_process_rss_mib() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.readline().strip().split()
        if len(fields) >= 2:
            rss_pages = int(fields[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return rss_pages * page_size / (1024 ** 2)
    except Exception:
        pass
    return _get_process_peak_rss_mib()


def _get_process_peak_rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 ** 2)
    return usage / 1024.0


def _maybe_cuda_empty_cache(args) -> None:
    if not getattr(args, "cuda_empty_cache", False):
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_bidirected_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index_bi = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return coalesce(edge_index_bi, num_nodes=num_nodes)


def _load_default_split(dataset_name: str, root_dir: str, wait_for_rank0: bool = False):
    return load_default_split(
        dataset_name,
        root_dir,
        dist_module=dist,
        wait_for_rank0=wait_for_rank0,
    )


def _load_data(args):
    data_path = args.dataset_dir + args.dataset
    feature = torch.load(data_path + "/x.pt")
    y = torch.load(data_path + "/y.pt")
    edge_index_global = torch.load(data_path + "/edge_index.pt")
    N = feature.shape[0]

    if args.to_bidirected:
        edge_index_global = _to_bidirected_edge_index(edge_index_global, num_nodes=N)

    if args.dataset == "pokec":
        y = torch.clamp(y, min=0)

    # 始终尝试数据集默认划分，找不到时回退随机 60/20/20 分割
    split_idx = _load_default_split(args.dataset, args.dataset_dir, wait_for_rank0=True)
    if split_idx is None:
        if args.rank == 0:
            print("[split] No default split found, falling back to random 60/20/20 split.")
        # Some versions of random_split_idx use positional, some use keyword arguments.
        split_idx = random_split_idx(y, 0.6, 0.2, 0.2, args.seed)
    else:
        if args.rank == 0:
            print("[split] Loaded official dataset split.")

    if args.rank == 0:
        print(args)
        if args.to_bidirected:
            print("[graph] Converted edge_index to bidirected after loading.")
        print(f"Dataset loaded: N={N:,}  E={edge_index_global.shape[1]:,}")
        print(f"  train={split_idx['train'].shape[0]:,}  "
              f"val={split_idx['valid'].shape[0]:,}  "
              f"test={split_idx['test'].shape[0]:,}")
    return feature, y, edge_index_global, N, split_idx


def _build_model(args, feature, y, device):
    out_dim = int(y.max().item()) + 1
    common = dict(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=feature.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=out_dim,
        attn_bias_dim=args.attn_bias_dim,
        dropout_rate=args.dropout_rate,
        input_dropout_rate=args.input_dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        ffn_dim=args.ffn_dim,
    )
    if args.model == "graphormer":
        model = Graphormer(
            **common,
            num_global_node=args.num_global_node,
        ).to(device)
        if getattr(args, "sparse_query_chunk_size", 0) > 0:
            model.set_sparse_attention_query_chunk_size(args.sparse_query_chunk_size)
        if getattr(args, "activation_checkpoint", False):
            model.set_activation_checkpoint(True)
    elif args.model == "gt":
        model = GT(
            **common,
            num_global_node=0,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model


def _build_and_broadcast_edges(args, edge_index_global, N, device, rw_device, sp_group,
                                sp_src_rank, sp_rank, nodes_per_rank, edge_seed=None,
                                edge_budget_state=None):
    """Build sparse-attention edges on src_rank and broadcast to all ranks."""
    if sp_rank == sp_src_rank:
        seed_ctx = (
            _fixed_torch_cpu_seed(edge_seed)
            if edge_seed is not None and torch.device(rw_device).type == "cpu"
            else contextlib.nullcontext()
        )
        with seed_ctx:
            parts = []
            if getattr(args, "include_self_loops", 0):
                self_loop = torch.arange(N, device=device, dtype=torch.long)
                self_loop = torch.stack([self_loop, self_loop], dim=0)
                parts.append(self_loop)
            real_edges = None
            if _use_real_edges_for_state(args, edge_budget_state):
                real_edges = edge_index_global.to(device)
            rw_heads = None
            if _use_rw_edges(args, edge_budget_state):
                rw_heads = build_head_hop_edges(
                    # Keep the original graph tensor stable across epochs.
                    # _get_random_walk_graph() caches by the input tensor identity;
                    # passing a fresh GPU copy here makes rank0 accumulate a new
                    # cached CSR every epoch when rw_device is CUDA.
                    edge_index=edge_index_global,
                    num_nodes=N,
                    num_heads=args.num_heads,
                    num_groups=1,
                    device=rw_device,
                    walk_length=getattr(args, "head_hop_walk_length", 4),
                    walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
                )
                if isinstance(rw_heads, list):
                    rw_heads = _merge_edge_index_list(rw_heads)
                if rw_heads.device != torch.device(device):
                    rw_heads = rw_heads.to(device)
            real_edges, rw_heads = _sample_random_edge_blocks_with_state(
                args, real_edges, rw_heads, edge_seed=edge_seed, edge_budget_state=edge_budget_state
            )
            if real_edges is not None:
                parts.append(real_edges)
            if rw_heads is not None:
                parts.append(rw_heads)
            merged = _merge_edge_index_list(parts)
            if merged is None:
                merged = edge_index_global.new_zeros((2, 0), dtype=torch.long).to(device)
            if args.model == "graphormer" and args.num_global_node > 0:
                merged = fix_edge_index(merged, N)
                merged = adjust_edge_index_nomerge(merged, nodes_per_rank)
        size_t = torch.tensor([merged.shape[1]], device=device, dtype=torch.long)
    else:
        size_t = torch.empty(1, device=device, dtype=torch.long)
        merged = None

    if dist.is_initialized():
        dist.broadcast(size_t, sp_src_rank, group=sp_group)
        if sp_rank != sp_src_rank:
            merged = torch.empty((2, int(size_t.item())), device=device,
                                 dtype=torch.long)
        dist.broadcast(merged, sp_src_rank, group=sp_group)

    return merged


def _pin_cpu_tensor(tensor):
    if tensor.device.type == "cpu" and torch.cuda.is_available():
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
    return tensor


def _random_block_sampling_enabled(args) -> bool:
    return bool(getattr(args, "random_edge_blocks", False))


def _adaptive_edge_budget_enabled(args) -> bool:
    return bool(getattr(args, "adaptive_edge_budget", False))


def _get_real_edge_budget(args, edge_budget_state=None) -> int:
    if edge_budget_state is not None and "real_edges_per_query" in edge_budget_state:
        return int(edge_budget_state["real_edges_per_query"])
    return int(getattr(args, "real_edges_per_query", 0))


def _get_rw_edge_budget(args, edge_budget_state=None) -> int:
    if edge_budget_state is not None and "rw_edges_per_query" in edge_budget_state:
        return int(edge_budget_state["rw_edges_per_query"])
    return int(getattr(args, "rw_edges_per_query", 0))


def _use_real_edges(args) -> bool:
    if _random_block_sampling_enabled(args) and int(getattr(args, "real_edges_per_query", 0)) > 0:
        return True
    return bool(getattr(args, "include_real_edges", 0))


def _use_real_edges_for_state(args, edge_budget_state=None) -> bool:
    if _random_block_sampling_enabled(args):
        return _get_real_edge_budget(args, edge_budget_state) > 0
    return bool(getattr(args, "include_real_edges", 0))


def _use_rw_edges(args, edge_budget_state=None) -> bool:
    if int(getattr(args, "head_hop_walks_per_node", 0)) <= 0:
        return False
    if _random_block_sampling_enabled(args):
        return _get_rw_edge_budget(args, edge_budget_state) > 0
    return True


def _edge_block_seed(args, edge_seed, offset: int) -> int:
    base = getattr(args, "seed", 0) if edge_seed is None else int(edge_seed)
    return int(base) + int(offset)


def _sample_edges_per_query_random(edge_index, max_edges_per_query: int, seed: int):
    if edge_index is None or edge_index.numel() == 0 or int(max_edges_per_query) <= 0:
        return edge_index

    max_edges_per_query = int(max_edges_per_query)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    if src.numel() <= max_edges_per_query:
        return edge_index

    # Build a per-edge pseudo-random key and sort by (dst, rand_key).
    scale = 1 << 31
    rand_key = torch.remainder(
        src * 1103515245 + dst * 214013 + int(seed) * 2654435761 + 12345,
        scale,
    )
    composite = dst * scale + rand_key
    perm = torch.argsort(composite)
    dst_sorted = dst[perm]

    arange = torch.arange(dst_sorted.numel(), device=dst_sorted.device, dtype=torch.long)
    is_new = torch.ones_like(dst_sorted, dtype=torch.bool)
    is_new[1:] = dst_sorted[1:] != dst_sorted[:-1]
    group_starts = torch.where(is_new, arange, torch.zeros_like(arange))
    group_starts = torch.cummax(group_starts, dim=0).values
    pos_in_group = arange - group_starts
    keep = pos_in_group < max_edges_per_query
    return edge_index[:, perm[keep]]


def _sample_random_edge_blocks(args, real_edges, rw_edges, edge_seed=None):
    if not _random_block_sampling_enabled(args):
        return real_edges, rw_edges

    if real_edges is not None and int(getattr(args, "real_edges_per_query", 0)) > 0:
        real_edges = _sample_edges_per_query_random(
            real_edges,
            getattr(args, "real_edges_per_query", 0),
            _edge_block_seed(args, edge_seed, 17),
        )

    if rw_edges is not None and int(getattr(args, "rw_edges_per_query", 0)) > 0:
        rw_edges = _sample_edges_per_query_random(
            rw_edges,
            getattr(args, "rw_edges_per_query", 0),
            _edge_block_seed(args, edge_seed, 37),
        )

    return real_edges, rw_edges


def _sample_random_edge_blocks_with_state(args, real_edges, rw_edges, edge_seed=None, edge_budget_state=None):
    if not _random_block_sampling_enabled(args):
        return real_edges, rw_edges

    real_budget = _get_real_edge_budget(args, edge_budget_state)
    rw_budget = _get_rw_edge_budget(args, edge_budget_state)

    if real_edges is not None and real_budget > 0:
        real_edges = _sample_edges_per_query_random(
            real_edges,
            real_budget,
            _edge_block_seed(args, edge_seed, 17),
        )

    if rw_edges is not None and rw_budget > 0:
        rw_edges = _sample_edges_per_query_random(
            rw_edges,
            rw_budget,
            _edge_block_seed(args, edge_seed, 37),
        )

    return real_edges, rw_edges


class _AdaptiveEdgeBudgetController:
    def __init__(self, args) -> None:
        self.enabled = _adaptive_edge_budget_enabled(args)
        self.block_size = max(1, int(getattr(args, "adaptive_edge_budget_block_size", 2)))
        self.max_real = max(0, int(getattr(args, "real_edges_per_query", 0)))
        self.max_rw = max(0, int(getattr(args, "rw_edges_per_query", 0)))
        self.warmup_epochs = max(0, int(getattr(args, "adaptive_edge_budget_warmup_epochs", 0)))
        self.gain_threshold = float(getattr(args, "adaptive_edge_budget_gain_threshold", 0.0))
        self.patience = max(3, int(getattr(args, "adaptive_edge_budget_patience", 1)))
        self.bad_rounds = 0
        self.frozen = not self.enabled
        if not self.enabled:
            self.real_budget = self.max_real
            self.rw_budget = self.max_rw
            return

        self.real_budget = min(self.block_size, self.max_real) if self.max_real > 0 else 0
        self.rw_budget = min(self.block_size, self.max_rw) if self.max_rw > 0 else 0
        if self.warmup_epochs <= 0:
            self.frozen = True

    def current_state(self):
        return {
            "real_edges_per_query": int(self.real_budget),
            "rw_edges_per_query": int(self.rw_budget),
        }

    def candidate_states(self):
        out = {}
        if self.real_budget < self.max_real:
            out["real"] = {
                "real_edges_per_query": min(self.max_real, self.real_budget + self.block_size),
                "rw_edges_per_query": self.rw_budget,
            }
        if self.rw_budget < self.max_rw:
            out["rw"] = {
                "real_edges_per_query": self.real_budget,
                "rw_edges_per_query": min(self.max_rw, self.rw_budget + self.block_size),
            }
        return out

    def update(self, choice, best_gain, next_state=None):
        if next_state is not None and choice is not None:
            self.real_budget = int(next_state["real_edges_per_query"])
            self.rw_budget = int(next_state["rw_edges_per_query"])
            self.bad_rounds = 0
            return
        if best_gain <= self.gain_threshold:
            self.bad_rounds += 1
            if self.bad_rounds >= self.patience:
                self.frozen = True


def _select_probe_nodes(split_idx, probe_size: int, seed: int):
    probe_size = max(0, int(probe_size))
    if probe_size <= 0:
        return None
    valid_idx = split_idx.get("valid")
    if valid_idx is None or valid_idx.numel() == 0:
        return None
    valid_idx = valid_idx.to(torch.long).cpu()
    if valid_idx.numel() <= probe_size:
        return valid_idx
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(valid_idx.numel(), generator=gen)[:probe_size]
    return valid_idx.index_select(0, perm).to(torch.long)


def _resolve_adaptive_edge_budget_args(args, split_idx):
    if not _adaptive_edge_budget_enabled(args):
        return
    args.random_edge_blocks = True
    if getattr(args, "adaptive_edge_budget_probe_size", 0) <= 0:
        valid_idx = split_idx.get("valid")
        valid_n = int(valid_idx.numel()) if valid_idx is not None else 0
        args.adaptive_edge_budget_probe_size = min(512, valid_n) if valid_n > 0 else 0
    if getattr(args, "adaptive_edge_budget_block_size", 0) <= 0:
        args.adaptive_edge_budget_block_size = 2
    if getattr(args, "adaptive_edge_budget_warmup_epochs", 0) <= 0:
        args.adaptive_edge_budget_warmup_epochs = max(1, min(5, max(1, int(args.epochs) // 10)))
    if getattr(args, "adaptive_edge_budget_patience", 0) <= 0:
        args.adaptive_edge_budget_patience = 1


def _edge_count(attn_edges) -> int:
    if isinstance(attn_edges, dict):
        return int(attn_edges["src"].numel())
    return int(attn_edges.size(1))


def _probe_loss_sp(args, model, x_local, local_y, local_probe_idx, edge_index_probe,
                   device, amp_dtype, sp_group, sp_world_size):
    was_training = model.training
    model.train()
    drop_states = _set_dropout_eval(model)

    loss_sum = torch.zeros(1, device=device, dtype=torch.float32)
    count = torch.zeros(1, device=device, dtype=torch.long)
    with torch.no_grad():
        with _autocast_context(device, amp_dtype):
            out_local = model(x_local, None, edge_index_probe, attn_type=args.attn_type)
        out_rows = int(out_local.size(0))
        if local_probe_idx is not None and local_probe_idx.numel() > 0:
            valid_probe = (local_probe_idx >= 0) & (local_probe_idx < out_rows)
            probe_idx_eff = local_probe_idx[valid_probe]
            if probe_idx_eff.numel() > 0:
                local_y_eff = local_y[:out_rows]
                loss_sum = F.nll_loss(
                    out_local.index_select(0, probe_idx_eff),
                    local_y_eff.index_select(0, probe_idx_eff).long(),
                    reduction="sum",
                ).to(torch.float32).view(1)
                count = torch.tensor([probe_idx_eff.numel()], device=device, dtype=torch.long)

    if sp_world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=sp_group)

    _restore_dropout(drop_states)
    if not was_training:
        model.eval()

    mean_loss = float(loss_sum.item() / max(int(count.item()), 1))
    del out_local
    _maybe_cuda_empty_cache(args)
    return mean_loss


def _maybe_update_edge_budget(
    args,
    controller,
    epoch: int,
    model,
    x_local,
    local_y,
    local_probe_idx,
    edge_index_global,
    N,
    device,
    rw_device,
    sp_group,
    sp_src_rank,
    sp_rank,
    local_N,
    amp_dtype,
    sp_world_size,
):
    if (not controller.enabled) or controller.frozen or epoch > controller.warmup_epochs:
        return
    if local_probe_idx is None:
        controller.frozen = True
        return

    probe_seed = int(getattr(args, "seed", 0)) + 100000 + int(epoch)
    base_state = controller.current_state()
    base_edges = _build_attention_edges(
        args, edge_index_global, N, device, rw_device,
        sp_group, sp_src_rank, sp_rank, local_N,
        edge_seed=probe_seed,
        edge_budget_state=base_state,
    )
    base_loss = _probe_loss_sp(
        args, model, x_local, local_y, local_probe_idx, base_edges,
        device, amp_dtype, sp_group, sp_world_size,
    )
    base_count = _edge_count(base_edges)
    del base_edges

    best_kind = None
    best_gain = float("-inf")
    best_state = None
    best_loss = None
    best_count = None
    for kind, cand_state in controller.candidate_states().items():
        cand_edges = _build_attention_edges(
            args, edge_index_global, N, device, rw_device,
            sp_group, sp_src_rank, sp_rank, local_N,
            edge_seed=probe_seed,
            edge_budget_state=cand_state,
        )
        cand_loss = _probe_loss_sp(
            args, model, x_local, local_y, local_probe_idx, cand_edges,
            device, amp_dtype, sp_group, sp_world_size,
        )
        cand_count = _edge_count(cand_edges)
        delta_edges = max(cand_count - base_count, 1)
        gain = (base_loss - cand_loss) / float(delta_edges)
        if gain > best_gain:
            best_kind = kind
            best_gain = gain
            best_state = cand_state
            best_loss = cand_loss
            best_count = cand_count
        del cand_edges

    controller.update(best_kind if best_gain > controller.gain_threshold else None,
                      best_gain, best_state if best_gain > controller.gain_threshold else None)

    if args.rank == 0:
        print(
            f"  ↳ BudgetCtrl epoch={epoch} probe_loss={base_loss:.4f} "
            f"edges={base_count} choice={best_kind} gain={best_gain:.6e} "
            f"next_real={controller.real_budget} next_rw={controller.rw_budget}"
            + (
                f" cand_loss={best_loss:.4f} cand_edges={best_count}"
                if best_loss is not None and best_count is not None
                else ""
            )
        )


@contextlib.contextmanager
def _fixed_torch_cpu_seed(seed: int):
    """Temporarily set CPU torch RNG without touching CUDA RNG state."""
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)


class _CpuRandomWalkEdgePrefetcher:
    """Build next epoch's CPU random-walk edges while the current epoch trains."""

    def __init__(self, enabled: bool) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self._future = None
        self._epoch = None

    @property
    def enabled(self) -> bool:
        return self._executor is not None

    def submit(self, epoch: int, fn) -> None:
        if self._executor is None or epoch <= 0 or self._future is not None:
            return
        self._epoch = int(epoch)
        self._future = self._executor.submit(fn)

    def pop(self, epoch: int):
        if self._future is None or self._epoch != int(epoch):
            return None
        result = self._future.result()
        self._future = None
        self._epoch = None
        return result

    def close(self) -> None:
        if self._future is not None:
            self._future.result()
            self._future = None
            self._epoch = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def _build_merged_edges(args, edge_index_global, N, final_device, rw_device, nodes_per_rank,
                        edge_seed=None, edge_budget_state=None):
    parts = []
    final_torch_device = torch.device(final_device)
    if getattr(args, "include_self_loops", 0):
        self_loop = torch.arange(N, device=final_torch_device, dtype=torch.long)
        self_loop = torch.stack([self_loop, self_loop], dim=0)
        parts.append(self_loop)
    real_edges = None
    if _use_real_edges_for_state(args, edge_budget_state):
        real_edges = edge_index_global.to(final_torch_device)
    rw_heads = None
    if _use_rw_edges(args, edge_budget_state):
        rw_heads = build_head_hop_edges(
            # See _build_and_broadcast_edges(): keep the source edge tensor stable
            # so the random-walk graph cache can be reused across epochs.
            edge_index=edge_index_global,
            num_nodes=N,
            num_heads=args.num_heads,
            num_groups=1,
            device=rw_device,
            walk_length=getattr(args, "head_hop_walk_length", 4),
            walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
        )
        if isinstance(rw_heads, list):
            rw_heads = _merge_edge_index_list(rw_heads)
        if rw_heads is not None:
            if rw_heads.device != final_torch_device:
                rw_heads = rw_heads.to(final_torch_device)
    real_edges, rw_heads = _sample_random_edge_blocks_with_state(
        args, real_edges, rw_heads, edge_seed=edge_seed, edge_budget_state=edge_budget_state
    )
    if real_edges is not None:
        parts.append(real_edges)
    if rw_heads is not None:
        parts.append(rw_heads)
    merged = _merge_edge_index_list(parts)
    if merged is None:
        merged = edge_index_global.new_zeros((2, 0), dtype=torch.long).to(final_torch_device)
    if args.model == "graphormer" and args.num_global_node > 0:
        merged = fix_edge_index(merged, N)
        merged = adjust_edge_index_nomerge(merged, nodes_per_rank)
    return merged


def _finalize_attention_edges(args, merged):
    if args.attn_type == "sparse" and getattr(args, "sparse_query_chunk_size", 0) > 0:
        return _pack_chunked_edges(merged, args.sparse_query_chunk_size, mode="gpu_chunk")
    return merged


def _build_prefetched_cpu_edges(args, edge_index_global, N, rw_device, nodes_per_rank, edge_seed,
                                edge_budget_state=None):
    seed_ctx = _fixed_torch_cpu_seed(edge_seed) if edge_seed is not None else contextlib.nullcontext()
    with seed_ctx:
        merged_cpu = _build_merged_edges(
            args, edge_index_global, N, "cpu", rw_device, nodes_per_rank,
            edge_seed=edge_seed, edge_budget_state=edge_budget_state
        )
    if merged_cpu.device.type == "cpu":
        merged_cpu = _pin_cpu_tensor(merged_cpu.contiguous())
    return merged_cpu


def _broadcast_prefetched_edges(merged_cpu, device, sp_group, sp_src_rank, sp_rank):
    if sp_rank == sp_src_rank:
        merged = merged_cpu.to(device=device, dtype=torch.long, non_blocking=(merged_cpu.device.type == "cpu"))
        size_t = torch.tensor([merged.shape[1]], device=device, dtype=torch.long)
    else:
        size_t = torch.empty(1, device=device, dtype=torch.long)
        merged = None

    if dist.is_initialized():
        dist.broadcast(size_t, sp_src_rank, group=sp_group)
        if sp_rank != sp_src_rank:
            merged = torch.empty((2, int(size_t.item())), device=device, dtype=torch.long)
        dist.broadcast(merged, sp_src_rank, group=sp_group)
    return merged


def _pack_chunked_edges(merged, chunk_size, mode):
    chunk = max(int(chunk_size), 1)
    edge_store = merged.detach()

    if mode == "cpu_stream":
        edge_store = edge_store.cpu()
    elif mode != "gpu_chunk":
        raise ValueError(f"Unsupported chunked edge payload mode: {mode}")

    if edge_store.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=edge_store.device)
        offsets = torch.zeros(1, dtype=torch.long, device=edge_store.device)
        if mode == "cpu_stream":
            empty = _pin_cpu_tensor(empty)
            offsets = _pin_cpu_tensor(offsets)
        return {
            "mode": mode,
            "src": empty,
            "dst": empty.clone(),
            "offsets": offsets,
            "chunk_size": chunk,
        }

    edge_index = edge_store[:2].to(torch.long).contiguous()
    edge_hops = edge_store[2].to(torch.long).contiguous() if edge_store.size(0) == 3 else None

    order = torch.argsort(edge_index[1])
    src_sorted = edge_index[0].index_select(0, order).contiguous()
    dst_sorted = edge_index[1].index_select(0, order).contiguous()
    hop_sorted = edge_hops.index_select(0, order).contiguous() if edge_hops is not None else None

    max_dst = int(dst_sorted.max().item()) if dst_sorted.numel() > 0 else -1
    num_chunks = max((max_dst + chunk) // chunk, 0)
    offsets = torch.zeros(num_chunks + 1, dtype=torch.long, device=dst_sorted.device)
    if dst_sorted.numel() > 0 and num_chunks > 0:
        dst_bins = torch.div(dst_sorted, chunk, rounding_mode="floor")
        counts = torch.bincount(dst_bins, minlength=num_chunks)
        offsets[1:] = counts.cumsum(dim=0)

    if mode == "cpu_stream":
        src_sorted = _pin_cpu_tensor(src_sorted)
        dst_sorted = _pin_cpu_tensor(dst_sorted)
        offsets = _pin_cpu_tensor(offsets)
        if hop_sorted is not None:
            hop_sorted = _pin_cpu_tensor(hop_sorted)

    payload = {
        "mode": mode,
        "src": src_sorted,
        "dst": dst_sorted,
        "offsets": offsets,
        "chunk_size": chunk,
    }
    if hop_sorted is not None:
        payload["hops"] = hop_sorted
    return payload


def _build_streaming_edges(args, edge_index_global, N, rw_device, nodes_per_rank, edge_seed=None,
                           edge_budget_state=None):
    seed_ctx = fixed_random_seed(edge_seed) if edge_seed is not None else contextlib.nullcontext()
    with seed_ctx:
        merged = _build_merged_edges(
            args, edge_index_global, N, "cpu", rw_device, nodes_per_rank,
            edge_seed=edge_seed, edge_budget_state=edge_budget_state
        )
    return _pack_chunked_edges(merged, args.sparse_query_chunk_size, mode="cpu_stream")


def _build_attention_edges(args, edge_index_global, N, device, rw_device, sp_group,
                           sp_src_rank, sp_rank, nodes_per_rank, edge_seed=None,
                           edge_budget_state=None):
    if getattr(args, "stream_edges_from_cpu", False):
        return _build_streaming_edges(
            args,
            edge_index_global,
            N,
            rw_device,
            nodes_per_rank,
            edge_seed=edge_seed,
            edge_budget_state=edge_budget_state,
        )
    merged = _build_and_broadcast_edges(
        args,
        edge_index_global,
        N,
        device,
        rw_device,
        sp_group,
        sp_src_rank,
        sp_rank,
        nodes_per_rank,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
    )
    return _finalize_attention_edges(args, merged)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (using model.train() so all-to-all is active across all ranks)
# ─────────────────────────────────────────────────────────────────────────────

def _set_dropout_eval(model):
    states = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            states.append((m, m.training))
            m.eval()
    return states

def _restore_dropout(states):
    for m, st in states:
        m.train(st)

def _eval_sp(args, model, feature, y, split_idx, edge_index_global, N,
             device, rw_device, sp_group, sp_src_rank, sp_rank, sp_world_size,
             rank_start, rank_end, local_N, amp_dtype=None, cached_edge_index=None,
             edge_budget_state=None):
    """Full-graph SP evaluation.

    Runs in model.train() mode so _SeqAllToAll is triggered and each rank
    computes its local nodes with full graph attention context.
    Predictions are all-gathered on rank-0.
    """
    # Build eval edges (fixed seed, rank-0 builds and broadcasts)
    if cached_edge_index is None:
        with fixed_random_seed(args.seed):
            edge_index_eval = _build_attention_edges(
                args, edge_index_global, N, device, rw_device,
                sp_group, sp_src_rank, sp_rank, local_N,
                edge_seed=args.seed if getattr(args, "stream_edges_from_cpu", False) else None,
                edge_budget_state=edge_budget_state,
            )
    else:
        edge_index_eval = cached_edge_index

    # Forward in train mode so all-to-all fires
    was_training = model.training
    model.train()
    drop_states = _set_dropout_eval(model)
    with torch.no_grad():
        x_local = feature[rank_start:rank_end].float().to(device)
        with _autocast_context(device, amp_dtype):
            out_local = model(x_local, None, edge_index_eval, attn_type=args.attn_type)
        pred_local = out_local.argmax(dim=1)  # [local_N] — keep on device for NCCL

    _restore_dropout(drop_states)
    if not was_training:
        model.eval()

    # All-gather predictions (NCCL requires CUDA tensors throughout)
    if sp_world_size > 1:
        local_pred_len = torch.tensor([pred_local.size(0)], dtype=torch.long, device=device)
        len_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(sp_world_size)]
        dist.all_gather(len_list, local_pred_len, group=sp_group)
        pred_lens = [int(t.item()) for t in len_list]
        max_pred_len = max(pred_lens) if pred_lens else 0

        if pred_local.size(0) < max_pred_len:
            padded = torch.zeros(max_pred_len, dtype=torch.long, device=device)
            padded[:pred_local.size(0)] = pred_local
            pred_local = padded

        gather_list = [torch.zeros(max_pred_len, dtype=torch.long, device=device)
                       for _ in range(sp_world_size)]
        dist.all_gather(gather_list, pred_local, group=sp_group)
        pred_chunks = [gather_list[i][:pred_lens[i]] for i in range(sp_world_size)]
        pred_global = torch.cat(pred_chunks, dim=0)[:N].cpu()
    else:
        pred_global = pred_local.cpu()

    result = None
    if args.rank == 0:
        valid_n = min(int(pred_global.size(0)), int(N))
        pred_global = pred_global[:valid_n]
        y_cpu = y[:valid_n].cpu().view(-1)
        accs = {}
        for sname, idx in split_idx.items():
            idx_valid = idx[idx < valid_n]
            correct = (pred_global[idx_valid] == y_cpu[idx_valid]).sum().item()
            accs[sname] = float(correct) / max(1, len(idx_valid))
        result = accs

    del pred_local, pred_global, out_local, x_local
    if cached_edge_index is None:
        del edge_index_eval
    _maybe_cuda_empty_cache(args)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full-Graph Node-Level Training with Sequence Parallel."
    )
    add_node_common_args(
        parser,
        defaults={
            "peak_lr": 1e-3,
            "warmup_updates": 20,
            "sequence_parallel_size": 1,
            "num_global_node": 1,
            "amp_dtype": "none",
        },
    )
    add_node_fullgraph_sp_args(parser)
    args = normalize_main_node_fullgraph_sp_args(parser.parse_args())
    if args.stream_edges_from_cpu and args.attn_type != "sparse":
        raise ValueError("--stream_edges_from_cpu currently requires --attn_type sparse.")
    if args.stream_edges_from_cpu and args.sparse_query_chunk_size <= 0:
        raise ValueError("--stream_edges_from_cpu requires --sparse_query_chunk_size > 0.")
    if _adaptive_edge_budget_enabled(args) and args.random_walk_prefetch:
        raise ValueError("--adaptive_edge_budget is not compatible with --random_walk_prefetch.")

    # ── Distributed init ────────────────────────────────────────────────────
    initialize_distributed(args)

    sp_initialized = sequence_parallel_is_initialized()
    sp_world_size = get_sequence_parallel_world_size() if sp_initialized else 1
    sp_rank = get_sequence_parallel_rank() if sp_initialized else 0
    sp_src_rank = get_sequence_parallel_src_rank() if sp_initialized else 0
    sp_group = get_sequence_parallel_group() if sp_initialized else None

    device = _resolve_device()
    amp_dtype = _resolve_amp_dtype(args)
    rw_device = resolve_random_walk_device(args, device)
    _set_seed(args.seed)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    feature, y, edge_index_global, N, split_idx = _load_data(args)
    feature = feature.float()
    _resolve_adaptive_edge_budget_args(args, split_idx)

    # ── Per-rank node slice ───────────────────────────────────────────────
    # _SeqAllToAll requires equal seq-dim length on all ranks: pad N to multiple of P
    pad_N = ((N + sp_world_size - 1) // sp_world_size) * sp_world_size
    if pad_N > N:
        feature = torch.cat([feature, feature.new_zeros(pad_N - N, feature.shape[1])], dim=0)
        y = torch.cat([y, y.new_full((pad_N - N,) + y.shape[1:], -1)], dim=0)
    nodes_per_rank = pad_N // sp_world_size
    rank_start = sp_rank * nodes_per_rank
    rank_end = rank_start + nodes_per_rank      # always nodes_per_rank per rank (padding included)
    local_N = nodes_per_rank                    # each rank always has this many nodes

    if args.model == "graphormer" and args.num_global_node > 0:
        sub_real_seq_len = nodes_per_rank + args.num_global_node
        global_token_indices = list(range(0, sp_world_size * sub_real_seq_len, sub_real_seq_len))
        set_global_token_indices(global_token_indices)
    else:
        set_global_token_indices([])
    set_last_batch_global_token_indices(None)

    # Local train indices: [rank_start, rank_end) BUT only real nodes (idx < N)
    train_idx_global = split_idx["train"]
    local_train_mask = (train_idx_global >= rank_start) & (train_idx_global < min(rank_end, N))
    local_train_idx = (train_idx_global[local_train_mask] - rank_start).to(device=device, dtype=torch.long)
    local_y = y[rank_start:rank_end].to(device)  # y is already padded to pad_N
    edge_budget_controller = _AdaptiveEdgeBudgetController(args)
    probe_idx_global = _select_probe_nodes(
        split_idx,
        getattr(args, "adaptive_edge_budget_probe_size", 0),
        getattr(args, "seed", 0),
    ) if edge_budget_controller.enabled else None
    if probe_idx_global is not None:
        local_probe_mask = (probe_idx_global >= rank_start) & (probe_idx_global < min(rank_end, N))
        local_probe_idx = (probe_idx_global[local_probe_mask] - rank_start).to(device=device, dtype=torch.long)
    else:
        local_probe_idx = None
    random_blocks_dynamic = (
        (_random_block_sampling_enabled(args) or edge_budget_controller.enabled)
        and (
            (_use_real_edges(args) and max(getattr(args, "real_edges_per_query", 0), edge_budget_controller.real_budget) > 0)
            or (int(getattr(args, "head_hop_walks_per_node", 0)) > 0 and max(getattr(args, "rw_edges_per_query", 0), edge_budget_controller.rw_budget) > 0)
        )
    )
    dynamic_edges = _use_rw_edges(args) or random_blocks_dynamic
    prefetch_cpu_rw = (
        dynamic_edges
        and getattr(args, "random_walk_prefetch", False)
        and not getattr(args, "stream_edges_from_cpu", False)
        and rw_device.type == "cpu"
        and device.startswith("cuda")
    )
    edge_prefetcher = _CpuRandomWalkEdgePrefetcher(prefetch_cpu_rw and sp_rank == sp_src_rank)

    if args.rank == 0:
        print(f"\n{'='*72}")
        print(f"Full-Graph SP Training  (sp_world_size={sp_world_size})")
        print(f"  Nodes per rank: {nodes_per_rank:,}  "
              f"(rank {sp_rank}: {rank_start:,}–{rank_end:,})")
        print(f"  Edge_index: full graph E={edge_index_global.shape[1]:,}")
        print(f"  Sparse edges: real={int(_use_real_edges(args))} "
              f"self={int(bool(args.include_self_loops))} "
              f"rw={int(_use_rw_edges(args))}")
        print(f"  Random-walk device: {rw_device}")
        print(f"  CPU RW prefetch: {int(prefetch_cpu_rw)}")
        print(f"  AMP dtype: {args.amp_dtype}")
        print(f"  CPU edge streaming: {int(bool(getattr(args, 'stream_edges_from_cpu', False)))}")
        print(f"  Sparse query chunk size: {getattr(args, 'sparse_query_chunk_size', 0)}")
        print(f"  Random edge blocks: {int(bool(_random_block_sampling_enabled(args)))} "
              f"(real={getattr(args, 'real_edges_per_query', 0)} "
              f"rw={getattr(args, 'rw_edges_per_query', 0)})")
        print(f"  Adaptive edge budget: {int(edge_budget_controller.enabled)} "
              f"(probe={getattr(args, 'adaptive_edge_budget_probe_size', 0)} "
              f"block={edge_budget_controller.block_size} "
              f"warmup={edge_budget_controller.warmup_epochs} "
              f"patience={edge_budget_controller.patience})")
        print(f"{'='*72}\n")

    # ── Build model ───────────────────────────────────────────────────────
    model = _build_model(args, feature, y, device)
    sync_params_and_buffers(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.startswith("cuda")))

    if args.rank == 0:
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay
    )
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup=args.warmup_updates,
        tot=args.epochs,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )

    best_val = 0.0
    best_test = 0.0
    best_epoch = -1
    loss_ema = None

    x_local = feature[rank_start:rank_end].to(device)  # [nodes_per_rank, d] – padded if needed
    cached_edge_index = None
    if not dynamic_edges:
        cached_edge_index = _build_attention_edges(
            args,
            edge_index_global,
            N,
            device,
            rw_device,
            sp_group,
            sp_src_rank,
            sp_rank,
            local_N,
            edge_budget_state=edge_budget_controller.current_state(),
        )

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # ── Build edges (rank-0 builds, broadcast to all) ─────────────────
        t_rw = time.time()
        if cached_edge_index is not None:
            edge_index_rw = cached_edge_index
        else:
            use_epoch_seed = (
                getattr(args, "stream_edges_from_cpu", False)
                or prefetch_cpu_rw
                or random_blocks_dynamic
            )
            edge_seed = args.seed + epoch if use_epoch_seed else None
            prefetched_cpu = edge_prefetcher.pop(epoch) if edge_prefetcher.enabled else None
            if prefetched_cpu is not None:
                merged = _broadcast_prefetched_edges(prefetched_cpu, device, sp_group, sp_src_rank, sp_rank)
                edge_index_rw = _finalize_attention_edges(args, merged)
            else:
                edge_index_rw = _build_attention_edges(
                    args, edge_index_global, N, device, rw_device,
                    sp_group, sp_src_rank, sp_rank, local_N,
                    edge_seed=edge_seed,
                    edge_budget_state=edge_budget_controller.current_state(),
                )
            if edge_prefetcher.enabled and epoch < args.epochs:
                next_epoch = epoch + 1
                next_seed = args.seed + next_epoch
                edge_prefetcher.submit(
                    next_epoch,
                    lambda next_seed=next_seed: _build_prefetched_cpu_edges(
                        args,
                        edge_index_global,
                        N,
                        rw_device,
                        local_N,
                        next_seed,
                    ),
                )
        rw_time = time.time() - t_rw

        # ── Forward ────────────────────────────────────────────────────────
        t_fwd = time.time()
        # Each rank inputs its slice [local_N, d]; after _SeqAllToAll inside
        # DistributedAttentionNodeLevel the full N nodes are visible for
        # sparse attention; output returns to [local_N, classes].
        with _autocast_context(device, amp_dtype):
            out_local = model(x_local, None, edge_index_rw, attn_type=args.attn_type)
            out_rows = int(out_local.size(0))
            local_y_eff = local_y[:out_rows]
            valid_train_mask = (local_train_idx >= 0) & (local_train_idx < out_rows)
            local_train_idx_eff = local_train_idx[valid_train_mask]

            if local_train_idx_eff.numel() > 0:
                loss = F.nll_loss(
                    out_local.index_select(0, local_train_idx_eff),
                    local_y_eff.index_select(0, local_train_idx_eff).long(),
                )
            else:
                loss = out_local.sum() * 0.0  # zero gradient on ranks with no train nodes
        fwd_time = time.time() - t_fwd

        # ── Backward ───────────────────────────────────────────────────────
        t_bwd = time.time()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        bwd_time = time.time() - t_bwd

        # ── Gradient sync across SP ranks ─────────────────────────────────
        if sp_world_size > 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(sp_world_size)
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=sp_group)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step()

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        del out_local, loss, local_y_eff, valid_train_mask, local_train_idx_eff
        if cached_edge_index is None:
            del edge_index_rw
        if 'merged' in locals():
            del merged
        if 'prefetched_cpu' in locals():
            del prefetched_cpu
        _maybe_cuda_empty_cache(args)

        epoch_time = time.time() - t_epoch
        cpu_rss = _get_process_rss_mib()
        cpu_rss_peak = _get_process_peak_rss_mib()
        if args.rank == 0:
            print(f"Epoch {epoch:04d} | loss={loss_val:.4f} (ema={loss_ema:.4f}) "
                  f"| t={epoch_time:.2f}s "
                  f"(rw={rw_time:.2f}s fwd={fwd_time:.2f}s bwd={bwd_time:.2f}s) "
                  f"| cpu_rss={cpu_rss:.1f}/{cpu_rss_peak:.1f} MiB")

        # ── Evaluation ─────────────────────────────────────────────────────
        if epoch % args.eval_every == 0:
            t_eval = time.time()
            accs = _eval_sp(
                args, model, feature, y, split_idx, edge_index_global, N,
                device, rw_device, sp_group, sp_src_rank, sp_rank, sp_world_size,
                rank_start, rank_end, local_N, amp_dtype=amp_dtype,
                cached_edge_index=cached_edge_index,
                edge_budget_state=edge_budget_controller.current_state(),
            )
            eval_time = time.time() - t_eval

            if args.rank == 0 and accs is not None:
                train_acc = accs.get("train", 0.0)
                val_acc = accs.get("valid", 0.0)
                test_acc = accs.get("test", 0.0)
                print(f"  ↳ Eval ({eval_time:.2f}s) | "
                      f"Train={train_acc:.2%}  Val={val_acc:.2%}  Test={test_acc:.2%}")
                if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = epoch
                    if args.save_model:
                        torch.save(model.state_dict(),
                                   os.path.join(args.model_dir,
                                                f"{args.dataset}_fg_sp.pkl"))
                if test_acc > best_test:
                    best_test = test_acc
                print(f"  ↳ Best: epoch={best_epoch}  val={best_val:.2%}  test={best_test:.2%}")
            del accs
            _maybe_cuda_empty_cache(args)

        _maybe_update_edge_budget(
            args,
            edge_budget_controller,
            epoch,
            model,
            x_local,
            local_y,
            local_probe_idx,
            edge_index_global,
            N,
            device,
            rw_device,
            sp_group,
            sp_src_rank,
            sp_rank,
            local_N,
            amp_dtype,
            sp_world_size,
        )

        if sp_world_size > 1:
            dist.barrier(group=sp_group)

    edge_prefetcher.close()

    if args.rank == 0:
        print(f"\n{'='*72}")
        print(f"Done.  Best epoch={best_epoch}  Val={best_val:.2%}  Test={best_test:.2%}")
        print(f"{'='*72}")

    # Peak GPU memory
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        rsvd = torch.cuda.max_memory_reserved() / (1024 ** 2)
        if dist.is_initialized():
            mem = torch.tensor([alloc, rsvd], device=device)
            gathered = [torch.zeros_like(mem) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, mem)
            if args.rank == 0:
                print("Peak GPU memory per rank (MiB):")
                for r, t in enumerate(gathered):
                    print(f"  rank {r}: allocated={t[0]:.1f}  reserved={t[1]:.1f}")
        else:
            print(f"Peak GPU memory: allocated={alloc:.1f} MiB  reserved={rsvd:.1f} MiB")

    cpu_rss = _get_process_rss_mib()
    cpu_rss_peak = _get_process_peak_rss_mib()
    cpu_mem_device = device if torch.cuda.is_available() else "cpu"
    if dist.is_initialized():
        mem = torch.tensor([cpu_rss, cpu_rss_peak], device=cpu_mem_device, dtype=torch.float32)
        gathered = [torch.zeros_like(mem) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, mem)
        if args.rank == 0:
            print("CPU RSS per rank (MiB):")
            for r, t in enumerate(gathered):
                print(f"  rank {r}: current={t[0]:.1f}  peak={t[1]:.1f}")
    elif args.rank == 0:
        print(f"CPU RSS: current={cpu_rss:.1f} MiB  peak={cpu_rss_peak:.1f} MiB")


if __name__ == "__main__":
    main()
