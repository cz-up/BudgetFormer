import torch
import math
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from gt_sp.gt_layer import DistributedAttentionNodeLevel, _SeqGather
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_global_token_indices,
)
from torch_scatter import scatter
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from collections import deque

try:
    from gt_sp.multi_tier import _MultiTierResourceManager, apply_tier
except ModuleNotFoundError as exc:
    if exc.name != "gt_sp.multi_tier":
        raise
    _MultiTierResourceManager = None

    def apply_tier(layer, mode: str, x, **kwargs):
        if mode == "retain":
            return layer(x, **kwargs)
        if mode in ("keep_mha", "ffn_only"):
            x = layer.forward_attn_only(x, **kwargs)
            return layer.forward_ffn_checkpointed(x)

        def _full(h):
            return layer(h, **kwargs)

        return checkpoint(_full, x, use_reentrant=False)


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        

class CoreAttention(nn.Module):
    """
    Core attn 
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(CoreAttention, self).__init__()

        # SP group: Per attention head and per partition values.
        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            seq_parallel_world_size = get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size 

        self.hidden_size_per_partition = hidden_size // world_size
        self.hidden_size_per_attention_head =  hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // world_size

        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate
        self._head_mass_sum = None
        self._head_mass_count = None

    def reset_head_mass_stats(self):
        self._head_mass_sum = torch.zeros(
            self.num_attention_heads_per_partition, dtype=torch.float32
        )
        self._head_mass_count = torch.zeros(
            self.num_attention_heads_per_partition, dtype=torch.float32
        )
        self.capture_full_attn = False
        self.last_full_attn_mean = None
        self.full_attention_extra_bias = None

    def get_head_mass_stats(self):
        if self._head_mass_sum is None:
            return None
        denom = torch.clamp(self._head_mass_count, min=1.0)
        return self._head_mass_sum / denom

    def enable_hop_mass_tracking(
        self,
        mass: float = 0.95,
        max_queries: int = 64,
        query_sampling: str = "random",
        max_batches: int = 1,
        max_hop: int = 15,
    ) -> None:
        self.track_hop_mass = True
        self.hop_mass_target = float(mass)
        self.hop_mass_max_queries = int(max_queries)
        self.hop_mass_query_sampling = str(query_sampling)
        self.hop_mass_max_batches = int(max_batches)
        self.hop_mass_max_hop = int(max_hop)
        self.reset_hop_mass_stats()

    def disable_hop_mass_tracking(self) -> None:
        self.track_hop_mass = False

    def set_full_attention_capture(self, enabled: bool = True) -> None:
        self.capture_full_attn = bool(enabled)
        if not self.capture_full_attn:
            self.last_full_attn_mean = None

    def get_last_full_attention_mean(self):
        return self.last_full_attn_mean

    def set_full_attention_extra_bias(self, extra_bias) -> None:
        self.full_attention_extra_bias = extra_bias

    def reset_hop_mass_stats(self) -> None:
        self.hop_mass_sum = []
        self.hop_mass_count = 0
        self.hop_mass_max_sum = 0.0
        self.hop_mass_max_count = 0
        self.hop_mass_max = 0
        self.hop_mass_batch_count = 0

    def get_hop_mass_stats(self):
        if not getattr(self, "track_hop_mass", False) or self.hop_mass_count == 0:
            return None
        return (
            list(self.hop_mass_sum),
            self.hop_mass_count,
            self.hop_mass_max_sum,
            self.hop_mass_max_count,
            self.hop_mass_max,
        )

    # def full_flash_attention(self, q, k, v, attn_bias=None, mask=None):
    #     return flash_attn_func(q, k, v, self.attention_dropout_rate)
    

    def full_attention(self, k, q, v, attn_bias, mask=None, edge_index=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        q = q.transpose(1, 2)   # [b, np, s+1, hn]
        v = v.transpose(1, 2)  # [b, np, s+1, hn]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, np, hn, s+1]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            # attn_bias = attn_bias.repeat(1, self.num_heads, 1, 1)
            x = x + attn_bias
        extra_bias = getattr(self, "full_attention_extra_bias", None)
        if extra_bias is not None:
            x = x + extra_bias.to(device=x.device, dtype=x.dtype)
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)

        x = torch.softmax(x, dim=3)
        if getattr(self, "capture_full_attn", False):
            self.last_full_attn_mean = x.mean(dim=1).detach().cpu()
        if getattr(self, "track_hop_mass", False):
            self._accumulate_hop_mass(x, edge_index)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        return x
    

    def sparse_attention_bias(self, q, k, v, edge_index, attn_bias):
        # q, k, v: [b, s, np, hn]  e: [total_edges, n, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+1, s+1]
        batch_size, node_num = k.size(0), k.size(1)
        num_heads = q.size(2)
        edge_hops = None
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 3:
            edge_hops = edge_index[2]
            edge_index = edge_index[:2]
        
        # Reshaping into [total_s, np, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, np, hn],  e: [total_edges, np, hn]
        q = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, num_heads, self.hidden_size_per_attention_head)

        if isinstance(edge_index, list):
            edge_list = edge_index
            if len(edge_list) > num_heads and sequence_parallel_is_initialized():
                head_start = get_sequence_parallel_rank() * self.num_attention_heads_per_partition
                edge_list = edge_list[head_start: head_start + num_heads]
            if len(edge_list) != num_heads:
                edge_list = [edge_list[i % len(edge_list)] for i in range(num_heads)]

            wV = torch.zeros_like(v)
            Z = v.new_zeros(v.size(0), num_heads, 1)
            for h in range(num_heads):
                e = edge_list[h]
                if e is None or e.numel() == 0:
                    continue
                src_idx = e[0].to(torch.long)
                dst_idx = e[1].to(torch.long)
                src = k[:, h, :][src_idx]
                dest = q[:, h, :][dst_idx]
                score = torch.mul(src, dest).sum(-1, keepdim=True).clamp(-5, 5)
                score = torch.exp(score / self.scale)
                msg = v[:, h, :][src_idx] * score
                scatter(msg, dst_idx, dim=0, out=wV[:, h, :], reduce='add')
                scatter(score, dst_idx, dim=0, out=Z[:, h, :], reduce='add')
                if self._head_mass_sum is not None and h < self._head_mass_sum.numel():
                    self._head_mass_sum[h] += float(score.sum().detach().cpu())
                    self._head_mass_count[h] += float(score.numel())
            return wV / (Z + 1e-6)

        # -> [total_edges, np, hn]
        src = k[edge_index[0].to(torch.long)]
        dest = q[edge_index[1].to(torch.long)]
        score = torch.mul(src, dest)  # element-wise multiplication

        # Scale scores by sqrt(d)
        score = score / self.scale

        # Use available edge features to modify the scores for edges
        # -> [total_edges, np, 1]
        score = score.sum(-1, keepdim=True).clamp(-5, 5)

        # [b, np, s+1, s+1] -> [b, s+1, s+1, np] -> [b, s+1, b, s+1, np]
        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 2, 3, 1).contiguous().unsqueeze(2).repeat(1, 1, batch_size, 1, 1)
            attn_bias = attn_bias.view(batch_size * node_num, batch_size * node_num, num_heads)
            attn_bias = attn_bias.repeat(1, 1, 1, num_heads)

            score = score + \
                    attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(2)

        # softmax -> [total_edges, np, 1]
        score = torch.exp(score)

        # Apply attention score to each source node to create edge messages
        # -> [total_edges, np, hn]
        msg = v[edge_index[0].to(torch.long)] * score

        # Add-up real msgs in destination nodes as given by edge_index[1]
        # -> [total_s, np, hn]
        wV = torch.zeros_like(v)
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        # Compute attention normalization coefficient
        # -> [total_s, np, 1]
        Z = score.new_zeros(v.size(0), num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)
        return x

    def _accumulate_sparse_hop_mass(
        self,
        edge_index: torch.Tensor,
        edge_hops: torch.Tensor,
        score: torch.Tensor,
        Z: torch.Tensor,
    ) -> None:
        if edge_index is None or edge_hops is None:
            return
        if not getattr(self, "track_hop_mass", False):
            return
        if self.hop_mass_max_batches > 0 and self.hop_mass_batch_count >= self.hop_mass_max_batches:
            return
        if score.numel() == 0:
            return
        self.hop_mass_batch_count += 1

        edge_hops = edge_hops.to(score.device).long()
        dst = edge_index[1].to(torch.long)
        denom = Z[dst] + 1e-6
        weights = (score / denom).squeeze(-1)
        if weights.numel() == 0:
            return
        weights = weights.mean(dim=1)

        max_hop_limit = int(self.hop_mass_max_hop)
        if max_hop_limit > 0:
            mask = (edge_hops >= 0) & (edge_hops <= max_hop_limit)
        else:
            mask = edge_hops >= 0
        if not mask.any():
            return
        edge_hops = edge_hops[mask]
        dst = dst[mask]
        weights = weights[mask]

        max_hop = int(edge_hops.max().item()) + 1
        if max_hop <= 0:
            return
        num_nodes = int(dst.max().item()) + 1
        combined = dst * max_hop + edge_hops
        node_hop = weights.new_zeros(num_nodes * max_hop)
        node_hop.scatter_add_(0, combined, weights)
        hop_mass = node_hop.view(num_nodes, max_hop).sum(dim=0)
        total = float(hop_mass.sum().item())
        if total <= 0:
            return
        hop_mass = hop_mass / total
        if len(self.hop_mass_sum) < max_hop:
            self.hop_mass_sum.extend([0.0] * (max_hop - len(self.hop_mass_sum)))
        for i in range(max_hop):
            self.hop_mass_sum[i] += float(hop_mass[i].item())
        self.hop_mass_count += 1

        mass_target = max(0.0, min(self.hop_mass_target, 1.0))
        if mass_target > 0.0:
            cum = hop_mass.cumsum(dim=0)
            hit = torch.nonzero(cum >= mass_target, as_tuple=False)
            if hit.numel() > 0:
                needed = int(hit[0].item())
                self.hop_mass_max_sum += float(needed)
                self.hop_mass_max_count += 1
                if needed > self.hop_mass_max:
                    self.hop_mass_max = needed

    def _accumulate_hop_mass(self, attn_probs: torch.Tensor, edge_index) -> None:
        if edge_index is None:
            return
        if not getattr(self, "track_hop_mass", False):
            return
        if self.hop_mass_max_batches > 0 and self.hop_mass_batch_count >= self.hop_mass_max_batches:
            return
        if isinstance(edge_index, list):
            edge_index = [e for e in edge_index if e is not None and e.numel() > 0]
            if not edge_index:
                return
            edge_index = torch.cat(edge_index, dim=1)
        if edge_index.numel() == 0:
            return

        self.hop_mass_batch_count += 1
        bsz, n_head, q_len, k_len = attn_probs.shape
        if bsz == 0 or q_len == 0 or k_len == 0:
            return

        edge_cpu = edge_index.detach().cpu()
        src = edge_cpu[0].long()
        dst = edge_cpu[1].long()
        valid = (src >= 0) & (dst >= 0) & (src < k_len) & (dst < k_len)
        if not valid.any():
            return
        src = src[valid]
        dst = dst[valid]

        adj = [[] for _ in range(k_len)]
        for s, d in zip(src.tolist(), dst.tolist()):
            adj[s].append(d)
            adj[d].append(s)

        probs = attn_probs.mean(dim=1).detach().cpu()
        start_q = 1 if q_len > 1 else 0
        if start_q >= q_len:
            return
        num_candidates = q_len - start_q
        sample_count = min(num_candidates, max(1, self.hop_mass_max_queries))
        query_sampling = getattr(self, "hop_mass_query_sampling", "random")
        if query_sampling == "random":
            query_indices = torch.randperm(num_candidates)[:sample_count].add_(start_q).tolist()
        else:
            query_indices = list(range(start_q, start_q + sample_count))
        max_hop = int(self.hop_mass_max_hop)
        mass_target = max(0.0, min(self.hop_mass_target, 1.0))

        for b in range(bsz):
            for q in query_indices:
                p = probs[b, q, :].float()
                if k_len > 1:
                    p[0] = 0.0
                total_mass = float(p.sum().item())
                if total_mass <= 0:
                    continue
                target_mass = mass_target * total_mass
                dist = [-1] * k_len
                dist[q] = 0
                q_deque = deque([q])
                while q_deque:
                    cur = q_deque.popleft()
                    d = dist[cur]
                    if max_hop > 0 and d >= max_hop:
                        continue
                    for nxt in adj[cur]:
                        if dist[nxt] != -1:
                            continue
                        dist[nxt] = d + 1
                        q_deque.append(nxt)

                hop_mass = []
                for idx, d in enumerate(dist):
                    if d < 0:
                        continue
                    if max_hop > 0 and d > max_hop:
                        continue
                    while len(hop_mass) <= d:
                        hop_mass.append(0.0)
                    hop_mass[d] += float(p[idx].item())
                if not hop_mass:
                    continue
                cum = 0.0
                trimmed = []
                for m in hop_mass:
                    cum += m
                    trimmed.append(m)
                    if cum >= target_mass:
                        break
                if cum <= 0:
                    continue
                needed_hop = len(trimmed) - 1
                if len(self.hop_mass_sum) < len(trimmed):
                    self.hop_mass_sum.extend([0.0] * (len(trimmed) - len(self.hop_mass_sum)))
                for i, m in enumerate(trimmed):
                    self.hop_mass_sum[i] += m / cum
                self.hop_mass_count += 1
                self.hop_mass_max_sum += float(needed_hop)
                self.hop_mass_max_count += 1
                if needed_hop > self.hop_mass_max:
                    self.hop_mass_max = needed_hop


    # ------------------------------------------------------------------
    # Two-phase sparse attention for communication-computation overlap.
    # Phase 1 (sparse_score_phase): depends only on Q and K
    # Phase 2 (sparse_aggregate_phase): depends on V and the scores
    # ------------------------------------------------------------------

    def sparse_score_phase(self, k, q, edge_index, attn_bias):
        """Compute sparse attention scores using only Q and K.

        Returns a dict of intermediate results consumed by
        ``sparse_aggregate_phase``.

        Supports only the plain-tensor edge_index path (no list / dict).
        """
        batch_size, node_num = k.size(0), k.size(1)
        num_heads = q.size(2)

        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 3:
            edge_index = edge_index[:2]

        q_flat = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k_flat = k.view(-1, num_heads, self.hidden_size_per_attention_head)

        src = k_flat[edge_index[0].to(torch.long)]
        dest = q_flat[edge_index[1].to(torch.long)]
        score = torch.mul(src, dest)
        score = score / self.scale
        score = score.sum(-1, keepdim=True).clamp(-5, 5)

        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 2, 3, 1).contiguous().unsqueeze(2).repeat(1, 1, batch_size, 1, 1)
            attn_bias = attn_bias.view(batch_size * node_num, batch_size * node_num, num_heads)
            attn_bias = attn_bias.repeat(1, 1, 1, num_heads)
            score = score + attn_bias[
                edge_index[0].to(torch.long),
                edge_index[1].to(torch.long),
                :,
            ].unsqueeze(2)

        score = torch.exp(score)

        return {
            "score": score,
            "edge_index": edge_index,
            "num_heads": num_heads,
            "total_nodes": q_flat.size(0),
            "batch_size": batch_size,
            "s_len": q.size(1),
        }

    def sparse_aggregate_phase(self, score_ctx, v):
        """Apply precomputed scores to V and scatter-aggregate.

        Consumes the dict returned by ``sparse_score_phase``.
        """
        score = score_ctx["score"]
        edge_index = score_ctx["edge_index"]
        num_heads = score_ctx["num_heads"]
        total_nodes = score_ctx["total_nodes"]

        v_flat = v.view(-1, num_heads, self.hidden_size_per_attention_head)

        msg = v_flat[edge_index[0].to(torch.long)] * score

        wV = torch.zeros_like(v_flat)
        scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

        Z = score.new_zeros(total_nodes, num_heads, 1)
        scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

        x = wV / (Z + 1e-6)
        return x

    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        # q, k, v: [b, s+p, np, hn], edge_index: [2, total_edges], attn_bias: [b, n, s+p, s+p]
        batch_size, s_len = q.size(0), q.size(1)
        attn_type = "sparse" if attn_type is None else str(attn_type).lower()
        if attn_type != "sparse":
            raise ValueError(
                f"GT is restricted to graph-neighborhood sparse attention to match the original implementation; "
                f"got attn_type={attn_type!r}."
            )
        x = self.sparse_attention_bias(q, k, v, edge_index, attn_bias)
        
        # [b, s+p, hp]
        x = x.view(batch_size, s_len, -1)

        return x


class MultiHeadAttention(nn.Module):
    """Distributed multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        local_attn = CoreAttention(
            hidden_size, attention_dropout_rate, num_heads)
        if sequence_parallel_is_initialized():
            self.dist_attn = DistributedAttentionNodeLevel(
                local_attn,
                get_sequence_parallel_group(),
            )
        else:
            self.dist_attn = local_attn


    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        # x: [b, s/p+1, h], attn_bias: [b, n_head, s+1, s+1]
        orig_q_size = x.size()
        # =====================
        # Query, Key, and Value
        # =====================

        # q, k, v: [b, s/p+1, h] -> [b, s/p+1, n_head, hn]
        batch_size = x.size(0) # number of sequences to train a time 
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size) 
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)
        # print(f'rank {get_sequence_parallel_rank()} q: {q[:, 0, :, :]}')
        # exit(0)
        

        # ==================================
        # core attention computation
        # ==================================
        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type)

        # =================
        # linear
        # =================

        # [b, s/p+1, h]

        assert x.size() == orig_q_size
        return x

    def _core_attn(self):
        return getattr(self.dist_attn, "local_attn", self.dist_attn)

    def reset_head_mass_stats(self):
        reset_fn = getattr(self._core_attn(), "reset_head_mass_stats", None)
        if reset_fn is not None:
            reset_fn()

    def get_head_mass_stats(self):
        get_fn = getattr(self._core_attn(), "get_head_mass_stats", None)
        if get_fn is not None:
            return get_fn()
        return None

    def enable_hop_mass_tracking(self, mass=0.95, max_queries=64, query_sampling="random", max_batches=1, max_hop=15):
        enable_fn = getattr(self._core_attn(), "enable_hop_mass_tracking", None)
        if enable_fn is not None:
            enable_fn(mass, max_queries, query_sampling, max_batches, max_hop)

    def set_full_attention_capture(self, enabled: bool = True):
        set_fn = getattr(self._core_attn(), "set_full_attention_capture", None)
        if set_fn is not None:
            set_fn(enabled)

    def get_last_full_attention_mean(self):
        get_fn = getattr(self._core_attn(), "get_last_full_attention_mean", None)
        if get_fn is not None:
            return get_fn()
        return None

    def set_full_attention_extra_bias(self, extra_bias) -> None:
        set_fn = getattr(self._core_attn(), "set_full_attention_extra_bias", None)
        if set_fn is not None:
            set_fn(extra_bias)

    def disable_hop_mass_tracking(self):
        disable_fn = getattr(self._core_attn(), "disable_hop_mass_tracking", None)
        if disable_fn is not None:
            disable_fn()

    def reset_hop_mass_stats(self):
        reset_fn = getattr(self._core_attn(), "reset_hop_mass_stats", None)
        if reset_fn is not None:
            reset_fn()

    def get_hop_mass_stats(self):
        get_fn = getattr(self._core_attn(), "get_hop_mass_stats", None)
        if get_fn is not None:
            return get_fn()
        return None


class EncoderLayer(nn.Module):
    def __init__(
        self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads,
    ):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.O = nn.Linear(hidden_size, hidden_size)

        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.FFN_layer1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN_layer2 = nn.Linear(hidden_size*2, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        y = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        y = self.self_attention_dropout(y)
        y = self.O(y)
        x = x + y
        x = self.layer_norm1(x)

        y = self.FFN_layer1(x)
        y = F.relu(y)
        y = self.self_attention_dropout(y)
        y = self.FFN_layer2(y)
        x = x + y
        x = self.layer_norm2(x)
        return x

    def forward_attn_only(self, x, attn_bias=None, edge_index=None, attn_type=None, **kwargs):
        """MHA sub-block (used by apply_tier keep_mha path).

        Returns LN1(x + attn_out); FFN takes this normalized residual as input.
        """
        y = self.self_attention(x, attn_bias, edge_index=edge_index, attn_type=attn_type)
        y = self.self_attention_dropout(y)
        y = self.O(y)
        return self.layer_norm1(x + y)

    def forward_ffn_checkpointed(self, x_ln):
        """FFN sub-block with gradient checkpointing.

        x_ln is LN1(x + attn_out) from forward_attn_only.
        """
        def _ffn(h):
            y = self.FFN_layer1(h)
            y = F.relu(y)
            y = self.self_attention_dropout(y)
            y = self.FFN_layer2(y)
            return self.layer_norm2(h + y)
        return checkpoint(_ffn, x_ln, use_reentrant=False)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y



class GT(nn.Module):
    """GT for node-level task.
    No global token.

    """
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        input_dropout_rate,
        attention_dropout_rate,
        ffn_dim,
        num_global_node,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        

        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads,
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.n_layers = n_layers

        self.MLP_layer = MLPReadout(hidden_dim, output_dim)   # 1 out dim since regression problem
        self.apply(lambda module: init_params(module, n_layers=n_layers))

        self.activation_checkpoint = False
        self.activation_checkpoint_mode = "none"
        self._comm_ckpt = None  # _MultiTierResourceManager | None

    def set_activation_checkpoint(self, enabled: bool = True, mode: str | None = None, deferred: bool = False) -> None:
        mode_aliases = {
            None: "none",
            "none": "none",
            "all": "layer",
            "layer": "layer",
            "ffn_only": "ffn_only",
            "multi_tier": "multi_tier",
        }
        resolved_mode = mode_aliases.get(mode)
        if resolved_mode is None:
            raise ValueError(
                f"activation_checkpoint_mode must be 'all', 'ffn_only', or 'multi_tier', got {mode!r}"
            )
        enabled = bool(enabled) and resolved_mode != "none"
        self.activation_checkpoint = enabled
        self.activation_checkpoint_mode = resolved_mode if enabled else "none"
        if self.activation_checkpoint_mode == "multi_tier":
            if _MultiTierResourceManager is None:
                raise RuntimeError(
                    "activation_checkpoint_mode='multi_tier' requires gt_sp/multi_tier.py, "
                    "but that module is not available in this environment."
                )
            self._comm_ckpt = _MultiTierResourceManager(len(self.layers), deferred=deferred)
        else:
            self._comm_ckpt = None

    def comm_aware_notify_budget_frozen(self, reuse_deferred_baseline: bool = False) -> None:
        if self._comm_ckpt is not None:
            self._comm_ckpt.notify_budget_frozen(reuse_deferred_baseline=reuse_deferred_baseline)

    def comm_aware_notify_step_end(self, device, t_bwd: float = None, t_fwd: float = None) -> None:
        if self._comm_ckpt is not None:
            self._comm_ckpt.notify_step_end(device, t_bwd=t_bwd, t_fwd=t_fwd)

    def forward(self, x, attn_bias, edge_index, perturb=None, attn_type=None):
        # x → [bs=1, s/p, x_d]
        x = x.unsqueeze(0)

        # [bs, s/p, x_d] → [bs, s/p, h]
        node_feature = self.node_encoder(x)

        node_feature -= perturb if perturb is not None else 0
        output = self.input_dropout(node_feature)

        use_ckpt = self.activation_checkpoint and self.training and torch.is_grad_enabled()
        ckpt_mode = getattr(self, "activation_checkpoint_mode", "none")

        if use_ckpt and ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
            self._comm_ckpt.plan(output.device)

        for i, enc_layer in enumerate(self.layers):
            if not use_ckpt:
                output = enc_layer(output, edge_index=edge_index, attn_type=attn_type)
                continue
            if ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
                layer_mode = self._comm_ckpt.mode(i)
            else:
                layer_mode = ckpt_mode
            output = apply_tier(enc_layer, layer_mode, output, edge_index=edge_index, attn_type=attn_type)

        if use_ckpt and ckpt_mode == "multi_tier" and self._comm_ckpt is not None:
            self._comm_ckpt.record_post_forward_memory(output.device)

        # Output part
        output = self.MLP_layer(output[0, :, :])

        return F.log_softmax(output, dim=1)

    def reset_head_mass_stats(self):
        for layer in self.layers:
            if hasattr(layer, "self_attention"):
                layer.self_attention.reset_head_mass_stats()

    def get_head_mass_stats(self):
        stats = []
        for layer in self.layers:
            if hasattr(layer, "self_attention"):
                layer_stats = layer.self_attention.get_head_mass_stats()
                if layer_stats is not None:
                    stats.append(layer_stats)
        if not stats:
            return None
        return sum(stats) / len(stats)

    def enable_hop_mass_tracking(self, mass=0.95, max_queries=64, query_sampling="random", max_batches=1, max_hop=15):
        for layer in self.layers:
            if hasattr(layer, "self_attention"):
                layer.self_attention.enable_hop_mass_tracking(mass, max_queries, query_sampling, max_batches, max_hop)

    def _selected_full_attention_layer_indices(self, layer_mode: str = "all"):
        if not self.layers:
            return []
        mode = str(layer_mode).lower()
        if mode == "last":
            return [len(self.layers) - 1]
        if mode == "all":
            return list(range(len(self.layers)))
        raise ValueError(f"Unsupported full-attention layer mode: {layer_mode}")

    def set_full_attention_capture_layers(self, enabled: bool = True, layer_mode: str = "all") -> None:
        enabled_indices = set(self._selected_full_attention_layer_indices(layer_mode)) if enabled else set()
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "self_attention"):
                layer.self_attention.set_full_attention_capture(i in enabled_indices)

    def get_full_attention_means_per_layer(self, layer_mode: str = "all"):
        attn_means = []
        for i in self._selected_full_attention_layer_indices(layer_mode):
            layer = self.layers[i]
            if hasattr(layer, "self_attention"):
                attn_means.append(layer.self_attention.get_last_full_attention_mean())
            else:
                attn_means.append(None)
        return attn_means

    def set_full_attention_extra_biases(self, extra_biases, layer_mode: str = "all") -> None:
        indices = self._selected_full_attention_layer_indices(layer_mode)
        if extra_biases is None:
            bias_by_index = {}
        else:
            if len(extra_biases) != len(indices):
                raise ValueError(
                    f"Expected {len(indices)} full-attention extra bias tensors for layer_mode={layer_mode}, "
                    f"got {len(extra_biases)}."
                )
            bias_by_index = {layer_idx: extra_biases[pos] for pos, layer_idx in enumerate(indices)}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "self_attention"):
                layer.self_attention.set_full_attention_extra_bias(bias_by_index.get(i, None))

    def set_last_layer_full_attention_capture(self, enabled: bool = True) -> None:
        self.set_full_attention_capture_layers(enabled, layer_mode="last")

    def get_last_layer_full_attention_mean(self):
        attn_means = self.get_full_attention_means_per_layer(layer_mode="last")
        return attn_means[0] if attn_means else None

    def set_last_layer_full_attention_extra_bias(self, extra_bias) -> None:
        self.set_full_attention_extra_biases([extra_bias], layer_mode="last")

    def disable_hop_mass_tracking(self):
        for layer in self.layers:
            if hasattr(layer, "self_attention"):
                layer.self_attention.disable_hop_mass_tracking()

    def reset_hop_mass_stats(self):
        for layer in self.layers:
            if hasattr(layer, "self_attention"):
                layer.self_attention.reset_hop_mass_stats()

    def get_hop_mass_stats_per_layer(self):
        stats = []
        for layer in self.layers:
            if hasattr(layer, "self_attention"):
                stats.append(layer.self_attention.get_hop_mass_stats())
            else:
                stats.append(None)
        return stats
