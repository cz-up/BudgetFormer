import torch
import math
from collections import deque
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter
from gt_sp.layer import DistributedAttentionNoMerge
from gt_sp.initialize import (
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_last_batch_flag,
)
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
class CoreAttention(nn.Module):
    """
    Core attention
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(CoreAttention, self).__init__()

        # SP group: Per attention head and per partition values.
        seq_parallel_world_size = 1
        if sequence_parallel_is_initialized():
            seq_parallel_world_size = get_sequence_parallel_world_size()
        world_size = seq_parallel_world_size 

        self.hidden_size_per_partition = hidden_size // world_size
        self.hidden_size_per_attention_head = hidden_size // num_heads
        self.num_attention_heads_per_partition = num_heads // world_size

        self.scale = math.sqrt(self.hidden_size_per_attention_head)
        self.num_heads = num_heads
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.attention_dropout_rate = attention_dropout_rate
        self.capture_full_attn = False
        self.last_full_attn_mean = None
        self.full_attention_extra_bias = None


        # entropy 统计参数
        self.track_entropy_stats = False
        self.entropy_sum = 0.0
        self.entropy_count = 0

        # self._shuffle = True
        self._shuffle = False
        if self._shuffle:
            print('Shuffling destination nodes in edge_index for sparse attention')


    def reset_parameters(self):
        torch.nn.init.constant_(self.b, 0.1)


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


    
    def sparse_attention_bias(self, k, q, v, edge_index, attn_bias):
        # kqv: [b, s+p, np, hn], edge_index: [2, n_edges], attn_bias: [b, s+p, s+p, np]
        batch_size, node_num = k.size(0), k.size(1)
        if self.training:
            num_heads = self.num_attention_heads_per_partition
        else:
            num_heads = self.num_heads
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 3:
            edge_index = edge_index[:2]
   
        # Reshaping into [total_s, np, hn] to
        # get projections for multi-head attention
        # kqv: [total_s, np, hn]
        q = q.view(-1, num_heads, self.hidden_size_per_attention_head)
        k = k.view(-1, num_heads, self.hidden_size_per_attention_head)
        v = v.view(-1, num_heads, self.hidden_size_per_attention_head)
        # q = q.half()
        # k = k.half()
        # v = v.half()
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(2).repeat(1, 1, batch_size, 1, 1)
            attn_bias = attn_bias.view(batch_size * node_num, batch_size * node_num, -1)

        if isinstance(edge_index, dict):
            if edge_index.get("mode") != "cpu_stream":
                raise ValueError(f"Unsupported sparse edge payload mode: {edge_index.get('mode')}")

            src_cpu = edge_index["src"]
            dst_cpu = edge_index["dst"]
            offsets = edge_index["offsets"]
            offsets_list = offsets.tolist()
            wV = v.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
            Z = v.new_zeros(v.size(0), num_heads, 1)
            non_blocking = src_cpu.device.type == "cpu"

            for start, end in zip(offsets_list[:-1], offsets_list[1:]):
                if end <= start:
                    continue
                src = src_cpu[start:end].to(device=k.device, dtype=torch.long, non_blocking=non_blocking)
                dst = dst_cpu[start:end].to(device=k.device, dtype=torch.long, non_blocking=non_blocking)
                src_k = k[src]
                dst_q = q[dst]
                score = torch.mul(src_k, dst_q)
                score = score / self.scale
                score = score.sum(-1, keepdim=True).clamp(-5, 5)

                if attn_bias is not None:
                    score = score + attn_bias[src, dst, :].unsqueeze(-1)

                score = torch.exp(score)
                msg = v[src] * score
                scatter(msg, dst, dim=0, out=wV, reduce='add')
                scatter(score, dst, dim=0, out=Z, reduce='add')

            x = wV / (Z + 1e-6)
        elif isinstance(edge_index, list):
            wV = None
            Z = None
            groups = max(1, len(edge_index))
            heads_per_group = (num_heads + groups - 1) // groups
            for g, edge_index_g in enumerate(edge_index):
                if edge_index_g is None or edge_index_g.numel() == 0:
                    continue
                h_start = g * heads_per_group
                h_end = min((g + 1) * heads_per_group, num_heads)
                if h_start >= h_end:
                    continue
                edge_index_g = edge_index_g.to(k.device)
                if self._shuffle:
                    destinations = edge_index_g[1, :].clone()
                    shuffled_indices = torch.randperm(destinations.size(0), device=destinations.device)
                    shuffled_destinations = destinations[shuffled_indices]
                    edge_index_g = torch.stack([edge_index_g[0, :], shuffled_destinations])
                src = k[edge_index_g[0].to(torch.long), h_start:h_end, :]
                dest = q[edge_index_g[1].to(torch.long), h_start:h_end, :]
                score = torch.mul(src, dest)
                score = score / self.scale
                score = score.sum(-1, keepdim=True).clamp(-5, 5)
                if attn_bias is not None:
                    score = score + attn_bias[
                        edge_index_g[0].to(torch.long),
                        edge_index_g[1].to(torch.long),
                        h_start:h_end,
                    ]
                score = torch.exp(score)
                msg = v[edge_index_g[0].to(torch.long), h_start:h_end, :] * score
                if wV is None:
                    wV = msg.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
                    Z = score.new_zeros(v.size(0), num_heads, 1)
                scatter(msg, edge_index_g[1], dim=0, out=wV[:, h_start:h_end, :], reduce='add')
                scatter(score, edge_index_g[1], dim=0, out=Z[:, h_start:h_end, :], reduce='add')
            if wV is None:
                wV = v.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
                Z = v.new_zeros(v.size(0), num_heads, 1)
            x = wV / (Z + 1e-6)
        else:
            if self._shuffle: # Only apply shuffling during training
                destinations = edge_index[1, :].clone() # 克隆一份原始目标节点用于比较
                shuffled_indices = torch.randperm(destinations.size(0), device=destinations.device)
                shuffled_destinations = destinations[shuffled_indices]
            
                # --- 开始统计未改变的边 ---
                unchanged_edges = (destinations == shuffled_destinations).sum().item()
                total_edges = destinations.size(0)
                unchanged_percentage = (unchanged_edges / total_edges) * 100 if total_edges > 0 else 0
                print(f"Unchanged edges after shuffling: {unchanged_edges}/{total_edges} ({unchanged_percentage:.2f}%)")
                # --- 统计结束 ---
            
                edge_index = torch.stack([edge_index[0, :], shuffled_destinations])

            # -> [n_edges, np, hn]
            src = k[edge_index[0].to(torch.long)] 
            dest = q[edge_index[1].to(torch.long)] 
            score = torch.mul(src, dest)  # element-wise multiplication
                
            # Scale scores by sqrt(d)
            score = score / self.scale

            # Use available edge features to modify the scores for edges
            # -> [total_edges, np, 1] 
            score = score.sum(-1, keepdim=True).clamp(-5, 5)

            if attn_bias is not None:
                score = score + \
                        attn_bias[edge_index[0].to(torch.long), edge_index[1].to(torch.long), :].unsqueeze(-1)

            # softmax -> [total_edges, np, 1]
            score = torch.exp(score) 

            # Apply attention score to each source node to create edge messages
            # -> [total_edges, np, hn]
            msg = v[edge_index[0].to(torch.long)] * score
            
            # Add-up real msgs in destination nodes as given by edge_index[1]
            # -> [total_s, np, hn]
            wV = msg.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
            scatter(msg, edge_index[1], dim=0, out=wV, reduce='add')

            # Compute attention normalization coefficient
            # -> [total_s, np, 1]
            Z = score.new_zeros(v.size(0), num_heads, 1)
            scatter(score, edge_index[1], dim=0, out=Z, reduce='add')

            x = wV / (Z + 1e-6)
        
        return x

    def full_attention(self, k, q, v, attn_bias, edge_index=None):
        # [b, np, sq+1, sk+1]
        output_size = (q.size(0),
                       q.size(2),
                       q.size(1),
                       k.size(1))
        if self.training:
            num_heads = self.num_attention_heads_per_partition
        else:
            num_heads = self.num_heads

        
        # [b, sq+1, np, hn] -> [sq+1, b * np, hn]
        q = q.view(output_size[2], output_size[0] * output_size[1], -1)
        # [b, sk+1, np, hn] -> [sk+1, b * np, hn]
        k = k.view(output_size[3], output_size[0] * output_size[1], -1)
        
        # Scaled Dot-Product Attention.
        q = q * self.scale
        
        # Raw attention scores. [b * np, sq+1, sk+1]
        x = torch.matmul(q.transpose(0, 1), # [b * np, sq+1, hn]
                         k.transpose(0, 1).transpose(1, 2))  # [b * np, hn, sk+1]

        # change view to [b, np, sq+1, sk+1]
        x = x.view(*output_size)

        if attn_bias is not None:
            # attn_bias: [b, s+1, s+1, 1] -> [b, s+1, s+1, np]
            # attn_bias = attn_bias.repeat(1, 1, 1, num_heads) 
            attn_bias = attn_bias.view(*output_size) # [b, s+1, s+1, np] -> [b, np, s+1, s+1]
            x = x + attn_bias
        extra_bias = getattr(self, "full_attention_extra_bias", None)
        if extra_bias is not None:
            x = x + extra_bias.to(device=x.device, dtype=x.dtype)
        attn_probs = torch.softmax(x, dim=3)
        if getattr(self, "capture_full_attn", False):
            self.last_full_attn_mean = attn_probs.mean(dim=1).detach().cpu()
        if getattr(self, "track_hop_mass", False):
            self._accumulate_hop_mass(attn_probs, edge_index)
        x = self.att_dropout(attn_probs)

        # =========================
        # Context layer. [s, b, hp]
        # =========================

        # value_layer -> context layer.
        # [b, sk+1, np, hn] --> [b, np, sq+1, hn]
  
        # [b, np, sq+1, hn]
        output_size = (v.size(0),
                       v.size(2),
                       q.size(0),
                       v.size(3))

        # -> [sk+1, b * np, hn]
        v = v.view(v.size(1), output_size[0] * output_size[1], -1)     

        # change view [b * np, sq+1, sk+1]
        x = x.view(output_size[0] * output_size[1], output_size[2], -1)  

        # matmul: [b * np, sq+1, hn]
        x = torch.bmm(x, v.transpose(0, 1)) # [b * np, sk+1, hn]
        
        # [b, np, sq+1, hn]
        x = x.view(*output_size)

        # [b, np, sq+1, hn] --> [b, sq+1, np, hn]
        x = x.permute(0, 2, 1, 3).contiguous()

        # [b, sq+1, np, hn] --> [b, sq+1, hp]
        x = x.view(output_size[0], output_size[2], -1)

        return x


    def _accumulate_hop_mass(self, attn_probs: torch.Tensor, edge_index: torch.Tensor) -> None:
        if edge_index is None or edge_index.numel() == 0:
            return
        if not getattr(self, "track_hop_mass", False):
            return
        if self.hop_mass_max_batches > 0 and self.hop_mass_batch_count >= self.hop_mass_max_batches:
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
                    if d <= 0:
                        continue
                    if max_hop > 0 and d > max_hop:
                        continue
                    while len(hop_mass) < d:
                        hop_mass.append(0.0)
                    hop_mass[d - 1] += float(p[idx].item())
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
                needed = len(trimmed)
                if len(self.hop_mass_sum) < needed:
                    self.hop_mass_sum.extend([0.0] * (needed - len(self.hop_mass_sum)))
                for i, m in enumerate(trimmed):
                    self.hop_mass_sum[i] += m / cum
                self.hop_mass_count += 1
                self.hop_mass_max_sum += float(needed)
                self.hop_mass_max_count += 1
                if needed > self.hop_mass_max:
                    self.hop_mass_max = needed


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
            mask = (edge_hops >= 1) & (edge_hops <= max_hop_limit)
        else:
            mask = edge_hops >= 1
        if not mask.any():
            return
        edge_hops = edge_hops[mask] - 1
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
                needed = int(hit[0].item()) + 1
                self.hop_mass_max_sum += float(needed)
                self.hop_mass_max_count += 1
                if needed > self.hop_mass_max:
                    self.hop_mass_max = needed



    def forward(self, q, k, v, attn_bias=None, edge_index=None, attn_type=None):
        # ===================================
        # Raw attention scores. [b, np, s+1, s+1]
        # ===================================
        # q, k, v: [b, s+1, np, hn]
        batch_size, s_len = q.size(0), q.size(1)
        
        if attn_type == "full":
            x = self.full_attention(k, q, v, attn_bias, edge_index=edge_index)
        elif attn_type == "sparse":
            x = self.sparse_attention_bias(k, q, v, edge_index, attn_bias)
            # x = x.float()
        elif attn_type == "flash":
            q = q.half()
            k = k.half()
            v = v.half()
            x = flash_attn_func(q, k, v, self.attention_dropout_rate)
            x = x.float()

        # [b, s+1, hp]
        x = x.view(batch_size, s_len, -1)

        return x


class MultiHeadAttention(nn.Module):
    """Distributed multi-headed attention.

    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads # hn
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)

        local_attn = CoreAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim)
        if sequence_parallel_is_initialized():
            self.dist_attn = DistributedAttentionNoMerge(local_attn, get_sequence_parallel_group())
        else:
            self.dist_attn = local_attn

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def reset_parameters(self):
        torch.nn.init.constant_(self.b, 0.1)

    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):
        # x: [b, s/p+1, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # q, k, v: [b, s/p+1, h] -> [b, s/p+1, n_head, hn]
        batch_size = x.size(0) # number of sequences to train a time 
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.att_size)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.att_size) 
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.att_size)

        # ==================================
        # core attention computation
        # ==================================

        # [b, s/p+1, h]
        x = self.dist_attn(q, k, v, attn_bias, edge_index, attn_type)

        # =================
        # linear
        # =================

        # [b, s/p+1, h]
        x = self.output_layer(x)  

        return x


class EncoderLayer(nn.Module):
    """A single encoder layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim):
        super(EncoderLayer, self).__init__()
  
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, attn_bias=None, edge_index=None, attn_type=None):

        # ==================================
        # MHA
        # ==================================     
          
        y = self.self_attention_norm(x) # x: [b, s/p+1, h]
        y = self.self_attention(y, attn_bias, edge_index, attn_type=attn_type)
        y = self.self_attention_dropout(y)
        x = x + y
        
        # ==================================
        # MLP
        # ==================================    
            
        y = self.ffn_norm(x) # x: [b, s/p+1, h]
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
    

class _CommAwareCheckpointer:
    """Communication-aware selective rematerialization for Graphormer SP training.

    Standard activation checkpointing checkpoints the full EncoderLayer, which
    forces backward to re-execute all A2A collective communications (3 pre-attn
    A2As + 1 post-attn A2A) and the sparse scatter-gather attention.  This class
    decides *per layer* whether to:

      "keep_mha"   – run MHA without checkpointing so that post-A2A Q/K/V
                     activations are retained by autograd, eliminating A2A
                     recomputation in backward.  Only the cheap FFN block is
                     checkpointed.
      "full_layer" – checkpoint the entire EncoderLayer (original behaviour).

    Why estimation-based budgets fail
    -----------------------------------
    It is tempting to estimate M_layer from the Q/K/V tensor shapes, but in
    keep_mha mode autograd must retain *all* intermediate tensors of the MHA
    block — not just Q/K/V post-A2A.  For sparse graph attention this includes
    the edge-indexed tensors ``src``, ``dest``, ``score``, ``msg``, ``wV``, and
    the pre-output-A2A context.  These are O(|E| × np × hn) and can easily be
    5-10× larger than the Q/K/V estimate alone, causing the budget to be wildly
    over-estimated and OOM.

    Measurement-based two-step calibration
    ----------------------------------------
    Instead of estimating M_layer, this class measures it directly via two
    warmup steps:

      Step 0 – WARMUP  : all layers → full_layer.  Peak stats are reset so a
                          clean high-watermark for this step is recorded.
      Step 1 – CALIBRATE: last layer → keep_mha, rest → full_layer.  Peak stats
                          are reset again.
      After step 1 – ``notify_step_end()`` reads both peaks and computes
                          M_layer = peak_calibrate − peak_warmup.
                          It then greedily plans the ACTIVE mode assignment.
      Step 2+ – ACTIVE : use pre-planned modes; ``plan()`` is a no-op.

    ``notify_step_end(device)`` must be called by the training loop after each
    step's backward + optimizer.step() + lr_scheduler.step().  This ensures the
    measurement includes the optimizer state (lazily initialised on the first
    step) and is taken when no transient activations are live.

    Layer priority
    --------------
    Layers are assigned "keep_mha" greedily from the last layer inward.
    Backward runs last-to-first, so keeping the final layers eliminates
    recomputation on the earliest part of the backward critical path.

    Safety constraint
    -----------------
    Budget = GPU_total × (1 − margin) − peak_warmup

    With k keep_mha layers the worst-case peak ≈ peak_warmup + k × M_layer.
    We want that ≤ GPU_total × (1 − margin), so k ≤ budget / M_layer.
    """

    # Internal states
    _WARMUP = 0
    _CALIBRATE = 1
    _ACTIVE = 2

    def __init__(self, n_layers: int, safety_margin: float = 0.20):
        self.n_layers = n_layers
        self.safety_margin = float(safety_margin)
        self._modes: list = ["full_layer"] * n_layers
        self._state: int = self._WARMUP
        self._peak_warmup: int = 0          # max_memory_allocated during warmup step
        self._fwd_mem_warmup: int = 0       # memory_allocated at end of warmup forward
        self._fwd_mem_calibrate: int = 0    # memory_allocated at end of calibrate forward
        self._m_layer: int = 0              # measured bytes per keep_mha layer
        self._last_n_keep: int = -1

    @staticmethod
    def _is_rank0() -> bool:
        try:
            import torch.distributed as _dist
            return (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            return True

    def plan(self, device) -> None:
        """Assign checkpoint modes for the upcoming forward pass.

        In WARMUP / CALIBRATE states resets the CUDA peak-memory counter so
        that ``notify_step_end`` can read a clean high-watermark window.
        In ACTIVE state this is a pure no-op.
        """
        if not torch.cuda.is_available():
            return

        if self._state == self._WARMUP:
            self._modes = ["full_layer"] * self.n_layers
            torch.cuda.reset_peak_memory_stats(device)
            if self._is_rank0():
                print(
                    f"[CommAwareCheckpointer] WARMUP: "
                    f"all {self.n_layers} layers → full_layer"
                )

        elif self._state == self._CALIBRATE:
            # Keep only the last layer to measure its live activation footprint.
            modes = ["full_layer"] * self.n_layers
            modes[-1] = "keep_mha"
            self._modes = modes
            torch.cuda.reset_peak_memory_stats(device)
            if self._is_rank0():
                print(
                    f"[CommAwareCheckpointer] CALIBRATE: "
                    f"last layer → keep_mha, rest → full_layer"
                )

        # ACTIVE: modes already set by notify_step_end; nothing to do here.

    def record_post_forward_memory(self, device) -> None:
        """Record live memory immediately after the encoder forward completes.

        Must be called from Graphormer.forward() after the layer loop but
        before returning (i.e. before loss computation and backward).  At this
        point every keep_mha layer's autograd tensors are live in memory, giving
        the true per-layer cost without the noise introduced by backward spikes
        or optimizer-state allocations.

        Why not use peak-based diff:
            In full_layer mode the backward recomputes one layer at a time,
            creating a spike that is the warmup peak.  With keep_mha=1 that
            spike is *eliminated* for the last layer, so peak_calibrate ≤
            peak_warmup and the diff is negative — useless.  Measuring live
            memory after forward sidesteps this entirely.
        """
        if not torch.cuda.is_available():
            return
        live = torch.cuda.memory_allocated(device)
        if self._state == self._WARMUP:
            self._fwd_mem_warmup = live
        elif self._state == self._CALIBRATE:
            self._fwd_mem_calibrate = live

    def notify_step_end(self, device) -> None:
        """Call after backward + optimizer.step() + lr_scheduler.step().

        Advances the internal state machine and, in CALIBRATE state, computes
        M_layer and the greedy layer assignment.
        """
        if not torch.cuda.is_available():
            return
        is_r0 = self._is_rank0()
        total = torch.cuda.get_device_properties(device).total_memory

        if self._state == self._WARMUP:
            self._peak_warmup = torch.cuda.max_memory_allocated(device)
            self._state = self._CALIBRATE
            if is_r0:
                print(
                    f"[CommAwareCheckpointer] warmup done: "
                    f"peak={self._peak_warmup/(1024**2):.0f} MiB, "
                    f"fwd_live={self._fwd_mem_warmup/(1024**2):.0f} MiB "
                    f"(total={total/(1024**2):.0f} MiB)"
                )

        elif self._state == self._CALIBRATE:
            # M_layer = additional live memory from one keep_mha layer,
            # measured at forward-end before any backward allocation.
            # This correctly captures all autograd-retained tensors:
            # post-A2A Q/K/V, sparse-attention intermediates (src/dest/score/
            # msg/wV), and the pre-output-A2A context — the quantities that
            # peak-based diff misses when backward spikes differ between steps.
            m_layer = self._fwd_mem_calibrate - self._fwd_mem_warmup
            if m_layer <= 0:
                # Extremely unlikely: might happen if forward allocations are
                # non-deterministic.  Use 15 % of warmup peak as safe fallback.
                m_layer = max(int(self._peak_warmup * 0.15), 1)
                if is_r0:
                    print(
                        f"[CommAwareCheckpointer] WARNING: "
                        f"fwd_calibrate ({self._fwd_mem_calibrate/(1024**2):.0f} MiB) "
                        f"≤ fwd_warmup ({self._fwd_mem_warmup/(1024**2):.0f} MiB); "
                        f"fallback M_layer={m_layer/(1024**2):.1f} MiB"
                    )
            self._m_layer = m_layer

            # Budget = room above the warmup full-layer peak, with margin.
            # Worst-case new peak ≈ peak_warmup + k × M_layer.
            budget = max(int(total * (1.0 - self.safety_margin) - self._peak_warmup), 0)
            modes = ["full_layer"] * self.n_layers
            remaining = budget
            for i in reversed(range(self.n_layers)):
                if remaining >= self._m_layer:
                    modes[i] = "keep_mha"
                    remaining -= self._m_layer
                else:
                    break
            self._modes = modes
            self._state = self._ACTIVE

            n_keep = sum(1 for m in modes if m == "keep_mha")
            self._last_n_keep = n_keep
            if is_r0:
                m_mib = self._m_layer / (1024**2)
                print(
                    f"[CommAwareCheckpointer] ACTIVE: "
                    f"M_layer={m_mib:.1f} MiB (live-measured), "
                    f"keep_mha={n_keep}/{self.n_layers} layers "
                    f"(peak_warmup={self._peak_warmup/(1024**2):.0f} MiB, "
                    f"budget={budget/(1024**2):.0f} MiB, "
                    f"retained={n_keep * m_mib:.1f} MiB)"
                )

        # ACTIVE: no re-planning needed (stable after calibration).

    def mode(self, layer_idx: int) -> str:
        return self._modes[layer_idx]


class Graphormer(nn.Module):
    """Graphormer for node-level task: one node - one token
        global token index: 0

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

        self.num_heads = num_heads
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(input_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.n_layers = n_layers
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_global_node = num_global_node
        
        self.graph_token = nn.Embedding(self.num_global_node, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(self.num_global_node, attn_bias_dim)
        # Project structural attn_bias (one-hot / positional) to per-head scalars.
        # Active only when attn_bias is not None (i.e. local_spd mode).
        self.attn_bias_proj = nn.Linear(attn_bias_dim, num_heads, bias=False)
        self.activation_checkpoint = False
        self.activation_checkpoint_mode = "none"
        self._comm_ckpt: _CommAwareCheckpointer | None = None
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    @staticmethod
    def _get_core_attention(layer):
        dist_attn = layer.self_attention.dist_attn
        return getattr(dist_attn, "local_attn", dist_attn)

    def enable_hop_mass_tracking(
        self,
        mass: float = 0.95,
        max_queries: int = 64,
        query_sampling: str = "random",
        max_batches: int = 1,
        max_hop: int = 15,
    ) -> None:
        for layer in self.layers:
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "enable_hop_mass_tracking"):
                core_attn.enable_hop_mass_tracking(mass, max_queries, query_sampling, max_batches, max_hop)

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
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "set_full_attention_capture"):
                core_attn.set_full_attention_capture(i in enabled_indices)

    def get_full_attention_means_per_layer(self, layer_mode: str = "all"):
        attn_means = []
        for i in self._selected_full_attention_layer_indices(layer_mode):
            core_attn = self._get_core_attention(self.layers[i])
            get_fn = getattr(core_attn, "get_last_full_attention_mean", None)
            attn_means.append(get_fn() if callable(get_fn) else None)
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
            core_attn = self._get_core_attention(layer)
            set_fn = getattr(core_attn, "set_full_attention_extra_bias", None)
            if callable(set_fn):
                set_fn(bias_by_index.get(i, None))

    def set_last_layer_full_attention_capture(self, enabled: bool = True) -> None:
        self.set_full_attention_capture_layers(enabled, layer_mode="last")

    def get_last_layer_full_attention_mean(self):
        attn_means = self.get_full_attention_means_per_layer(layer_mode="last")
        return attn_means[0] if attn_means else None

    def set_last_layer_full_attention_extra_bias(self, extra_bias) -> None:
        self.set_full_attention_extra_biases([extra_bias], layer_mode="last")

    def disable_hop_mass_tracking(self) -> None:
        for layer in self.layers:
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "disable_hop_mass_tracking"):
                core_attn.disable_hop_mass_tracking()

    def set_activation_checkpoint(self, enabled: bool = True, mode: str | None = None) -> None:
        mode_aliases = {
            None: "none",
            "none": "none",
            "all": "layer",
            "layer": "layer",
            "ffn_only": "ffn_only",
            "adaptive": "comm_aware",
            "comm_aware": "comm_aware",
        }
        resolved_mode = mode_aliases.get(mode)
        if resolved_mode is None:
            raise ValueError(
                f"activation_checkpoint_mode must be 'all', 'ffn_only', or 'adaptive', got {mode!r}"
            )
        enabled = bool(enabled) and resolved_mode != "none"
        self.activation_checkpoint = enabled
        self.activation_checkpoint_mode = resolved_mode if enabled else "none"
        if self.activation_checkpoint_mode == "comm_aware":
            self._comm_ckpt = _CommAwareCheckpointer(len(self.layers))
        else:
            self._comm_ckpt = None

    def comm_aware_notify_step_end(self, device) -> None:
        """Call from the training loop after backward + optimizer.step() + lr_scheduler.step().

        Advances the comm-aware checkpointer state machine.  This is a no-op
        when comm_aware mode is not active.
        """
        if self._comm_ckpt is not None:
            self._comm_ckpt.notify_step_end(device)

    def reset_hop_mass_stats(self) -> None:
        for layer in self.layers:
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "reset_hop_mass_stats"):
                core_attn.reset_hop_mass_stats()

    def get_hop_mass_stats_per_layer(self):
        stats = []
        for layer in self.layers:
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "get_hop_mass_stats"):
                stats.append(core_attn.get_hop_mass_stats())
            else:
                stats.append(None)
        return stats



    def forward(self, x, attn_bias, edge_index, perturb=None, attn_type=None):
        # x → [bs=1, s/p, x_d]
        x = x.unsqueeze(0)
        n_graph = x.shape[0]

        # [bs, s/p, x_d] → [bs, s/p, h]
        node_feature = self.node_encoder(x)

        if perturb is not None:
            node_feature += perturb

        if self.num_global_node > 0:
            # [bs, num_global_node, h]
            global_node_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
            # [b, s/p + num_global_node, h]
            node_feature = torch.cat([global_node_feature, node_feature], dim=1)

        output = self.input_dropout(node_feature)

        if attn_bias is not None:
            # attn_bias: [sub_s, full_s, attn_bias_dim]  (returned by get_batch_blockize)
            # Project to per-head scalars: [sub_s, full_s, num_heads]
            attn_bias_proj = self.attn_bias_proj(attn_bias.to(output.device))  # [sub_s, full_s, num_heads]

            if self.num_global_node > 0:
                # Also build virtual-distance bias for global token (same shape as graph_token_virtual_distance)
                # vd: [1, attn_bias_dim] → [1, num_heads]
                vd = self.attn_bias_proj(self.graph_token_virtual_distance.weight)  # [num_global, num_heads]

                sub_s = attn_bias_proj.size(0)
                full_s = attn_bias_proj.size(1)

                # Build [sub_s+num_gl, full_s+num_gl, num_heads] graph_attn_bias
                # Row for global token: constant vd repeated
                global_row = vd.unsqueeze(1).expand(self.num_global_node, full_s + self.num_global_node, self.num_heads)
                # Col for global token: constant vd repeated
                global_col = vd.unsqueeze(0).expand(sub_s, self.num_global_node, self.num_heads)
                # Main block + global col  → [sub_s, full_s+num_gl, num_heads]
                main_with_gcol = torch.cat([global_col, attn_bias_proj], dim=1)
                # Stack global row on top → [sub_s+num_gl, full_s+num_gl, num_heads]
                graph_attn_bias = torch.cat([global_row, main_with_gcol], dim=0)
                # Add batch dim expected by encoder layers (batch_size=1)
                graph_attn_bias = graph_attn_bias.unsqueeze(0)
            else:
                graph_attn_bias = attn_bias_proj.unsqueeze(0)
        else:
            graph_attn_bias = None

        # transformer encoder
        use_ckpt = self.activation_checkpoint and self.training and torch.is_grad_enabled()
        ckpt_mode = getattr(self, "activation_checkpoint_mode", "none")

        # comm_aware: plan() only resets peak stats (WARMUP/CALIBRATE) or is a
        # no-op (ACTIVE). Modes are set by notify_step_end() after each optimizer
        # step, where the true peak — including optimizer state and all transient
        # allocations — has been fully recorded.
        if use_ckpt and ckpt_mode == "comm_aware" and self._comm_ckpt is not None:
            self._comm_ckpt.plan(output.device)

        for i, enc_layer in enumerate(self.layers):
            if not use_ckpt:
                output = enc_layer(output, attn_bias=graph_attn_bias, edge_index=edge_index, attn_type=attn_type)
                continue

            # Resolve per-layer mode: comm_aware assigns each layer independently.
            if ckpt_mode == "comm_aware" and self._comm_ckpt is not None:
                layer_mode = self._comm_ckpt.mode(i)
            else:
                layer_mode = ckpt_mode  # "layer" or "ffn_only" applies uniformly

            if layer_mode in ("keep_mha", "ffn_only"):
                # MHA runs without checkpointing: autograd retains post-A2A Q/K/V
                # and sparse-attention outputs, so backward never re-executes any
                # A2A collective or the graph-edge scatter-gather.
                y = enc_layer.self_attention_norm(output)
                y = enc_layer.self_attention(y, attn_bias=graph_attn_bias, edge_index=edge_index, attn_type=attn_type)
                y = enc_layer.self_attention_dropout(y)
                output = output + y
                # Only the FFN block is checkpointed: 2 dense linears + LayerNorm,
                # cheap to recompute and free of any communication.
                def _ffn(h, layer=enc_layer):
                    y = layer.ffn_norm(h)
                    y = layer.ffn(y)
                    y = layer.ffn_dropout(y)
                    return h + y
                output = checkpoint(_ffn, output)
            else:
                # "full_layer" / "layer": original behaviour.
                def _full(h, layer=enc_layer):
                    return layer(h, attn_bias=graph_attn_bias, edge_index=edge_index, attn_type=attn_type)
                output = checkpoint(_full, output)

        # Snapshot live memory right after the encoder loop, before backward.
        # All keep_mha autograd tensors are live here; this is the measurement
        # point for M_layer calibration.
        if use_ckpt and ckpt_mode == "comm_aware" and self._comm_ckpt is not None:
            self._comm_ckpt.record_post_forward_memory(output.device)

        output = self.final_ln(output)

        # output part
        start = int(self.num_global_node)
        output = self.downstream_out_proj(output[0, start:, :])
        return F.log_softmax(output, dim=1)


