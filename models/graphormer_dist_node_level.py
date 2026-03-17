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
    
    
import os
class CoreAttention(nn.Module):
    """
    Core attention
    """
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim, chunk_size: int | None = None):
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

        env_chunk = os.getenv("TORCH_FULL_ATTN_CHUNK", "").strip()
        self.full_chunk_size = chunk_size if chunk_size else (int(env_chunk) if env_chunk.isdigit() else 0)
        env_sparse_chunk = os.getenv("TORCH_SPARSE_QUERY_CHUNK", "").strip()
        self.sparse_query_chunk_size = int(env_sparse_chunk) if env_sparse_chunk.isdigit() else 0


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
        max_batches: int = 1,
        max_hop: int = 15,
    ) -> None:
        self.track_hop_mass = True
        self.hop_mass_target = float(mass)
        self.hop_mass_max_queries = int(max_queries)
        self.hop_mass_max_batches = int(max_batches)
        self.hop_mass_max_hop = int(max_hop)
        self.reset_hop_mass_stats()

    def disable_hop_mass_tracking(self) -> None:
        self.track_hop_mass = False

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
        edge_hops = None
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.size(0) == 3:
            edge_hops = edge_index[2]
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
            edge_mode = edge_index.get("mode")
            if edge_mode == "cpu_stream":
                return self._sparse_attention_chunked_cpu_stream(
                    k,
                    q,
                    v,
                    edge_index,
                    attn_bias,
                    num_heads,
                )
            if edge_mode == "gpu_chunk":
                return self._sparse_attention_chunked_prepacked(
                    k,
                    q,
                    v,
                    edge_index,
                    attn_bias,
                    num_heads,
                )

        if isinstance(edge_index, list):
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
            if self.sparse_query_chunk_size and self.sparse_query_chunk_size > 0:
                return self._sparse_attention_chunked(k, q, v, edge_index, attn_bias, edge_hops, num_heads)

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

            if getattr(self, "track_hop_mass", False) and edge_hops is not None:
                self._accumulate_sparse_hop_mass(edge_index, edge_hops, score, Z)

            x = wV / (Z + 1e-6)
        
        return x

    def set_sparse_attention_query_chunk_size(self, chunk_size: int | None) -> None:
        self.sparse_query_chunk_size = int(chunk_size) if chunk_size else 0

    def _sparse_attention_chunked(self, k, q, v, edge_index, attn_bias, edge_hops, num_heads):
        src_idx = edge_index[0].to(torch.long)
        dst_idx = edge_index[1].to(torch.long)
        if src_idx.numel() == 0:
            return torch.zeros_like(v)

        order = torch.argsort(dst_idx)
        src_sorted = src_idx[order]
        dst_sorted = dst_idx[order]
        if edge_hops is not None:
            edge_hops = edge_hops[order]

        chunk = max(int(self.sparse_query_chunk_size), 1)
        total_nodes = int(v.size(0))
        wV = None
        Z = None

        for start in range(0, total_nodes, chunk):
            end = min(start + chunk, total_nodes)
            start_t = dst_sorted.new_tensor(start)
            end_t = dst_sorted.new_tensor(end)
            left = int(torch.searchsorted(dst_sorted, start_t, right=False).item())
            right = int(torch.searchsorted(dst_sorted, end_t, right=False).item())
            if right <= left:
                continue

            src_chunk = src_sorted[left:right]
            dst_chunk = dst_sorted[left:right]
            local_dst = dst_chunk - start

            src = k[src_chunk]
            dest = q[dst_chunk]
            score = torch.mul(src, dest)
            score = score / self.scale
            score = score.sum(-1, keepdim=True).clamp(-5, 5)

            if attn_bias is not None:
                score = score + attn_bias[src_chunk, dst_chunk, :].unsqueeze(-1)

            score = torch.exp(score)
            msg = v[src_chunk] * score

            if wV is None:
                wV = msg.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
                Z = score.new_zeros(v.size(0), num_heads, 1)
            chunk_nodes = end - start
            wV_chunk = msg.new_zeros(chunk_nodes, num_heads, self.hidden_size_per_attention_head)
            Z_chunk = score.new_zeros(chunk_nodes, num_heads, 1)
            scatter(msg, local_dst, dim=0, out=wV_chunk, reduce='add')
            scatter(score, local_dst, dim=0, out=Z_chunk, reduce='add')

            wV[start:end] = wV_chunk
            Z[start:end] = Z_chunk

        if wV is None:
            wV = v.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
            Z = v.new_zeros(v.size(0), num_heads, 1)
        return wV / (Z + 1e-6)

    def _sparse_attention_chunked_prepacked(self, k, q, v, edge_payload, attn_bias, num_heads):
        src_sorted = edge_payload["src"]
        dst_sorted = edge_payload["dst"]
        offsets = edge_payload["offsets"]
        chunk = max(int(edge_payload["chunk_size"]), 1)
        edge_hops = edge_payload.get("hops")

        if src_sorted.numel() == 0:
            return torch.zeros_like(v)

        total_nodes = int(v.size(0))
        wV = None
        Z = None

        for chunk_id in range(max(int(offsets.numel()) - 1, 0)):
            left = int(offsets[chunk_id].item())
            right = int(offsets[chunk_id + 1].item())
            if right <= left:
                continue

            start = chunk_id * chunk
            end = min(start + chunk, total_nodes)
            src_chunk = src_sorted[left:right]
            dst_chunk = dst_sorted[left:right]
            local_dst = dst_chunk - start

            src = k[src_chunk]
            dest = q[dst_chunk]
            score = torch.mul(src, dest)
            score = score / self.scale
            score = score.sum(-1, keepdim=True).clamp(-5, 5)

            if attn_bias is not None:
                score = score + attn_bias[src_chunk, dst_chunk, :].unsqueeze(-1)

            score = torch.exp(score)
            msg = v[src_chunk] * score

            if wV is None:
                wV = msg.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
                Z = score.new_zeros(v.size(0), num_heads, 1)
            chunk_nodes = end - start
            wV_chunk = msg.new_zeros(chunk_nodes, num_heads, self.hidden_size_per_attention_head)
            Z_chunk = score.new_zeros(chunk_nodes, num_heads, 1)
            scatter(msg, local_dst, dim=0, out=wV_chunk, reduce='add')
            scatter(score, local_dst, dim=0, out=Z_chunk, reduce='add')

            wV[start:end] = wV_chunk
            Z[start:end] = Z_chunk

            if getattr(self, "track_hop_mass", False) and edge_hops is not None:
                edge_hops_chunk = edge_hops[left:right]
                edge_index_chunk = torch.stack([src_chunk, dst_chunk], dim=0)
                self._accumulate_sparse_hop_mass(edge_index_chunk, edge_hops_chunk, score, Z)

        if wV is None:
            wV = v.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
            Z = v.new_zeros(v.size(0), num_heads, 1)
        return wV / (Z + 1e-6)

    def _sparse_attention_chunked_cpu_stream(self, k, q, v, edge_payload, attn_bias, num_heads):
        src_cpu = edge_payload["src"]
        dst_cpu = edge_payload["dst"]
        offsets_cpu = edge_payload["offsets"]
        chunk = max(int(edge_payload["chunk_size"]), 1)
        edge_hops_cpu = edge_payload.get("hops")

        if src_cpu.numel() == 0:
            return torch.zeros_like(v)

        total_nodes = int(v.size(0))
        num_chunks = (total_nodes + chunk - 1) // chunk
        wV = None
        Z = None

        for chunk_id in range(num_chunks):
            if chunk_id + 1 >= offsets_cpu.numel():
                continue
            left = int(offsets_cpu[chunk_id].item())
            right = int(offsets_cpu[chunk_id + 1].item())
            if right <= left:
                continue

            start = chunk_id * chunk
            end = min(start + chunk, total_nodes)

            src_chunk = src_cpu[left:right].to(k.device, dtype=torch.long, non_blocking=True)
            dst_chunk = dst_cpu[left:right].to(k.device, dtype=torch.long, non_blocking=True)
            local_dst = dst_chunk - start

            src = k[src_chunk]
            dest = q[dst_chunk]
            score = torch.mul(src, dest)
            score = score / self.scale
            score = score.sum(-1, keepdim=True).clamp(-5, 5)

            if attn_bias is not None:
                score = score + attn_bias[src_chunk, dst_chunk, :].unsqueeze(-1)

            score = torch.exp(score)
            msg = v[src_chunk] * score

            if wV is None:
                wV = msg.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
                Z = score.new_zeros(v.size(0), num_heads, 1)
            chunk_nodes = end - start
            wV_chunk = msg.new_zeros(chunk_nodes, num_heads, self.hidden_size_per_attention_head)
            Z_chunk = score.new_zeros(chunk_nodes, num_heads, 1)
            scatter(msg, local_dst, dim=0, out=wV_chunk, reduce='add')
            scatter(score, local_dst, dim=0, out=Z_chunk, reduce='add')

            wV[start:end] = wV_chunk
            Z[start:end] = Z_chunk

            if getattr(self, "track_hop_mass", False) and edge_hops_cpu is not None:
                edge_hops_chunk = edge_hops_cpu[left:right].to(score.device, dtype=torch.long, non_blocking=True)
                edge_index_chunk = torch.stack([src_chunk, dst_chunk], dim=0)
                self._accumulate_sparse_hop_mass(edge_index_chunk, edge_hops_chunk, score, Z)

        if wV is None:
            wV = v.new_zeros(v.size(0), num_heads, self.hidden_size_per_attention_head)
            Z = v.new_zeros(v.size(0), num_heads, 1)
        return wV / (Z + 1e-6)

    def set_full_attention_chunk_size(self, chunk_size: int | None) -> None:
        self.full_chunk_size = int(chunk_size) if chunk_size else 0

    def _full_attention_chunked(self, k, q, v, attn_bias, edge_index=None):
        batch_size, seq_q, num_heads_local, head_dim = q.size()
        seq_k = k.size(1)
        chunk = max(int(self.full_chunk_size), 1)

        q_flat = q.permute(0, 2, 1, 3).contiguous().view(-1, seq_q, head_dim)
        k_flat = k.permute(0, 2, 1, 3).contiguous().view(-1, seq_k, head_dim)
        v_flat = v.permute(0, 2, 1, 3).contiguous().view(-1, seq_k, head_dim)
        k_t = k_flat.transpose(1, 2)

        if attn_bias is not None:
            bias = attn_bias.view(batch_size, num_heads_local, seq_q, seq_k)
            bias = bias.permute(0, 1, 2, 3).contiguous().view(-1, seq_q, seq_k)
        else:
            bias = None

        attn_probs_full = None

        context = q_flat.new_empty((q_flat.size(0), seq_q, head_dim))
        for start in range(0, seq_q, chunk):
            end = min(start + chunk, seq_q)
            q_chunk = q_flat[:, start:end, :] * self.scale
            scores = torch.bmm(q_chunk, k_t)
            if bias is not None:
                scores = scores + bias[:, start:end, :]
            probs = torch.softmax(scores, dim=-1)
            probs = self.att_dropout(probs)
            context[:, start:end, :] = torch.bmm(probs, v_flat)
            if attn_probs_full is not None:
                attn_probs_full[:, start:end, :] = probs

        context = context.view(batch_size, num_heads_local, seq_q, head_dim).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_q, -1)

        return context



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
        attn_probs = torch.softmax(x, dim=3)
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
        max_q = min(q_len, start_q + max(1, self.hop_mass_max_queries))
        max_hop = int(self.hop_mass_max_hop)
        mass_target = max(0.0, min(self.hop_mass_target, 1.0))

        for b in range(bsz):
            for q in range(start_q, max_q):
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
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    @staticmethod
    def _get_core_attention(layer):
        dist_attn = layer.self_attention.dist_attn
        return getattr(dist_attn, "local_attn", dist_attn)

    def enable_hop_mass_tracking(
        self,
        mass: float = 0.95,
        max_queries: int = 64,
        max_batches: int = 1,
        max_hop: int = 15,
    ) -> None:
        for layer in self.layers:
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "enable_hop_mass_tracking"):
                core_attn.enable_hop_mass_tracking(mass, max_queries, max_batches, max_hop)

    def disable_hop_mass_tracking(self) -> None:
        for layer in self.layers:
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "disable_hop_mass_tracking"):
                core_attn.disable_hop_mass_tracking()

    def set_sparse_attention_query_chunk_size(self, chunk_size: int | None) -> None:
        for layer in self.layers:
            core_attn = self._get_core_attention(layer)
            if hasattr(core_attn, "set_sparse_attention_query_chunk_size"):
                core_attn.set_sparse_attention_query_chunk_size(chunk_size)

    def set_activation_checkpoint(self, enabled: bool = True) -> None:
        self.activation_checkpoint = bool(enabled)

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
        for enc_layer in self.layers:
            if self.activation_checkpoint and self.training and torch.is_grad_enabled():
                def custom_forward(hidden_states, layer=enc_layer):
                    return layer(hidden_states, attn_bias=graph_attn_bias, edge_index=edge_index, attn_type=attn_type)
                output = checkpoint(custom_forward, output)
            else:
                output = enc_layer(output, attn_bias=graph_attn_bias, edge_index=edge_index, attn_type=attn_type)
        output = self.final_ln(output)

        # output part
        start = int(self.num_global_node)
        output = self.downstream_out_proj(output[0, start:, :])
        return F.log_softmax(output, dim=1)


