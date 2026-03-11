from collections import deque

import torch
import torch.distributed as dist

from gt_sp.utils import build_head_hop_edges, _merge_edge_index_list, gen_sub_edge_index, fixed_random_seed


def estimate_L_star(args, idx_batch_cpu, edge_index_global, num_nodes_global, seed, debug_print=False):
    base_L = max(1, int(args.head_hop_walk_length))
    repeats = max(1, int(getattr(args, "adaptive_eval_repeats", 1)))
    mass_target = float(getattr(args, "full_attn_hop_mass", 0.95))
    mass_target = max(0.0, min(mass_target, 1.0))
    max_hop = int(getattr(args, "full_attn_hop_max_hop", 15))
    num_queries = int(getattr(args, "full_attn_hop_max_queries", 64))

    edge_index_i = gen_sub_edge_index(edge_index_global, idx_batch_cpu, num_nodes_global)
    if edge_index_i is None or edge_index_i.numel() == 0:
        return base_L
    edge_index_i = edge_index_i.to("cpu")
    num_nodes = int(idx_batch_cpu.numel())
    if num_nodes <= 0:
        return base_L

    adj = [[] for _ in range(num_nodes)]
    src = edge_index_i[0].long().tolist()
    dst = edge_index_i[1].long().tolist()
    for s, d in zip(src, dst):
        if s < 0 or d < 0 or s >= num_nodes or d >= num_nodes:
            continue
        adj[s].append(d)
        adj[d].append(s)

    hop_hist = []
    total = 0
    for rep in range(repeats):
        with fixed_random_seed(seed + rep):
            take = min(max(1, num_queries), num_nodes)
            queries = torch.randperm(num_nodes)[:take].tolist()
        for q in queries:
            dist_arr = [-1] * num_nodes
            dist_arr[q] = 0
            q_deque = deque([q])
            while q_deque:
                cur = q_deque.popleft()
                d = dist_arr[cur]
                if max_hop > 0 and d >= max_hop:
                    continue
                for nxt in adj[cur]:
                    if dist_arr[nxt] != -1:
                        continue
                    nd = d + 1
                    dist_arr[nxt] = nd
                    q_deque.append(nxt)
                    if nd <= 0:
                        continue
                    if max_hop > 0 and nd > max_hop:
                        continue
                    if nd > len(hop_hist):
                        hop_hist.extend([0] * (nd - len(hop_hist)))
                    hop_hist[nd - 1] += 1
                    total += 1

    if total <= 0 or not hop_hist:
        return base_L

    if debug_print and (not dist.is_initialized() or dist.get_rank() == 0):
        show_n = min(10, len(hop_hist))
        norm = float(total)
        vals = [h / norm for h in hop_hist[:show_n]]
        val_str = ", ".join(f"{v:.4f}" for v in vals)
        print(f"[adaptive] hop_dist avg (first {show_n}): [{val_str}] (len={len(hop_hist)})")

    cum = 0.0
    for i, count in enumerate(hop_hist):
        cum += count
        if cum / float(total) >= mass_target:
            return i + 1
    return len(hop_hist)


def coverage_for_R(args, idx_batch_cpu, edge_index_global, num_nodes_global, seed, walks_per_node):
    edge_index_i = gen_sub_edge_index(edge_index_global, idx_batch_cpu, num_nodes_global)
    if edge_index_i is None or edge_index_i.numel() == 0:
        return 0.0

    with fixed_random_seed(seed):
        edge_index_i = build_head_hop_edges(
            edge_index=edge_index_i.to("cpu"),
            num_nodes=int(idx_batch_cpu.numel()),
            num_heads=args.num_heads,
            num_groups=1,
            device="cpu",
            walk_length=getattr(args, "head_hop_walk_length", 4),
            walks_per_node=int(walks_per_node),
        )
        if isinstance(edge_index_i, list):
            edge_index_i = _merge_edge_index_list(edge_index_i)

    num_nodes = int(idx_batch_cpu.numel())
    if num_nodes <= 0 or edge_index_i is None or edge_index_i.numel() == 0:
        return 0.0

    nodes = torch.unique(edge_index_i.detach().cpu()[1].view(-1))
    valid = (nodes >= 0) & (nodes < num_nodes)
    covered = int(valid.sum().item())
    return covered / max(1, num_nodes)
