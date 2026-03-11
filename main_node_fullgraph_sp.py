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
import os
import time
import random
import contextlib
import torch.distributed as dist

from models.gt_dist_node_level import GT
from models.graphormer_dist_node_level import Graphormer
from utils.lr import PolynomialDecayLR
import argparse

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
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_default_split(dataset_name: str, root_dir: str, wait_for_rank0: bool = False):
    import os
    # 1. 如果已经有预先保存好的划分，直接加载
    split_path = os.path.join(root_dir, dataset_name, 'split_idx.pt')
    if os.path.exists(split_path):
        if wait_for_rank0 and dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
        split_idx = torch.load(split_path, map_location='cpu', weights_only=True)
        if wait_for_rank0 and dist.is_initialized() and dist.get_rank() == 0:
            dist.barrier()
        return split_idx

    name = dataset_name
    
    # 2. 如果是 ogbn 数据集，使用 ogb 官方包
    if name.startswith("ogbn-"):
        try:
            from ogb.nodeproppred import NodePropPredDataset
        except Exception:
            return None
        if wait_for_rank0 and dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
        dataset = NodePropPredDataset(name=name, root=root_dir)
        split_idx = dataset.get_idx_split()
        out = {}
        for key, val in split_idx.items():
            if isinstance(val, torch.Tensor):
                out[key] = val.to(torch.long)
            else:
                out[key] = torch.tensor(val, dtype=torch.long)
        if wait_for_rank0 and dist.is_initialized() and dist.get_rank() == 0:
            dist.barrier()
        return out

    # 3. 尝试从 PyTorch Geometric 加载其它数据集的默认划分
    if wait_for_rank0 and dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
        
    out = None
    try:
        data = None
        if name in ['cora', 'citeseer', 'pubmed']:
            from torch_geometric.datasets import Planetoid
            dataset = Planetoid(root=root_dir, name=name)
            data = dataset[0]
        elif name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
            from torch_geometric.datasets import HeterophilousGraphDataset
            pyg_name = name.capitalize()
            dataset = HeterophilousGraphDataset(root=root_dir, name=pyg_name)
            data = dataset[0]
            
        if data is not None and hasattr(data, 'train_mask'):
            def mask_to_idx(mask):
                if mask.dim() == 2:
                    mask = mask[:, 0]  # 对于多维 mask 默认取第一列
                return torch.nonzero(mask, as_tuple=True)[0].to(torch.long)
            
            out = {
                "train": mask_to_idx(data.train_mask),
                "valid": mask_to_idx(data.val_mask),
                "test": mask_to_idx(data.test_mask)
            }
    except Exception:
        pass

    if wait_for_rank0 and dist.is_initialized() and dist.get_rank() == 0:
        dist.barrier()
        
    return out


def _load_data(args):
    data_path = args.dataset_dir + args.dataset
    feature = torch.load(data_path + "/x.pt")
    y = torch.load(data_path + "/y.pt")
    edge_index_global = torch.load(data_path + "/edge_index.pt")
    N = feature.shape[0]

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
    elif args.model == "gt":
        model = GT(
            **common,
            num_global_node=0,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model


def _build_and_broadcast_edges(args, edge_index_global, N, device, sp_group,
                                sp_src_rank, sp_rank, nodes_per_rank):
    """Build sparse-attention edges on src_rank and broadcast to all ranks."""
    if sp_rank == sp_src_rank:
        parts = []
        if getattr(args, "include_self_loops", 0):
            self_loop = torch.arange(N, device=device, dtype=torch.long)
            self_loop = torch.stack([self_loop, self_loop], dim=0)
            parts.append(self_loop)
        if getattr(args, "include_real_edges", 0):
            parts.append(edge_index_global.to(device))
        if getattr(args, "head_hop_walks_per_node", 2) > 0:
            rw_heads = build_head_hop_edges(
                edge_index=edge_index_global.to(device),
                num_nodes=N,
                num_heads=args.num_heads,
                num_groups=1,
                device=device,
                walk_length=getattr(args, "head_hop_walk_length", 4),
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            )
            if isinstance(rw_heads, list):
                rw_heads = _merge_edge_index_list(rw_heads)
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
             device, sp_group, sp_src_rank, sp_rank, sp_world_size,
             rank_start, rank_end, local_N):
    """Full-graph SP evaluation.

    Runs in model.train() mode so _SeqAllToAll is triggered and each rank
    computes its local nodes with full graph attention context.
    Predictions are all-gathered on rank-0.
    """
    # Build eval edges (fixed seed, rank-0 builds and broadcasts)
    with fixed_random_seed(args.seed):
        edge_index_eval = _build_and_broadcast_edges(
            args, edge_index_global, N, device,
            sp_group, sp_src_rank, sp_rank, local_N,
        )

    # Forward in train mode so all-to-all fires
    was_training = model.training
    model.train()
    drop_states = _set_dropout_eval(model)
    with torch.no_grad():
        x_local = feature[rank_start:rank_end].float().to(device)
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

    if args.rank == 0:
        valid_n = min(int(pred_global.size(0)), int(N))
        pred_global = pred_global[:valid_n]
        y_cpu = y[:valid_n].cpu().view(-1)
        accs = {}
        for sname, idx in split_idx.items():
            idx_valid = idx[idx < valid_n]
            correct = (pred_global[idx_valid] == y_cpu[idx_valid]).sum().item()
            accs[sname] = float(correct) / max(1, len(idx_valid))
        return accs
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full-Graph Node-Level Training with Sequence Parallel."
    )

    # Data
    parser.add_argument("--dataset_dir", type=str, default="./dataset/")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")

    # Model
    parser.add_argument("--model", type=str, default="graphormer")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--attn_bias_dim", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--input_dropout_rate", type=float, default=0.1)
    parser.add_argument("--attention_dropout_rate", type=float, default=0.5)
    parser.add_argument("--attn_type", type=str, default="sparse")

    # Training
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--peak_lr", type=float, default=1e-3)
    parser.add_argument("--end_lr", type=float, default=1e-9)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_updates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--model_dir", type=str, default="./model_ckpt/")

    # Random-walk edges
    parser.add_argument("--head_hop_walk_length", type=int, default=4)
    parser.add_argument("--head_hop_walks_per_node", type=int, default=2)
    parser.add_argument("--include_real_edges", type=int, default=0)
    parser.add_argument("--include_self_loops", type=int, default=0)

    # Distributed (reuse main_node_sp.py args)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--distributed-backend", default="nccl",
                        choices=["nccl", "gloo", "ccl"])
    parser.add_argument("--distributed-timeout-minutes", type=int, default=10)
    parser.add_argument("--sequence-parallel-size", type=int, default=1)

    # Compat
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--num_global_node", type=int, default=1,
                        help="Number of virtual graph tokens used by Graphormer.")

    args = parser.parse_args()
    if args.model != "graphormer":
        args.num_global_node = 0
    elif args.num_global_node not in (0, 1):
        raise ValueError("main_node_fullgraph_sp.py currently supports at most one Graphormer virtual node.")

    # ── Distributed init ────────────────────────────────────────────────────
    initialize_distributed(args)

    sp_initialized = sequence_parallel_is_initialized()
    sp_world_size = get_sequence_parallel_world_size() if sp_initialized else 1
    sp_rank = get_sequence_parallel_rank() if sp_initialized else 0
    sp_src_rank = get_sequence_parallel_src_rank() if sp_initialized else 0
    sp_group = get_sequence_parallel_group() if sp_initialized else None

    device = _resolve_device()
    _set_seed(args.seed)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    feature, y, edge_index_global, N, split_idx = _load_data(args)
    feature = feature.float()

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

    if args.rank == 0:
        print(f"\n{'='*72}")
        print(f"Full-Graph SP Training  (sp_world_size={sp_world_size})")
        print(f"  Nodes per rank: {nodes_per_rank:,}  "
              f"(rank {sp_rank}: {rank_start:,}–{rank_end:,})")
        print(f"  Edge_index: full graph E={edge_index_global.shape[1]:,}")
        print(f"  Sparse edges: real={int(bool(args.include_real_edges))} "
              f"self={int(bool(args.include_self_loops))} "
              f"rw={int(getattr(args, 'head_hop_walks_per_node', 0) > 0)}")
        print(f"{'='*72}\n")

    # ── Build model ───────────────────────────────────────────────────────
    model = _build_model(args, feature, y, device)
    sync_params_and_buffers(model)

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

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # ── Build edges (rank-0 builds, broadcast to all) ─────────────────
        t_rw = time.time()
        edge_index_rw = _build_and_broadcast_edges(
            args, edge_index_global, N, device,
            sp_group, sp_src_rank, sp_rank, local_N,
        )
        rw_time = time.time() - t_rw

        # ── Forward ────────────────────────────────────────────────────────
        t_fwd = time.time()
        # Each rank inputs its slice [local_N, d]; after _SeqAllToAll inside
        # DistributedAttentionNodeLevel the full N nodes are visible for
        # sparse attention; output returns to [local_N, classes].
        out_local = model(x_local, None, edge_index_rw, attn_type=args.attn_type)
        fwd_time = time.time() - t_fwd

        # ── Loss (only on this rank's train nodes) ─────────────────────────
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

        # ── Backward ───────────────────────────────────────────────────────
        t_bwd = time.time()
        loss.backward()
        bwd_time = time.time() - t_bwd

        # ── Gradient sync across SP ranks ─────────────────────────────────
        if sp_world_size > 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(sp_world_size)
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=sp_group)

        optimizer.step()
        lr_scheduler.step()

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        epoch_time = time.time() - t_epoch
        if args.rank == 0:
            print(f"Epoch {epoch:04d} | loss={loss_val:.4f} (ema={loss_ema:.4f}) "
                  f"| t={epoch_time:.2f}s "
                  f"(rw={rw_time:.2f}s fwd={fwd_time:.2f}s bwd={bwd_time:.2f}s)")

        # ── Evaluation ─────────────────────────────────────────────────────
        if epoch % args.eval_every == 0:
            t_eval = time.time()
            accs = _eval_sp(
                args, model, feature, y, split_idx, edge_index_global, N,
                device, sp_group, sp_src_rank, sp_rank, sp_world_size,
                rank_start, rank_end, local_N,
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

        if sp_world_size > 1:
            dist.barrier(group=sp_group)

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


if __name__ == "__main__":
    main()
