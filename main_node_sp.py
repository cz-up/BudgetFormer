import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
import math
from collections import deque
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from utils.lr import PolynomialDecayLR
import argparse
import scipy.sparse as sp
import os
import time
import random
import torch.distributed as dist
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
    last_batch_flag,
)
from gt_sp.reducer import sync_params_and_buffers
from gt_sp.evaluate import sparse_eval_gpu
from gt_sp.utils import (
    get_batch_blockize,
    build_head_hop_edges,
    _merge_edge_index_list,
    gen_sub_edge_index,
    random_split_idx,
    fixed_random_seed,
)
from utils.parser_node_level import parser_add_main_args


def _load_ogbn_split(dataset_name: str, root_dir: str, wait_for_rank0: bool = False):
    name = dataset_name
    if name.endswith("_n"):
        name = name[:-2]
    if not name.startswith("ogbn-"):
        return None
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


def run_step(args, model, device, feature, y, idx_batch_cpu, sub_split_seq_lens, edge_index_global, N, attn_type):
    x_i, y_i, edge_index_i, attn_bias = get_batch_blockize(
        args,
        feature,
        y,
        idx_batch_cpu,
        sub_split_seq_lens,
        edge_index_global,
        N,
        device=device,
    )

    x_i = x_i.to(device)
    y_i = y_i.to(device)
    if isinstance(edge_index_i, list):
        edge_index_i = [e.to(device) for e in edge_index_i]
    else:
        edge_index_i = edge_index_i.to(device)

    out_i = model(x_i, attn_bias, edge_index_i, attn_type=attn_type)
    loss = F.nll_loss(out_i, y_i.long())
    return loss, out_i, y_i


def _eval_with_params(
    args,
    model,
    feature,
    y,
    split_idx,
    edge_index_global,
    device,
    walk_length=None,
    walks_per_node=None,
    repeats=1,
):
    orig_len = args.head_hop_walk_length
    orig_walks = args.head_hop_walks_per_node
    if walk_length is not None:
        args.head_hop_walk_length = int(walk_length)
    if walks_per_node is not None:
        args.head_hop_walks_per_node = int(walks_per_node)
    acc_sum = 0.0
    try:
        for r in range(max(1, int(repeats))):
            with fixed_random_seed(args.seed + r):
                acc_sum += sparse_eval_gpu(
                    args,
                    model,
                    feature,
                    y,
                    split_idx,
                    None,
                    edge_index_global,
                    device,
                )
    finally:
        args.head_hop_walk_length = orig_len
        args.head_hop_walks_per_node = orig_walks
    return acc_sum / max(1, int(repeats))


def _forward_outputs(
    args,
    model,
    device,
    feature,
    y,
    idx_batch_cpu,
    sub_split_seq_lens,
    edge_index_global,
    N,
    attn_type,
):
    x_i, y_i, edge_index_i, attn_bias = get_batch_blockize(
        args,
        feature,
        y,
        idx_batch_cpu,
        sub_split_seq_lens,
        edge_index_global,
        N,
        device=device,
    )
    x_i = x_i.to(device)
    y_i = y_i.to(device)
    if isinstance(edge_index_i, list):
        edge_index_i = [e.to(device) for e in edge_index_i]
    else:
        edge_index_i = edge_index_i.to(device)
    out_i = model(x_i, attn_bias, edge_index_i, attn_type=attn_type)
    return out_i, y_i


def _estimate_L_star(
    args,
    idx_batch_cpu,
    edge_index_global,
    N,
    seed,
    debug_print=False,
):
    base_L = max(1, int(args.head_hop_walk_length))
    repeats = max(1, int(getattr(args, "adaptive_eval_repeats", 1)))
    mass_target = float(getattr(args, "full_attn_hop_mass", 0.95))
    mass_target = max(0.0, min(mass_target, 1.0))
    max_hop = int(getattr(args, "full_attn_hop_max_hop", 15))
    num_queries = int(getattr(args, "full_attn_hop_max_queries", 64))

    edge_index_i = gen_sub_edge_index(edge_index_global, idx_batch_cpu, N)
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

    if debug_print:
        if not dist.is_initialized() or dist.get_rank() == 0:
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


def _coverage_for_R(
    args,
    device,
    feature,
    y,
    idx_batch_cpu,
    sub_split_seq_lens,
    edge_index_global,
    N,
    attn_type,
    seed,
    walks_per_node,
):
    base_R = int(args.head_hop_walks_per_node)
    try:
        with fixed_random_seed(seed):
            args.head_hop_walks_per_node = int(walks_per_node)
            edge_index_i = gen_sub_edge_index(edge_index_global, idx_batch_cpu, N)
            if edge_index_i is None or edge_index_i.numel() == 0:
                return 0.0
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
    finally:
        args.head_hop_walks_per_node = base_R
    num_nodes = int(idx_batch_cpu.numel())
    if num_nodes <= 0 or edge_index_i is None:
        return 0.0
    if edge_index_i.numel() == 0:
        return 0.0
    nodes = torch.unique(edge_index_i.detach().cpu()[1].view(-1))
    valid = (nodes >= 0) & (nodes < num_nodes)
    covered = int(valid.sum().item())
    return covered / max(1, num_nodes)


def main():
    parser = argparse.ArgumentParser(description="TorchGT node-level training (SP).")
    parser_add_main_args(parser)
    args = parser.parse_args()

    # Initialize distributed
    initialize_distributed(args)
    device = f"cuda:{torch.cuda.current_device()}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    # Dataset
    feature = torch.load(args.dataset_dir + args.dataset + "/x.pt")  # [N, x_dim]
    y = torch.load(args.dataset_dir + args.dataset + "/y.pt")  # [N]
    edge_index_global = torch.load(args.dataset_dir + args.dataset + "/edge_index.pt")
    N = feature.shape[0]

    if args.dataset == "pokec":
        y = torch.clamp(y, min=0)
    if getattr(args, "use_ogbn_split", False):
        split_idx = _load_ogbn_split(args.dataset, args.dataset_dir, wait_for_rank0=True)
        if split_idx is None:
            if args.rank == 0:
                print("Warning: ogbn split unavailable, fallback to random split.")
            split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)
    else:
        split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)

    if args.rank == 0:
        print(args)
        print("Dataset load successfully")
        print(
            f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}"
        )
        print(
            f"Training iters: {split_idx['train'].size(0) // args.seq_len + 1}, Val iters: {split_idx['valid'].size(0) // args.seq_len + 1}, Test iters: {split_idx['test'].size(0) // args.seq_len + 1}"
        )

    # Broadcast train indexes to all ranks
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()

    train_idx = split_idx["train"]
    if args.rank == 0:
        flatten_train_idx = train_idx.to("cuda")
    else:
        total_numel = train_idx.numel()
        flatten_train_idx = torch.empty(total_numel, device=device, dtype=torch.int64)
    if seq_parallel_world_size > 1:
        dist.broadcast(flatten_train_idx, src_rank, group=group)

    # Initialize global token indices
    seq_len_per_rank = get_sequence_length_per_rank()
    sub_real_seq_len = seq_len_per_rank + args.num_global_node
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))

    # Last batch fix sequence length
    if flatten_train_idx.shape[0] % args.seq_len != 0:
        last_batch_node_num = flatten_train_idx.shape[0] % args.seq_len
        if last_batch_node_num % seq_parallel_world_size != 0:
            div = last_batch_node_num // seq_parallel_world_size
            last_batch_node_num = div * seq_parallel_world_size + (seq_parallel_world_size - 1)

        x_dummy_list = [t for t in torch.tensor_split(torch.zeros(last_batch_node_num,), seq_parallel_world_size, dim=0)]
        sub_split_seq_lens = [t.shape[0] for t in x_dummy_list]  # e.g., [14, 14, 14, 13]
        sub_real_seq_len = max(sub_split_seq_lens) + args.num_global_node
        global_token_indices_last_batch = list(
            range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len)
        )
    else:
        sub_split_seq_lens = None
        global_token_indices_last_batch = None

    set_global_token_indices(global_token_indices)
    set_last_batch_global_token_indices(global_token_indices_last_batch)

    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item() + 1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node,
        ).to(device)
    elif args.model == "gt":
        model = GT(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=feature.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=y.max().item() + 1,
            attn_bias_dim=args.attn_bias_dim,
            dropout_rate=args.dropout_rate,
            input_dropout_rate=args.input_dropout_rate,
            attention_dropout_rate=args.attention_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    if args.rank == 0:
        print("Model params:", sum(p.numel() for p in model.parameters()))

    # Sync params and buffers
    sync_params_and_buffers(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup=args.warmup_updates,
        tot=args.epochs,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0
    )

    loss_list = []
    epoch_t_list = []
    val_acc_list = []
    test_acc_list = []
    best_epoch = -1
    best_val = 0
    best_test = 0
    max_walk_length = None
    max_walks_per_node = None
    adaptive_enabled = getattr(args, "adaptive_walk", False)
    adaptive_stage = "stable"
    adaptive_no_improve = 0
    adaptive_cov_no_improve = 0
    if adaptive_enabled:
        adaptive_stage = "tune_L"

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "reset_head_mass_stats"):
            model.reset_head_mass_stats()
        idx_train_shuffle = flatten_train_idx[torch.randperm(flatten_train_idx.size(0))].to("cpu")

        num_batch = idx_train_shuffle.size(0) // args.seq_len + 1
        batch_indices = torch.split(idx_train_shuffle, args.seq_len)

        iter_t_list = []
        t_epoch_start = time.time()
        iter_idx = 1
        needed_L_sum = 0.0
        needed_L_count = 0
        cov_diff_sum = 0.0
        cov_diff_count = 0
        do_diff_check = adaptive_enabled and adaptive_stage == "tune_L" and args.rank == 0
        do_cov_check = adaptive_enabled and adaptive_stage == "tune_R" and args.rank == 0
        for idx_batch_cpu in batch_indices:
            t_iter_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            attn_type = args.attn_type
            loss, out_i, y_i = run_step(
                args,
                model,
                device,
                feature,
                y,
                idx_batch_cpu,
                sub_split_seq_lens,
                edge_index_global,
                N,
                attn_type,
            )
            loss.backward()

            for _, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad.div_(get_sequence_parallel_world_size())
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())

            optimizer.step()
            iter_t_list.append(time.time() - t_iter_start)
            if do_diff_check and iter_idx <= args.adaptive_embed_batches:
                needed_L = _estimate_L_star(
                    args,
                    idx_batch_cpu,
                    edge_index_global,
                    N,
                    seed=args.seed + epoch * 1000 + iter_idx,
                    debug_print=(iter_idx == 1),
                )
                needed_L_sum += float(needed_L)
                needed_L_count += 1
            if do_cov_check and iter_idx <= args.adaptive_embed_batches:
                base_R = int(args.head_hop_walks_per_node)
                cov_R = _coverage_for_R(
                    args,
                    device,
                    feature,
                    y,
                    idx_batch_cpu,
                    sub_split_seq_lens,
                    edge_index_global,
                    N,
                    attn_type,
                    seed=args.seed + epoch * 2000 + iter_idx,
                    walks_per_node=base_R,
                )
                cov_Rp1 = _coverage_for_R(
                    args,
                    device,
                    feature,
                    y,
                    idx_batch_cpu,
                    sub_split_seq_lens,
                    edge_index_global,
                    N,
                    attn_type,
                    seed=args.seed + epoch * 2000 + iter_idx,
                    walks_per_node=base_R + 1,
                )
                cov_diff_sum += (cov_Rp1 - cov_R)
                cov_diff_count += 1
            iter_idx += 1

        loss_list.append(loss.item())
        lr_scheduler.step()
        if args.rank == 0:
            epoch_t_list.append(np.sum(iter_t_list))
            print(
                f"Epoch: {epoch:03d}, Loss: {np.mean(loss_list):.4f}, Epoch Time: {np.mean(epoch_t_list):.3f}s"
            )
        train_loss = np.mean(loss_list)

        val_acc = None
        avg_needed_L = None
        avg_cov_diff = None
        if adaptive_enabled and do_diff_check:
            avg_needed_L = float(needed_L_sum / max(1, needed_L_count))
        if adaptive_enabled and do_cov_check:
            avg_cov_diff = float(cov_diff_sum / max(1, cov_diff_count))

        if args.rank == 0 and adaptive_enabled:
            if avg_needed_L is not None and adaptive_stage == "tune_L":
                base_L = int(args.head_hop_walk_length)
                if avg_needed_L <= base_L:
                    adaptive_no_improve += 1
                    print(f"[adaptive] keep L={base_L} (L* {avg_needed_L:.2f})")
                else:
                    next_L = max(base_L + 1, int(math.ceil(avg_needed_L)))
                    args.head_hop_walk_length = next_L
                    adaptive_no_improve = 0
                    print(f"[adaptive] increase L to {args.head_hop_walk_length} (L* {avg_needed_L:.2f})")
                if adaptive_no_improve >= args.adaptive_patience:
                    adaptive_stage = "tune_R"
                    adaptive_cov_no_improve = 0
                    print(f"[adaptive] L fixed at {args.head_hop_walk_length}, start tuning R")

        if args.rank == 0 and adaptive_enabled and adaptive_stage == "tune_R" and avg_cov_diff is not None:
            if avg_cov_diff <= args.adaptive_cov_delta:
                adaptive_cov_no_improve += 1
                print(f"[adaptive] keep R={args.head_hop_walks_per_node} (cov diff {avg_cov_diff:.6f})")
            else:
                args.head_hop_walks_per_node += 1
                adaptive_cov_no_improve = 0
                print(f"[adaptive] increase R to {args.head_hop_walks_per_node} (cov diff {avg_cov_diff:.6f})")
            if adaptive_cov_no_improve >= args.adaptive_patience:
                adaptive_stage = "stable"
                print(f"[adaptive] R fixed at {args.head_hop_walks_per_node} (cov diff {avg_cov_diff:.6f})")

        if adaptive_enabled and dist.is_initialized():
            if args.rank == 0:
                stage_id = 0 if adaptive_stage == "tune_L" else 1 if adaptive_stage == "tune_R" else 2
                state = torch.tensor(
                    [args.head_hop_walk_length, args.head_hop_walks_per_node, stage_id],
                    device=device,
                    dtype=torch.long,
                )
            else:
                state = torch.empty(3, device=device, dtype=torch.long)
            dist.broadcast(state, 0)
            args.head_hop_walk_length = int(state[0].item())
            args.head_hop_walks_per_node = int(state[1].item())
            stage_id = int(state[2].item())
            adaptive_stage = "tune_L" if stage_id == 0 else "tune_R" if stage_id == 1 else "stable"

        if epoch % 5 == 0:
            stats = None
            t4 = time.time()
            with fixed_random_seed(args.seed):
                train_acc = sparse_eval_gpu(args, model, feature, y, split_idx["train"], None, edge_index_global, device)
            if val_acc is None:
                with fixed_random_seed(args.seed):
                    val_acc = sparse_eval_gpu(args, model, feature, y, split_idx["valid"], None, edge_index_global, device)
            stats_needed = args.full_attn_hop_stats and hasattr(model, "enable_hop_mass_tracking")
            if stats_needed:
                model.reset_hop_mass_stats()
                model.enable_hop_mass_tracking(
                    mass=args.full_attn_hop_mass,
                    max_queries=args.full_attn_hop_max_queries,
                    max_batches=args.full_attn_hop_max_batches,
                    max_hop=args.full_attn_hop_max_hop,
                )
                with fixed_random_seed(args.seed):
                    _ = sparse_eval_gpu(args, model, feature, y, split_idx["valid"], None, edge_index_global, device)
                stats = model.get_hop_mass_stats_per_layer()
                model.disable_hop_mass_tracking()

            with fixed_random_seed(args.seed):
                test_acc = sparse_eval_gpu(args, model, feature, y, split_idx["test"], None, edge_index_global, device)
            t5 = time.time()
            if args.rank == 0:
                print("------------------------------------------------------------------------------------")
                print(f'Eval time {t5-t4}s')
                print("Epoch: {:03d}, Loss: {:4f}, Train acc: {:.2%}, Val acc: {:.2%}, Test acc: {:.2%}, Epoch Time: {:.3f}s".format(
                    epoch, np.mean(loss_list), train_acc, val_acc, test_acc, np.mean(epoch_t_list)))
                if stats is not None:
                    print("Attention hop-mass stats (per layer):")
                    for i, stat in enumerate(stats):
                        if stat is None:
                            print(f"  layer {i}: no data")
                            continue
                        hop_sums, hop_count, max_sum, max_count, max_hop = stat
                        mean_max_hop = max_sum / max(1, max_count)
                        if hop_count > 0:
                            ratios = [h / hop_count for h in hop_sums]
                            ratio_str = ", ".join(f"{r:.3f}" for r in ratios)
                            print(
                                f"  layer {i}: mean_max_hop={mean_max_hop:.3f}, max_hop={max_hop}, "
                                f"hop_ratios=[{ratio_str}]"
                            )
                        else:
                            print(f"  layer {i}: mean_max_hop={mean_max_hop:.3f}, max_hop={max_hop}, hop_ratios=NA")
                print("------------------------------------------------------------------------------------")

                if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = epoch
                    if args.save_model:
                        torch.save(model.state_dict(), args.model_dir + f"{args.dataset}.pkl")

                if test_acc > best_test:
                    best_test = test_acc

                val_acc_list.append(val_acc)
                test_acc_list.append(test_acc)

        if seq_parallel_world_size > 1:
            dist.barrier()
        

    if args.rank == 0:
        print(f"Best epoch: {best_epoch}, validation accuracy: {best_val:.2%}, test accuracy: {best_test:.2%}")
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        peak_tensor = torch.tensor([peak_alloc, peak_reserved], device=device, dtype=torch.long)
        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered = [torch.zeros_like(peak_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, peak_tensor)
            if args.rank == 0:
                print("Peak GPU memory per rank (MiB):")
                for r, t in enumerate(gathered):
                    alloc_mib = t[0].item() / (1024 ** 2)
                    reserved_mib = t[1].item() / (1024 ** 2)
                    print(f"  rank {r}: allocated={alloc_mib:.2f}, reserved={reserved_mib:.2f}")
        else:
            alloc_mib = peak_alloc / (1024 ** 2)
            reserved_mib = peak_reserved / (1024 ** 2)
            print(f"Peak GPU memory (MiB): allocated={alloc_mib:.2f}, reserved={reserved_mib:.2f}")


if __name__ == "__main__":
    main()
