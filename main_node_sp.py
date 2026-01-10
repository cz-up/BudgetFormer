import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
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
    random_split_idx,
    get_batch_reorder_blockize,
    fixed_random_seed,
    gen_sub_edge_index,
    build_head_hop_edges,
    edge_index_new_edge_ratio,
)
from utils.parser_node_level import parser_add_main_args
import math


def run_step(args, model, device, feature, y, idx_batch_cpu, sub_split_seq_lens, edge_index_global, N, attn_type):
    if getattr(args, "head_hop_edges", False):
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
    else:
        metis_k = getattr(args, "metis_k", 1)
        x_i, y_i, edge_index_i, attn_bias = get_batch_reorder_blockize(
            args,
            feature,
            y,
            idx_batch_cpu,
            sub_split_seq_lens,
            device,
            edge_index_global,
            N,
            metis_k,
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

    def _set_auto_rw_params(edge_index: torch.Tensor, num_nodes: int) -> None:
        num_edges = edge_index.size(1)
        avg_degree = num_edges / max(1, num_nodes)
        deg_base = max(avg_degree, 2.0)
        est_diameter = math.log(max(num_nodes, 2)) / math.log(deg_base)
        walks = int(round(max(1.0, avg_degree) * getattr(args, "head_rw_walks_factor", 1.0)))
        length = int(round(max(1.0, est_diameter) * getattr(args, "head_rw_length_factor", 1.0)))
        args.head_rw_walks = max(1, walks)
        args.head_rw_length = max(1, length)

    # Dataset
    feature = torch.load(args.dataset_dir + args.dataset + "/x.pt")  # [N, x_dim]
    y = torch.load(args.dataset_dir + args.dataset + "/y.pt")  # [N]
    edge_index_global = torch.load(args.dataset_dir + args.dataset + "/edge_index.pt")
    N = feature.shape[0]
    _set_auto_rw_params(edge_index_global, N)

    if args.dataset == "pokec":
        y = torch.clamp(y, min=0)
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

    stage = 1
    no_length_increase = 0
    stable_walks = 0
    prev_new_ratio = None
    prev_far_near_ratio = None
    max_walk_length = getattr(args, "head_hop_walk_length_max", None)
    max_walks_per_node = getattr(args, "head_hop_walks_per_node_max", None)
    far_near_margin = getattr(args, "head_hop_far_near_margin", 0.02)

    def _sync_stage_state():
        if not dist.is_initialized():
            return stage, no_length_increase, stable_walks
        payload = torch.tensor(
            [
                args.head_hop_walk_length,
                args.head_hop_walks_per_node,
                stage,
                no_length_increase,
                stable_walks,
            ],
            device=device,
            dtype=torch.long,
        )
        dist.broadcast(payload, 0)
        args.head_hop_walk_length = int(payload[0].item())
        args.head_hop_walks_per_node = int(payload[1].item())
        synced_stage = int(payload[2].item())
        synced_no_inc = int(payload[3].item())
        synced_stable = int(payload[4].item())
        return synced_stage, synced_no_inc, synced_stable

    def _build_edges_for_walks(idx_nodes, walk_length, walks_per_node):
        edge_index_i = gen_sub_edge_index(edge_index_global, idx_nodes, N)
        return build_head_hop_edges(
            edge_index=edge_index_i,
            num_nodes=idx_nodes.size(0),
            num_heads=args.num_heads,
            num_groups=getattr(args, "head_groups", args.num_heads),
            device=edge_index_i.device,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
        )
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
    prev_loss = None
    prev_walk_length = getattr(args, "head_hop_walk_length", 4)
    prev_walks_per_node = getattr(args, "head_hop_walks_per_node", 2)
    max_walk_length = None
    max_walks_per_node = None

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
            torch.cuda.synchronize()
            iter_t_list.append(time.time() - t_iter_start)
            iter_idx += 1

        loss_list.append(loss.item())
        lr_scheduler.step()
        if args.rank == 0:
            epoch_t_list.append(np.sum(iter_t_list))
            print(
                f"Epoch: {epoch:03d}, Loss: {np.mean(loss_list):.4f}, Epoch Time: {np.mean(epoch_t_list):.3f}s"
            )
        train_loss = np.mean(loss_list)
        if getattr(args, "head_hop_edges", False) and getattr(args, "head_hop_dynamic", False) and hasattr(model, "get_head_mass_stats"):
            stats = model.get_head_mass_stats()
            if stats is not None and args.head_groups >= 2:
                num_heads = stats.numel()
                groups = min(args.head_groups, num_heads)
                heads_per_group = (num_heads + groups - 1) // groups
                group_vals = [[] for _ in range(groups)]
                for h in range(num_heads):
                    g = min(h // heads_per_group, groups - 1)
                    group_vals[g].append(float(stats[h]))
                near = np.mean(group_vals[0]) if group_vals[0] else 0.0
                far = np.mean(group_vals[-1]) if group_vals[-1] else 0.0
                if prev_loss is not None and train_loss > prev_loss:
                    args.head_hop_walk_length = prev_walk_length
                    args.head_hop_walks_per_node = prev_walks_per_node
                else:
                    prev_walk_length = args.head_hop_walk_length
                    prev_walks_per_node = args.head_hop_walks_per_node
                    if far > near:
                        args.head_hop_walk_length = min(max_walk_length, args.head_hop_walk_length + 1)
                    else:
                        args.head_hop_walk_length = max(1, args.head_hop_walk_length - 1)
                        args.head_hop_walks_per_node = min(
                            max_walks_per_node, args.head_hop_walks_per_node + 1
                        )
                if args.rank == 0:
                    print(
                        f"Head-hop adjust: near={near:.4f} far={far:.4f} "
                        f"walk_length={args.head_hop_walk_length} "
                        f"walks_per_node={args.head_hop_walks_per_node}"
                    )
        prev_loss = train_loss

        if getattr(args, "head_hop_edges", False) and args.rank == 0:
            if stage == 1:
                stats = model.get_head_mass_stats() if hasattr(model, "get_head_mass_stats") else None
                if stats is None or args.head_groups < 2:
                    no_length_increase += 1
                    if args.rank == 0:
                        print("Stage1: skip (no head mass stats or head_groups < 2)")
                else:
                    num_heads = stats.numel()
                    groups = min(args.head_groups, num_heads)
                    heads_per_group = (num_heads + groups - 1) // groups
                    group_vals = [[] for _ in range(groups)]
                    for h in range(num_heads):
                        g = min(h // heads_per_group, groups - 1)
                        group_vals[g].append(float(stats[h]))
                    near = np.mean(group_vals[0]) if group_vals[0] else 0.0
                    far = np.mean(group_vals[-1]) if group_vals[-1] else 0.0
                    ratio = far / (near + 1e-6)
                    if prev_far_near_ratio is not None and ratio > prev_far_near_ratio + far_near_margin:
                        next_len = args.head_hop_walk_length + 1
                        if max_walk_length is None or next_len <= max_walk_length:
                            args.head_hop_walk_length = next_len
                            no_length_increase = 0
                            if args.rank == 0:
                                print(
                                    f"Stage1: walk_length -> {args.head_hop_walk_length} "
                                    f"(far/near {prev_far_near_ratio:.4f} -> {ratio:.4f})"
                                )
                        else:
                            no_length_increase += 1
                            if args.rank == 0:
                                print("Stage1: max walk_length reached, hold.")
                    else:
                        no_length_increase += 1
                        if args.rank == 0:
                            print(
                                f"Stage1: keep walk_length={args.head_hop_walk_length} "
                                f"(far/near {ratio:.4f})"
                            )
                    prev_far_near_ratio = ratio

                if no_length_increase >= 3:
                    stage = 2
                    no_length_increase = 0
                    if args.rank == 0:
                        print("Stage1 -> Stage2")

            if stage == 2:
                idx_probe = split_idx["valid"][: min(args.seq_len, split_idx["valid"].size(0))]
                with fixed_random_seed(args.seed):
                    edges_prev = _build_edges_for_walks(
                        idx_probe, args.head_hop_walk_length, args.head_hop_walks_per_node
                    )
                with fixed_random_seed(args.seed):
                    edges_next = _build_edges_for_walks(
                        idx_probe, args.head_hop_walk_length, args.head_hop_walks_per_node + 1
                    )
                new_ratio = edge_index_new_edge_ratio(edges_prev, edges_next, num_nodes=idx_probe.size(0))
                if args.rank == 0:
                    print(
                        f"Stage2: walks_per_node={args.head_hop_walks_per_node} "
                        f"new_edge_ratio={new_ratio:.4f}"
                    )
                if prev_new_ratio is not None and new_ratio < prev_new_ratio:
                    stage = 3
                    stable_walks = 0
                    if args.rank == 0:
                        print(
                            "Stage2 -> Stage3 (new_edge_ratio decreased; freeze walks_per_node)"
                        )
                else:
                    if max_walks_per_node is None or args.head_hop_walks_per_node + 1 <= max_walks_per_node:
                        args.head_hop_walks_per_node += 1
                        if args.rank == 0:
                            print(f"Stage2: walks_per_node -> {args.head_hop_walks_per_node}")
                    else:
                        stage = 3
                        if args.rank == 0:
                            print("Stage2 -> Stage3 (max walks_per_node reached)")
                prev_new_ratio = new_ratio

        if args.rank == 0 and epoch % 5 ==0:
            t4 = time.time()
            with fixed_random_seed(args.seed):
                train_acc = sparse_eval_gpu(args, model, feature, y, split_idx["train"], None, edge_index_global, device)
            with fixed_random_seed(args.seed):
                val_acc = sparse_eval_gpu(args, model, feature, y, split_idx["valid"], None, edge_index_global, device)
            with fixed_random_seed(args.seed):
                test_acc = sparse_eval_gpu(args, model, feature, y, split_idx["test"], None, edge_index_global, device)
            t5 = time.time()
            print("------------------------------------------------------------------------------------")
            print(f'Eval time {t5-t4}s')
            print("Epoch: {:03d}, Loss: {:4f}, Train acc: {:.2%}, Val acc: {:.2%}, Test acc: {:.2%}, Epoch Time: {:.3f}s".format(
                epoch, np.mean(loss_list), train_acc, val_acc, test_acc, np.mean(epoch_t_list)))
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

        if getattr(args, "head_hop_edges", False):
            stage, no_length_increase, stable_walks = _sync_stage_state()

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
