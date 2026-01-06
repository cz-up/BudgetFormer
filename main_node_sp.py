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
)
from utils.parser_node_level import parser_add_main_args
import math


def resolve_attn_type(args, iter_idx, switch_points):
    if args.attn_type == "hybrid":
        return "full" if iter_idx in switch_points else "sparse"
    if args.attn_type == "full":
        return "full"
    if args.attn_type == "flash":
        return "flash"
    return "sparse"


def run_step(args, model, device, feature, y, idx_batch_cpu, sub_split_seq_lens, edge_index_global, N, attn_type):
    x_i, y_i, edge_index_i, attn_bias = get_batch_blockize(
        args,
        feature,
        y,
        idx_batch_cpu,
        sub_split_seq_lens,
        device,
        edge_index_global,
        N,
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

    def _prepare_group_nodes(group_nodes, model_name: str, device: str):
        prepared = []
        for nodes in group_nodes:
            nodes = nodes.to(device)
            if nodes.numel() > 0:
                nodes = torch.unique(nodes)
            if model_name == "graphormer":
                nodes = nodes + 1
                nodes = torch.cat([nodes.new_zeros(1), nodes])
            prepared.append(nodes)
        return prepared

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
    best_val = 0
    best_test = 0
    prev_loss = None
    prev_walk_length = getattr(args, "head_hop_walk_length", 4)
    prev_walks_per_node = getattr(args, "head_hop_walks_per_node", 2)
    max_walk_length = max(1, prev_walk_length * 2)
    max_walks_per_node = max(1, prev_walks_per_node * 2)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "reset_head_mass_stats"):
            model.reset_head_mass_stats()
        idx_train_shuffle = flatten_train_idx[torch.randperm(flatten_train_idx.size(0))].to("cpu")

        num_batch = idx_train_shuffle.size(0) // args.seq_len + 1
        batch_indices = torch.split(idx_train_shuffle, args.seq_len)
        if args.attn_type == "hybrid":
            percent_list = [(i + 1) / args.switch_freq for i in range(args.switch_freq)]
            switch_points = {int(len(batch_indices) * percentage) for percentage in percent_list}
        else:
            switch_points = set()

        iter_t_list = []
        t_epoch_start = time.time()
        iter_idx = 1
        for idx_batch_cpu in batch_indices:
            t_iter_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            attn_type = resolve_attn_type(args, iter_idx, switch_points)
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

        if args.rank == 0 and epoch % 5 ==0:
            t4 = time.time()
            train_acc = sparse_eval_gpu(args, model, feature, y, split_idx["train"], None, edge_index_global, device)
            val_acc = sparse_eval_gpu(args, model, feature, y, split_idx["valid"], None, edge_index_global, device)
            test_acc = sparse_eval_gpu(args, model, feature, y, split_idx["test"], None, edge_index_global, device)
            t5 = time.time()
            print("------------------------------------------------------------------------------------")
            print(f'Eval time {t5-t4}s')
            print("Epoch: {:03d}, Loss: {:4f}, Train acc: {:.2%}, Val acc: {:.2%}, Test acc: {:.2%}, Epoch Time: {:.3f}s".format(
                epoch, np.mean(loss_list), train_acc, val_acc, test_acc, np.mean(epoch_t_list)))
            print("------------------------------------------------------------------------------------")

            if val_acc > best_val:
                best_val = val_acc
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f"{args.dataset}.pkl")

            if test_acc > best_test:
                best_test = test_acc

            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)

        if seq_parallel_world_size > 1:
            dist.barrier()
        

    if args.rank == 0:
        print(f"Best validation accuracy: {best_val:.2%}, test accuracy: {best_test:.2%}")


if __name__ == "__main__":
    main()
