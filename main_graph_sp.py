import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from utils.lr import PolynomialDecayLR
import argparse
import os
import time
import random
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from gt_sp.evaluate import calc_acc
from gt_sp.initialize import (
    initialize_distributed,
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import sync_params_and_buffers
from gt_sp.utils import pad_x_bs, pad_2d_bs, build_head_hop_edges
from data.dataset import GraphormerDataset
from models.graphormer_dist_graph_level_mp_malnet import Graphormer
from models.gt_dist_graph_level_mp_malnet import GT
from utils.parser_graph_level import parser_add_main_args


def get_sp_rank_data(args, batch, device):
    """Split batch tensors for sequence-parallel ranks and pad to equal length."""
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    sub_split_seq_lens = batch.sub_split_seq_lens  # per-rank sequence lengths

    x_i_list = [t for t in torch.split(batch.x, sub_split_seq_lens, dim=1)]
    in_degree_list = [t for t in torch.split(batch.in_degree, sub_split_seq_lens, dim=1)]
    out_degree_list = [t for t in torch.split(batch.out_degree, sub_split_seq_lens, dim=1)]

    padlen = max(sub_split_seq_lens)
    sub_real_seq_len = padlen + args.num_global_node

    x_i_list_pad = [pad_x_bs(t, padlen) for t in x_i_list]
    in_degree_list_pad = [pad_2d_bs(t, padlen) for t in in_degree_list]
    out_degree_list_pad = [pad_2d_bs(t, padlen) for t in out_degree_list]

    global_token_indices = list(range(0, args.sequence_parallel_size * sub_real_seq_len, sub_real_seq_len))
    set_global_token_indices(global_token_indices)

    x_i = x_i_list_pad[seq_parallel_world_rank].to(device)
    y_i = batch.y.to(device)
    in_degree_i = in_degree_list_pad[seq_parallel_world_rank].to(device)
    out_degree_i = out_degree_list_pad[seq_parallel_world_rank].to(device)
    edge_index = batch.edge_index.to(device)
    if args.model == "gt":
        batch_node_num = batch.x.size(1) + args.num_global_node
        total_nodes = batch.x.size(1) * batch.x.size(0)
        src = edge_index[0]
        dst = edge_index[1]
        src_local = src % batch_node_num
        dst_local = dst % batch_node_num
        keep = (src_local != 0) & (dst_local != 0)
        if keep.any():
            src = src[keep]
            dst = dst[keep]
            src_graph = src // batch_node_num
            dst_graph = dst // batch_node_num
            src = src_graph * batch.x.size(1) + (src_local[keep] - 1)
            dst = dst_graph * batch.x.size(1) + (dst_local[keep] - 1)
            edge_index = torch.stack([src, dst], dim=0)
        else:
            edge_index = edge_index.new_empty((2, 0), dtype=edge_index.dtype)
        edge_index = edge_index.clamp_min(0).clamp_max(total_nodes - 1)
    return x_i, y_i, in_degree_i, out_degree_i, edge_index


def resolve_attn_type(args, iter_idx, switch_points):
    if args.attn_type == "hybrid":
        return "full" if iter_idx in switch_points else "sparse"
    if args.attn_type == "full":
        return "full"
    if args.attn_type == "flash":
        return "flash"
    return "sparse"


def run_step(args, model, device, batch, criterion, attn_type):
    x_i, y_i, in_degree_i, out_degree_i, edge_index = get_sp_rank_data(args, batch, device)
    if getattr(args, "head_hop_edges", False) and attn_type == "sparse":
        total_nodes = x_i.size(0) * x_i.size(1)
        edge_index = build_head_hop_edges(
            edge_index=edge_index.to("cpu"),
            num_nodes=total_nodes,
            num_heads=args.num_heads,
            num_groups=getattr(args, "head_groups", args.num_heads),
            device="cpu",
            walk_length=getattr(args, "head_hop_walk_length", 4),
            walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
        )
        edge_index = [e.to(device) for e in edge_index]
    pred = model(x_i, in_degree_i, out_degree_i, edge_index, attn_type=attn_type)
    if args.dataset in ["ZINC"]:
        pred_loss = pred.view(-1)
        y_true = y_i.view(-1)
    else:
        pred_loss = pred
        y_true = y_i.view(-1)
    loss = criterion(pred_loss, y_true)
    return loss, pred, y_i


def train(args, model, device, packed_data, optimizer, criterion, epoch, lr_scheduler):
    model.train()
    model.to(device)

    loss_list, iter_t_list = [], []
    y_pred_list = []
    y_true_list = []

    if args.attn_type == "hybrid":
        percent_list = [(i + 1) / args.switch_freq for i in range(args.switch_freq)]
        switch_points = [int(len(packed_data) * percentage) for percentage in percent_list]
    else:
        switch_points = set()
    iter_idx = 1
    for batch in packed_data:
        attn_type = resolve_attn_type(args, iter_idx, switch_points)

        t0 = time.time()
        loss, pred, y_i = run_step(args, model, device, batch, criterion, attn_type)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.div_(get_sequence_parallel_world_size())
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())

        optimizer.step()

        iter_t_list.append(time.time() - t0)
        loss_list.append(loss.item())
        lr_scheduler.step()
        iter_idx += 1

        if args.dataset in ["MalNetTiny", "MalNet", "CIFAR10"]:
            y_pred_list.append(pred.argmax(1))
            y_true_list.append(y_i.view(-1))

    if args.dataset in ["MalNetTiny", "MalNet", "CIFAR10"]:
        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        eval_metric = calc_acc(y_true, y_pred)
        if args.rank == 0:
            print(f"train acc: {eval_metric}")

    return np.mean(loss_list)


@torch.no_grad()
def eval_gpu(args, model, device, packed_data, criterion, evaluator, metric, split_name):
    model.eval()
    model.to(device)

    y_pred_list = []
    y_true_list = []
    loss_list = []
    for batch in packed_data:
        attn_type = resolve_attn_type(args, 1, set())
        loss, pred, y_i = run_step(args, model, device, batch, criterion, attn_type)
        loss_list.append(loss.item())

        y_pred_list.append(pred.detach().cpu())
        y_true_list.append(y_i.detach().cpu())

    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    eval_metric = None
    if metric is not None:
        metric_lower = str(metric).lower()
        if metric_lower in ("mae", "rmse"):
            y_true_flat = y_true.view(-1)
            y_pred_flat = y_pred.view(-1)
            if metric_lower == "mae":
                eval_metric = torch.mean(torch.abs(y_pred_flat - y_true_flat)).item()
            else:
                eval_metric = torch.sqrt(torch.mean((y_pred_flat - y_true_flat) ** 2)).item()
        elif metric_lower in ("rocauc", "ap") and evaluator is not None:
            y_pred_eval = y_pred
            if y_pred_eval.dim() == 1:
                y_pred_eval = y_pred_eval.view(-1, 1)
            if y_true.dim() == 1:
                y_true_eval = y_true.view(-1, 1)
            else:
                y_true_eval = y_true
            y_pred_eval = torch.sigmoid(y_pred_eval)
            eval_metric = evaluator.eval(
                {"y_true": y_true_eval.numpy(), "y_pred": y_pred_eval.numpy()}
            )[metric_lower]
        elif evaluator is not None:
            y_pred_eval = y_pred
            if y_pred_eval.dim() == 1:
                y_pred_eval = y_pred_eval.view(-1, 1)
            if y_true.dim() == 1:
                y_true_eval = y_true.view(-1, 1)
            else:
                y_true_eval = y_true
            eval_result = evaluator.eval(
                {"y_true": y_true_eval.numpy(), "y_pred": y_pred_eval.numpy()}
            )
            if metric_lower in eval_result:
                eval_metric = eval_result[metric_lower]
    elif args.dataset in ["MalNetTiny", "MalNet", "CIFAR10"]:
        eval_metric = calc_acc(y_true, y_pred.argmax(1))

    if args.rank == 0:
        print(f"{split_name} loss {np.mean(loss_list):.4f}, metric {eval_metric}")
    return eval_metric


def main():
    parser = argparse.ArgumentParser(description="Graph-level training with sequence-parallel.")
    parser_add_main_args(parser)
    args = parser.parse_args()
    initialize_distributed(args)

    device = f"cuda:{torch.cuda.current_device()}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    dataset = GraphormerDataset(
        dataset_name=args.dataset,
        dataset_dir=args.dataset_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        multi_hop_max_dist=args.multi_hop_max_dist,
        spatial_pos_max=args.spatial_pos_max,
        myargs=args,
    )

    train_loader = dataset.train_dataloader()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()
    if args.rank == 0:
        print(args)
        print("Dataset loaded.")

    num_class = dataset.dataset["num_class"]
    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=args.dataset,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,
            output_dim=num_class,
        ).to(device)
    elif args.model == "gt":
        model = GT(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.input_dropout_rate,
            ffn_dim=args.ffn_dim,
            dataset_name=args.dataset,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            attention_dropout_rate=args.attention_dropout_rate,
            output_dim=num_class,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    if args.rank == 0:
        print("Model params:", sum(p.numel() for p in model.parameters()))

    sync_params_and_buffers(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup=args.warmup_updates,
        tot=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0
    )
    criterion = dataset.dataset.get("loss_fn", nn.CrossEntropyLoss())

    metric_mode = dataset.dataset.get("metric_mode", "max")
    if str(metric_mode).lower() == "min":
        best_val = float("inf")
    else:
        best_val = float("-inf")
    best_test = None
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, criterion, epoch, lr_scheduler)
        if args.rank == 0:
            print(f"Epoch {epoch} train loss {train_loss:.4f}")

        val_metric = eval_gpu(
            args, model, device, val_loader, criterion, dataset.dataset.get("evaluator"), dataset.dataset.get("metric"), "valid"
        )
        test_metric = eval_gpu(
            args, model, device, test_loader, criterion, dataset.dataset.get("evaluator"), dataset.dataset.get("metric"), "test"
        )

        if val_metric is not None:
            if str(metric_mode).lower() == "min":
                improved = val_metric < best_val
            else:
                improved = val_metric > best_val
        else:
            improved = False

        if improved:
            best_val = val_metric
            best_test = test_metric if test_metric is not None else best_test
            if args.save_model and args.rank == 0:
                torch.save(model.state_dict(), args.model_dir + f"{args.dataset}.pkl")

    if args.rank == 0:
        print(f"Best val metric: {best_val}, best test metric: {best_test}, mode: {metric_mode}")


if __name__ == "__main__":
    main()
