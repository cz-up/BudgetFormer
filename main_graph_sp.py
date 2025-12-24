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
from gt_sp.utils import pad_x_bs, pad_2d_bs
from data.dataset import GraphormerDataset
from models.graphormer_dist_graph_level_mp_malnet import Graphormer
from models.gt_dist_graph_level_mp_malnet import GT
from utils.parser_graph_level import parser_add_main_args


def get_sp_rank_data(args, batch, device):
    """Split batch tensors for sequence-parallel ranks and pad to equal length."""
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    sub_split_seq_lens = batch[5]  # per-rank sequence lengths

    x_i_list = [t for t in torch.split(batch[0], sub_split_seq_lens, dim=1)]
    in_degree_list = [t for t in torch.split(batch[2], sub_split_seq_lens, dim=1)]
    out_degree_list = [t for t in torch.split(batch[3], sub_split_seq_lens, dim=1)]

    padlen = max(sub_split_seq_lens)
    sub_real_seq_len = padlen + args.num_global_node

    x_i_list_pad = [pad_x_bs(t, padlen) for t in x_i_list]
    in_degree_list_pad = [pad_2d_bs(t, padlen) for t in in_degree_list]
    out_degree_list_pad = [pad_2d_bs(t, padlen) for t in out_degree_list]

    global_token_indices = list(range(0, args.sequence_parallel_size * sub_real_seq_len, sub_real_seq_len))
    set_global_token_indices(global_token_indices)

    x_i = x_i_list_pad[seq_parallel_world_rank].to(device)
    y_i = batch[1].to(device)
    in_degree_i = in_degree_list_pad[seq_parallel_world_rank].to(device)
    out_degree_i = out_degree_list_pad[seq_parallel_world_rank].to(device)
    edge_index = batch[4].to(device)
    return x_i, y_i, in_degree_i, out_degree_i, edge_index


def train(args, model, device, packed_data, optimizer, criterion, epoch, lr_scheduler):
    model.train()
    model.to(device)

    loss_list, iter_t_list = [], []
    y_pred_list = []
    y_true_list = []

    if args.attn_type == "hybrid":
        percent_list = [(i + 1) / args.switch_freq for i in range(args.switch_freq)]
        switch_points = [int(len(packed_data) * percentage) for percentage in percent_list]
    iter_idx = 1
    for batch in packed_data:
        x_i, y_i, in_degree_i, out_degree_i, edge_index = get_sp_rank_data(args, batch, device)

        if args.attn_type == "hybrid":
            attn_type = "full" if iter_idx in switch_points else "sparse"
        elif args.attn_type == "full":
            attn_type = "full"
        elif args.attn_type == "flash":
            attn_type = "flash"
        else:
            attn_type = "sparse"

        t0 = time.time()
        pred = model(x_i, in_degree_i, out_degree_i, edge_index, attn_type=attn_type)

        if args.dataset in ["ZINC"]:
            pred = pred.view(-1)
            y_true = y_i.view(-1)
        else:
            y_true = y_i.view(-1)

        loss = criterion(pred, y_true)

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
            y_true_list.append(y_true)

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
        x_i, y_i, in_degree_i, out_degree_i, edge_index = get_sp_rank_data(args, batch, device)
        if args.attn_type == "full":
            attn_type = "full"
        elif args.attn_type == "flash":
            attn_type = "flash"
        else:
            attn_type = "sparse"

        pred = model(x_i, in_degree_i, out_degree_i, edge_index, attn_type=attn_type)

        if args.dataset in ["ZINC"]:
            pred = pred.view(-1)
            y_true = y_i.view(-1)
        else:
            y_true = y_i.view(-1)

        loss = criterion(pred, y_true)
        loss_list.append(loss.item())

        if args.dataset in ["MalNetTiny", "MalNet", "CIFAR10"]:
            y_pred_list.append(pred.argmax(1))
            y_true_list.append(y_true)

    if args.dataset in ["MalNetTiny", "MalNet", "CIFAR10"]:
        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        eval_metric = calc_acc(y_true, y_pred)
    else:
        eval_metric = None

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
        name=args.dataset,
        root=args.dataset_dir,
        split="train",
        num_workers=0,
        seq_len=args.seq_len,
        seq_len_sp=args.seq_len // get_sequence_parallel_world_size(),
        sp_group_size=get_sequence_parallel_world_size(),
    )

    # Note: pre-packed batches for SP
    packed_data = dataset.packed_data
    if args.rank == 0:
        print(args)
        print("Dataset loaded.")

    if args.model == "graphormer":
        model = Graphormer(
            n_layers=args.n_layers,
            num_heads=args.num_heads,
            input_dim=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            output_dim=dataset.num_class,
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
            input_dim=args.hidden_dim,
            hidden_dim=args.hidden_dim,
            output_dim=dataset.num_class,
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
    criterion = nn.CrossEntropyLoss()

    best_val = 0
    best_test = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, packed_data["train"], optimizer, criterion, epoch, lr_scheduler)
        if args.rank == 0:
            print(f"Epoch {epoch} train loss {train_loss:.4f}")

        val_metric = eval_gpu(
            args, model, device, packed_data["valid"], criterion, dataset.evaluator, dataset.metric, "valid"
        )
        test_metric = eval_gpu(
            args, model, device, packed_data["test"], criterion, dataset.evaluator, dataset.metric, "test"
        )

        if val_metric is not None and val_metric > best_val:
            best_val = val_metric
            best_test = test_metric if test_metric is not None else best_test
            if args.save_model and args.rank == 0:
                torch.save(model.state_dict(), args.model_dir + f"{args.dataset}.pkl")

    if args.rank == 0:
        print(f"Best val metric: {best_val}, best test metric: {best_test}")


if __name__ == "__main__":
    main()
