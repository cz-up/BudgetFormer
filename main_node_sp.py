import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import json
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from utils.lr import PolynomialDecayLR
import argparse
import os
import time
import random
import torch.distributed as dist
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
    last_batch_flag,
)
from gt_sp.reducer import sync_params_and_buffers
from gt_sp.evaluate import sparse_eval_gpu
from gt_sp.utils import (
    get_batch_blockize,
    get_seed_batch_size,
    random_split_idx,
    fixed_random_seed,
)
from utils.parser_node_level import (
    add_node_batch_sp_args,
    add_node_common_args,
    normalize_main_node_sp_args,
)
from utils.split_utils import load_default_split
from torch_geometric.utils import coalesce


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


def _load_node_level_data(args, device):
    feature = torch.load(args.dataset_dir + args.dataset + "/x.pt")
    y = torch.load(args.dataset_dir + args.dataset + "/y.pt")
    edge_index_global = torch.load(args.dataset_dir + args.dataset + "/edge_index.pt")
    N = feature.shape[0]

    if args.dataset == "pokec":
        y = torch.clamp(y, min=0)

    # 始终尝试数据集默认划分，找不到时回退随机 60/20/20 分割
    split_idx = _load_default_split(args.dataset, args.dataset_dir, wait_for_rank0=True)
    if split_idx is None:
        if args.rank == 0:
            print("[split] No default split found, falling back to random 60/20/20 split.")
        split_idx = random_split_idx(y, frac_train=0.6, frac_valid=0.2, frac_test=0.2, seed=args.seed)
    else:
        if args.rank == 0:
            print("[split] Loaded official dataset split.")

    if args.rank == 0:
        print(args)
        print("Dataset load successfully")
        print(
            f"Train nodes: {split_idx['train'].shape[0]}, Val nodes: {split_idx['valid'].shape[0]}, Test nodes: {split_idx['test'].shape[0]}"
        )
        print(
            f"Training iters: {split_idx['train'].size(0) // args.seq_len + 1}, Val iters: {split_idx['valid'].size(0) // args.seq_len + 1}, Test iters: {split_idx['test'].size(0) // args.seq_len + 1}"
        )

    return feature, y, edge_index_global, N, split_idx


def _prepare_sequence_training_state(args, split_idx, device):
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    profile_single_machine = (not dist.is_initialized()) or (seq_parallel_world_size <= 1)
    profile_single_machine = False

    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()

    train_idx = split_idx["train"]

    if args.rank == 0:
        flatten_train_idx = train_idx.to(device)
    else:
        flatten_train_idx = torch.empty(train_idx.numel(), device=device, dtype=torch.int64)
    if seq_parallel_world_size > 1:
        dist.broadcast(flatten_train_idx, src_rank, group=group)

    if sequence_parallel_is_initialized():
        seq_len_per_rank = get_sequence_length_per_rank()
    else:
        seq_len_per_rank = args.seq_len
    sub_real_seq_len = seq_len_per_rank + args.num_global_node
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))

    if flatten_train_idx.shape[0] % args.seq_len != 0:
        last_batch_node_num = flatten_train_idx.shape[0] % args.seq_len
        if last_batch_node_num % seq_parallel_world_size != 0:
            div = last_batch_node_num // seq_parallel_world_size
            last_batch_node_num = div * seq_parallel_world_size + (seq_parallel_world_size - 1)

        x_dummy_list = [t for t in torch.tensor_split(torch.zeros(last_batch_node_num,), seq_parallel_world_size, dim=0)]
        sub_split_seq_lens = [t.shape[0] for t in x_dummy_list]
        sub_real_seq_len = max(sub_split_seq_lens) + args.num_global_node
        global_token_indices_last_batch = list(
            range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len)
        )
    else:
        sub_split_seq_lens = None
        global_token_indices_last_batch = None

    set_global_token_indices(global_token_indices)
    set_last_batch_global_token_indices(global_token_indices_last_batch)
    return flatten_train_idx, sub_split_seq_lens, seq_parallel_world_size, profile_single_machine


def _get_current_batch_split_state(args, batch_size: int):
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    if batch_size >= args.seq_len:
        return None, None

    x_dummy_list = [t for t in torch.tensor_split(torch.zeros(batch_size,), seq_parallel_world_size, dim=0)]
    rest_split_sizes = [t.shape[0] for t in x_dummy_list]
    sub_real_seq_len = max(rest_split_sizes) + args.num_global_node
    global_token_indices_last_batch = list(
        range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len)
    )
    return rest_split_sizes, global_token_indices_last_batch


def _build_node_model(args, feature, y, device):
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
    return model


def _print_runtime_profile(runtime, iter_t_list, epoch):
    profile_keys = ["zero_grad", "run_step", "backward", "grad_sync", "optim_step"]
    total_profile_t = sum(runtime[k] for k in profile_keys)
    total_profile_t = max(total_profile_t, 1e-12)
    iters = max(1, len(iter_t_list))
    print(f"[Node Profile][Epoch {epoch}] iters={iters}")
    for k in profile_keys:
        t = runtime[k]
        print(f"  - {k:10s}: {t:.3f}s ({100.0 * t / total_profile_t:.1f}%), {1000.0 * t / iters:.2f} ms/iter")


def _format_hop_stats_dirname(args) -> str:
    dirname = (
        f"attn{getattr(args, 'full_attn_hop_mass', 0.95):g}"
        f"_maxhop{int(getattr(args, 'full_attn_hop_max_hop', 64))}"
        f"_seq{int(getattr(args, 'seq_len', 0))}"
    )
    tag = str(getattr(args, "full_attn_hop_stats_tag", "")).strip()
    if tag:
        dirname = f"{dirname}_{tag}"
    return dirname


def _serialize_hop_mass_stats(stats):
    out = {}
    if stats is None:
        return out
    for i, stat in enumerate(stats):
        if stat is None:
            continue
        hop_sums, hop_count, max_sum, max_count, max_hop = stat[:5]
        hop_node_counts = stat[5] if len(stat) > 5 else []
        if hop_count > 0:
            hop_ratios = [float(h / hop_count) for h in hop_sums]
        else:
            hop_ratios = []
        graph_ratios_raw = hop_ratios[1:] if len(hop_ratios) > 1 else []
        graph_total = sum(graph_ratios_raw)
        graph_ratios = [r / graph_total for r in graph_ratios_raw] if graph_total > 0 else graph_ratios_raw
        # hop_relative: per-node attention relative to uniform baseline (same as NodeFormer)
        total_nc = sum(hop_node_counts)
        N_eff = total_nc / hop_count if hop_count > 0 else 1.0
        hop_relative = [
            float(hop_sums[h] / hop_node_counts[h] * N_eff) if h < len(hop_node_counts) and hop_node_counts[h] > 0 else 0.0
            for h in range(len(hop_sums))
        ]
        out[str(i)] = {
            "mean_max_hop": float(max_sum / max(1, max_count)),
            "max_hop": int(max_hop),
            "hop_count": int(hop_count),
            "N_eff": float(N_eff),
            "hop_ratios_start": 0,
            "hop_ratios": hop_ratios,
            "hop_node_counts": hop_node_counts,
            "graph_hop_ratios_start": 1,
            "graph_hop_ratios": graph_ratios,
            "hop_relative": hop_relative,
        }
    return out


def _write_hop_mass_stats(args, epoch: int, stats, train_acc: float, val_acc: float, test_acc: float) -> None:
    if not stats:
        return
    stats_root = getattr(args, "full_attn_hop_stats_dir", "./plot/hop_stats")
    model_dir = os.path.join(
        stats_root,
        str(args.model),
        str(args.dataset),
        _format_hop_stats_dirname(args),
    )
    os.makedirs(model_dir, exist_ok=True)
    payload = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "epoch": int(epoch),
        "mass": float(getattr(args, "full_attn_hop_mass", 0.95)),
        "max_queries": int(getattr(args, "full_attn_hop_max_queries", 64)),
        "query_sampling": str(getattr(args, "full_attn_hop_query_sampling", "random")),
        "max_batches": int(getattr(args, "full_attn_hop_max_batches", 1)),
        "max_hop_limit": int(getattr(args, "full_attn_hop_max_hop", 64)),
        "hop_ratios_start": 0,
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "layers": _serialize_hop_mass_stats(stats),
    }
    epoch_path = os.path.join(model_dir, f"epoch_{epoch:04d}.json")
    latest_path = os.path.join(model_dir, "latest.json")
    with open(epoch_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"[hop_stats] wrote {epoch_path}")


def _load_default_split(dataset_name: str, root_dir: str, wait_for_rank0: bool = False):
    return load_default_split(
        dataset_name,
        root_dir,
        dist_module=dist,
        wait_for_rank0=wait_for_rank0,
    )
def run_step(args, model, device, feature, y, idx_batch_cpu, sub_split_seq_lens,
             edge_index_global, N, attn_type):
    batch_mode = getattr(args, "batch_subgraph_mode", "induced")
    current_split_seq_lens = None if batch_mode == "seed_rw" else sub_split_seq_lens
    if batch_mode != "seed_rw" and idx_batch_cpu.shape[0] < args.seq_len:
        current_split_seq_lens, current_global_token_indices_last_batch = _get_current_batch_split_state(
            args, int(idx_batch_cpu.shape[0])
        )
        set_last_batch_global_token_indices(current_global_token_indices_last_batch)

    x_i, y_i, edge_index_i, attn_bias = get_batch_blockize(
        args,
        feature,
        y,
        idx_batch_cpu,
        current_split_seq_lens,
        edge_index_global,
        N,
        device=device,
        seed_label_mask=None,
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


@torch.no_grad()
def _collect_hop_mass_stats_main_node_sp(args, model, device, feature, y, sub_idx, edge_index_global, N):
    """
    Restore the original hop-stats semantics for main_node_sp.py:
    use the batch induced subgraph itself for evaluation-time hop tracking.
    Shuffle the validation/query nodes globally before batching so the sampled
    queries are not always drawn from the fixed prefix of the split when
    max_batches is small.
    """
    model.eval()
    if sub_idx.numel() > 1:
        sub_idx = sub_idx[torch.randperm(sub_idx.size(0))]
    num_batch = (sub_idx.size(0) + args.seq_len - 1) // args.seq_len

    for i in range(num_batch):
        idx_i = sub_idx[i * args.seq_len:(i + 1) * args.seq_len]
        rest_split_sizes = None
        if idx_i.shape[0] < args.seq_len:
            rest_split_sizes, current_global_token_indices_last_batch = _get_current_batch_split_state(
                args,
                int(idx_i.shape[0]),
            )
            set_last_batch_global_token_indices(current_global_token_indices_last_batch)

        x_i, y_i, edge_index_i, attn_bias = get_batch_blockize(
            args,
            feature,
            y,
            idx_i,
            rest_split_sizes,
            edge_index_global,
            N,
            device=device,
            seed_label_mask=None,
            force_induced_edges=True,
            apply_graphormer_virtual_edges=False,
        )

        x_i = x_i.to(device)
        y_i = y_i.to(device)
        if isinstance(edge_index_i, list):
            edge_index_i = [e.to(device) for e in edge_index_i]
        else:
            edge_index_i = edge_index_i.to(device)

        _ = model(x_i, attn_bias, edge_index_i, attn_type=args.attn_type)

    return model.get_hop_mass_stats_per_layer()


def main():
    parser = argparse.ArgumentParser(description="TorchGT node-level batch training (SP).")
    add_node_common_args(parser)
    add_node_batch_sp_args(parser)
    args = normalize_main_node_sp_args(parser.parse_args())

    # Initialize distributed
    initialize_distributed(args)
    device = _resolve_device()
    _set_seed(args.seed)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    feature, y, edge_index_global, N, split_idx = _load_node_level_data(args, device)
    if getattr(args, 'to_bidirected', False):
        edge_index_global = coalesce(
            torch.cat([edge_index_global, edge_index_global.flip(0)], dim=1),
            num_nodes=N,
        )
        if args.rank == 0:
            print(f"[to_bidirected] Converted to bidirected graph: {edge_index_global.shape[1]} edges")
    seed_batch_size = get_seed_batch_size(args)

    flatten_train_idx, sub_split_seq_lens, seq_parallel_world_size, profile_single_machine = (
        _prepare_sequence_training_state(args, split_idx, device)
    )
    # Auto-set attn_bias_dim to match the encoding dimensionality
    _mode = getattr(args, 'attn_bias_mode', 'none')
    if _mode == 'local_spd':
        args.attn_bias_dim = getattr(args, 'attn_bias_max_dist', 5) + 1
        if args.rank == 0:
            print(f'[local_spd] Setting attn_bias_dim = {args.attn_bias_dim} (max_dist+1)')
    model = _build_node_model(args, feature, y, device)

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
    epoch_wall_t_list = []
    # Inference time: val and test timed independently (CUDA-synced), first
    # eval skipped as warm-up.
    val_time_list, test_time_list = [], []
    _eval_warmup_done = False
    best_epoch = -1
    best_val = 0
    best_test = 0

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        if hasattr(model, "reset_head_mass_stats"):
            model.reset_head_mass_stats()
        idx_train_shuffle = flatten_train_idx[torch.randperm(flatten_train_idx.size(0))].to("cpu")
        batch_indices = torch.split(idx_train_shuffle, seed_batch_size)

        iter_t_list = []
        runtime = defaultdict(float) if profile_single_machine else None
        for idx_batch_cpu in batch_indices:
            t_iter_start = time.time()
            t_zero = time.time() if profile_single_machine else None
            optimizer.zero_grad(set_to_none=True)
            if profile_single_machine:
                runtime["zero_grad"] += time.time() - t_zero
            attn_type = args.attn_type
            t_run = time.time() if profile_single_machine else None
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
            if profile_single_machine:
                runtime["run_step"] += time.time() - t_run
            t_backward = time.time() if profile_single_machine else None
            loss.backward()
            if profile_single_machine:
                runtime["backward"] += time.time() - t_backward

            if dist.is_initialized() and sequence_parallel_is_initialized() and seq_parallel_world_size > 1:
                t_sync = time.time() if profile_single_machine else None
                for _, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.div_(get_sequence_parallel_world_size())
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=get_sequence_parallel_group())
                if profile_single_machine:
                    runtime["grad_sync"] += time.time() - t_sync

            t_step = time.time() if profile_single_machine else None
            optimizer.step()
            if profile_single_machine:
                runtime["optim_step"] += time.time() - t_step
            iter_t_list.append(time.time() - t_iter_start)

        loss_list.append(loss.item())
        lr_scheduler.step()
        if args.rank == 0:
            train_loop_time = np.sum(iter_t_list)
            print(
                f"Epoch: {epoch:03d}, Loss: {np.mean(loss_list):.4f}, Train Loop Time: {train_loop_time:.3f}s"
            )
            if profile_single_machine:
                _print_runtime_profile(runtime, iter_t_list, epoch)
        val_acc = None

        if epoch % args.eval_every == 0:
            stats = None
            # train acc: reported only, NOT counted in inference-time stats
            with fixed_random_seed(args.seed):
                train_acc = sparse_eval_gpu(args, model, feature, y, split_idx["train"], None, edge_index_global, device)
            # val inference time
            t_val = None
            if val_acc is None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                _t_val_start = time.perf_counter()
                with fixed_random_seed(args.seed):
                    val_acc = sparse_eval_gpu(args, model, feature, y, split_idx["valid"], None, edge_index_global, device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                t_val = time.perf_counter() - _t_val_start
            stats_needed = (
                args.full_attn_hop_stats
                and args.attn_type == "full"
                and hasattr(model, "enable_hop_mass_tracking")
            )
            if stats_needed:
                model.reset_hop_mass_stats()
                model.enable_hop_mass_tracking(
                    mass=args.full_attn_hop_mass,
                    max_queries=args.full_attn_hop_max_queries,
                    query_sampling=args.full_attn_hop_query_sampling,
                    max_batches=args.full_attn_hop_max_batches,
                    max_hop=args.full_attn_hop_max_hop,
                )
                with fixed_random_seed(args.seed + epoch):
                    stats = _collect_hop_mass_stats_main_node_sp(
                        args,
                        model,
                        device,
                        feature,
                        y,
                        split_idx["valid"],
                        edge_index_global,
                        N,
                    )
                model.disable_hop_mass_tracking()
            elif args.full_attn_hop_stats and args.rank == 0 and args.attn_type != "full":
                print(
                    f"[hop_stats] skipped because attn_type={args.attn_type}; "
                    "hop-mass stats are collected only in full attention mode."
                )

            # test inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            _t_test_start = time.perf_counter()
            with fixed_random_seed(args.seed):
                test_acc = sparse_eval_gpu(args, model, feature, y, split_idx["test"], None, edge_index_global, device)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_test = time.perf_counter() - _t_test_start
            if args.rank == 0:
                # Skip first eval as warm-up.
                if not _eval_warmup_done:
                    _eval_warmup_done = True
                else:
                    if t_val is not None:
                        val_time_list.append(t_val)
                    test_time_list.append(t_test)
                epoch_wall_so_far = time.time() - t_epoch_start
                print("------------------------------------------------------------------------------------")
                _val_str = f"val: {t_val*1000:.2f} ms" if t_val is not None else "val: (cached)"
                print(f"Inference time | {_val_str} | test: {t_test*1000:.2f} ms")
                print("Epoch: {:03d}, Loss: {:4f}, Train acc: {:.2%}, Val acc: {:.2%}, Test acc: {:.2%}, Epoch Wall Time: {:.3f}s".format(
                    epoch, np.mean(loss_list), train_acc, val_acc, test_acc, epoch_wall_so_far))
                if stats is not None:
                    print("Attention hop-mass stats (per layer):")
                    for i, stat in enumerate(stats):
                        if stat is None:
                            print(f"  layer {i}: no data")
                            continue
                        hop_sums, hop_count, max_sum, max_count, max_hop = stat[:5]
                        mean_max_hop = max_sum / max(1, max_count)
                        if hop_count > 0:
                            ratios = [h / hop_count for h in hop_sums]
                            ratio_str = ", ".join(f"{r:.3f}" for r in ratios)
                            print(
                                f"  layer {i}: mean_max_hop={mean_max_hop:.3f}, max_hop={max_hop}, "
                                f"hop_ratios(0-hop first)=[{ratio_str}]"
                            )
                        else:
                            print(f"  layer {i}: mean_max_hop={mean_max_hop:.3f}, max_hop={max_hop}, hop_ratios(0-hop first)=NA")
                    _write_hop_mass_stats(args, epoch, stats, train_acc, val_acc, test_acc)
                print("------------------------------------------------------------------------------------")

                if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = epoch
                    if args.save_model:
                        torch.save(model.state_dict(), args.model_dir + f"{args.dataset}.pkl")

                if test_acc > best_test:
                    best_test = test_acc

        if seq_parallel_world_size > 1:
            dist.barrier()
        if args.rank == 0:
            epoch_wall_time = time.time() - t_epoch_start
            epoch_wall_t_list.append(epoch_wall_time)
            print(
                f"[Epoch Wall] {epoch:03d}: {epoch_wall_time:.3f}s (avg {np.mean(epoch_wall_t_list):.3f}s)"
            )

    if args.rank == 0:
        print(f"Best epoch: {best_epoch}, validation accuracy: {best_val:.2%}, test accuracy: {best_test:.2%}")
        val_size = int(split_idx["valid"].numel())
        test_size = int(split_idx["test"].numel())
        if len(val_time_list) > 0:
            mean_val_s = float(np.mean(val_time_list))
            val_thr = val_size / mean_val_s if mean_val_s > 0 else 0.0
            print(f"Avg val  inference time (excluding warm-up): {mean_val_s*1000:.2f} ms  "
                  f"({val_size} nodes, throughput={val_thr:,.0f} nodes/s)")
        else:
            print("Avg val  inference time: n/a")
        if len(test_time_list) > 0:
            mean_test_s = float(np.mean(test_time_list))
            test_thr = test_size / mean_test_s if mean_test_s > 0 else 0.0
            print(f"Avg test inference time (excluding warm-up): {mean_test_s*1000:.2f} ms  "
                  f"({test_size} nodes, throughput={test_thr:,.0f} nodes/s)")
        else:
            print("Avg test inference time: n/a")
        if len(val_time_list) > 0 and len(test_time_list) > 0:
            mean_sum_s = float(np.mean(val_time_list)) + float(np.mean(test_time_list))
            combined_thr = (val_size + test_size) / mean_sum_s if mean_sum_s > 0 else 0.0
            print(f"Avg val+test inference time:                 {mean_sum_s*1000:.2f} ms  "
                  f"({val_size + test_size} nodes, throughput={combined_thr:,.0f} nodes/s)")
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
