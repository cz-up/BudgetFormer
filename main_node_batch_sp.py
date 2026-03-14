import torch
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict
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
from gt_sp.adaptive_walk import estimate_L_star, coverage_for_R
from gt_sp.evaluate import sparse_eval_gpu
from gt_sp.utils import (
    get_batch_blockize,
    random_split_idx,
    fixed_random_seed,
)
from utils.parser_node_level import (
    add_node_batch_sp_args,
    add_node_common_args,
    normalize_main_node_batch_sp_args,
)


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

def run_step(args, model, device, feature, y, idx_batch_cpu, sub_split_seq_lens,
             edge_index_global, N, attn_type):
    current_split_seq_lens = sub_split_seq_lens
    if idx_batch_cpu.shape[0] < args.seq_len:
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
    parser = argparse.ArgumentParser(description="TorchGT node-level batch training (SP).")
    add_node_common_args(parser)
    add_node_batch_sp_args(parser)
    args = normalize_main_node_batch_sp_args(parser.parse_args())

    # Initialize distributed
    initialize_distributed(args)
    device = _resolve_device()
    _set_seed(args.seed)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    feature, y, edge_index_global, N, split_idx = _load_node_level_data(args, device)

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
    best_epoch = -1
    best_val = 0
    best_test = 0
    adaptive_enabled = getattr(args, "adaptive_walk", False)
    adaptive_stage = "stable"
    adaptive_no_improve = 0
    adaptive_cov_no_improve = 0
    if adaptive_enabled:
        adaptive_stage = "tune_L"

    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        if hasattr(model, "reset_head_mass_stats"):
            model.reset_head_mass_stats()
        idx_train_shuffle = flatten_train_idx[torch.randperm(flatten_train_idx.size(0))].to("cpu")
        batch_indices = torch.split(idx_train_shuffle, args.seq_len)

        iter_t_list = []
        runtime = defaultdict(float) if profile_single_machine else None
        iter_idx = 1
        needed_L_sum = 0.0
        needed_L_count = 0
        cov_diff_sum = 0.0
        cov_diff_count = 0
        do_diff_check = adaptive_enabled and adaptive_stage == "tune_L" and args.rank == 0
        do_cov_check = adaptive_enabled and adaptive_stage == "tune_R" and args.rank == 0
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
            if do_diff_check and iter_idx <= args.adaptive_embed_batches:
                needed_L = estimate_L_star(
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
                cov_R = coverage_for_R(
                    args,
                    idx_batch_cpu,
                    edge_index_global,
                    N,
                    seed=args.seed + epoch * 2000 + iter_idx,
                    walks_per_node=base_R,
                )
                cov_Rp1 = coverage_for_R(
                    args,
                    idx_batch_cpu,
                    edge_index_global,
                    N,
                    seed=args.seed + epoch * 2000 + iter_idx,
                    walks_per_node=base_R + 1,
                )
                cov_diff_sum += (cov_Rp1 - cov_R)
                cov_diff_count += 1
            iter_idx += 1

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

        if epoch % args.eval_every == 0:
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
                epoch_wall_so_far = time.time() - t_epoch_start
                print("------------------------------------------------------------------------------------")
                print(f'Eval time {t5-t4}s')
                print("Epoch: {:03d}, Loss: {:4f}, Train acc: {:.2%}, Val acc: {:.2%}, Test acc: {:.2%}, Epoch Wall Time: {:.3f}s".format(
                    epoch, np.mean(loss_list), train_acc, val_acc, test_acc, epoch_wall_so_far))
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
