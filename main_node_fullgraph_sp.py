"""Full-Graph Node-Level Training with Sequence Parallel (SP).

Architecture:
  Each SP rank holds a slice of the N graph nodes (indices [rank_start, rank_end)).
  Inside DistributedAttentionNodeLevel, _SeqAllToAll scatter-heads/gather-seq
  transforms each rank's [b, N/P, H, hn] into [b, N, H/P, hn] so CoreAttention
  sees ALL N nodes with H/P heads per head partition. After the second all-to-all
  the result returns to [b, N/P, H, hn]. This means sparse attention across the
  full edge_index is correct without any additional change to the attention code.
"""

import argparse
import contextlib
import copy
import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
import torch.nn.functional as F

from gt_sp.comm_profiler import (
    enable_comm_profiler,
    get_comm_profile_summary,
    reset_comm_profiler,
)
from gt_sp.initialize import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_parallel_world_size,
    initialize_distributed,
    sequence_parallel_is_initialized,
    set_global_token_indices,
    set_last_batch_global_token_indices,
)
from gt_sp.reducer import build_gradient_reducer, sync_params_and_buffers
from gt_sp.utils import clear_random_walk_graph_cache, resolve_edge_build_device
from utils.fullgraph_sp_support import (
    EDGE_POLICY_CPU_BROADCAST,
    EDGE_POLICY_CPU_BROADCAST_PREFETCH,
    EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
    EDGE_POLICY_GPU_EPHEMERAL,
    EDGE_POLICY_GPU_PERSIST,
    _AdaptiveEdgeBudgetController,
    _adaptive_edge_budget_enabled,
    _autocast_context,
    _build_attention_edges,
    _build_dst_csr,
    clear_merged_edge_cache,
    _build_model,
    _build_optimizer_bundle,
    _bootstrap_initial_edge_budget,
    _build_nagphormer_rw_csr,
    _build_rw_hop_features,
    _compute_laplacian_pe,
    _compute_multihop_features,
    _generate_expander_edges,
    _cuda_empty_cache,
    _eval_sp,
    _measure_edge_h2d_time,
    _try_cache_edge_index_on_gpu,
    _fixed_edge_budget_state_from_args,
    _get_process_peak_rss_mib,
    _get_process_rss_mib,
    _load_data,
    _maybe_update_edge_budget,
    _random_block_sampling_enabled,
    _resolve_adaptive_edge_budget_config,
    _resolve_amp_dtype,
    _resolve_device,
    _resolve_real_edge_sampling_device,
    _select_probe_nodes,
    _set_seed,
    _use_real_edges,
    _use_rw_edges,
)
from utils.parser_node_level import (
    add_node_common_args,
    add_node_fullgraph_sp_args,
    normalize_main_node_fullgraph_sp_args,
)


def _format_comm_profile(summary, train_time_s: float):
    if not summary:
        return []
    denom_ms = max(float(train_time_s) * 1000.0, 1e-6)
    ordered = (
        "seq_all_to_all_fwd",
        "seq_all_to_all_bwd",
        "grad_all_reduce",
        "edge_broadcast",
        "edge_full_h2d",
        "edge_build_local",
        "edge_real_sample",
        "edge_rw_build",
        "edge_rw_sample",
        "edge_merge",
    )
    lines = []
    for name in ordered:
        stat = summary.get(name)
        if stat is None or stat.get("kind", "timing") != "timing":
            continue
        total_ms = float(stat["total_ms"])
        payload_mib = float(stat["total_bytes"]) / (1024.0 ** 2)
        lines.append(
            f"  - {name:18s}: total={total_ms:8.2f} ms  "
            f"avg={stat['avg_ms']:7.2f} ms  "
            f"count={stat['count']:3d}  "
            f"share={100.0 * total_ms / denom_ms:5.1f}%  "
            f"payload={payload_mib:8.2f} MiB"
        )
    return lines


def _format_edge_cardinality(summary):
    real_edges = summary.get("edge_real_edges")
    rw_edges = summary.get("edge_rw_edges")
    merged_edges = summary.get("edge_merged_edges")
    if real_edges is None and rw_edges is None and merged_edges is None:
        return []
    return [
        "  - edge_cardinality  : "
        f"real={int(real_edges['value']) if real_edges is not None else 0:,}  "
        f"rw={int(rw_edges['value']) if rw_edges is not None else 0:,}  "
        f"merged={int(merged_edges['value']) if merged_edges is not None else 0:,}"
    ]


def _aggregate_comm_profile(summary):
    if not dist.is_initialized():
        return summary

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, summary)

    merged = {}
    for rank_summary in gathered:
        if not rank_summary:
            continue
        for name, stat in rank_summary.items():
            kind = stat.get("kind", "timing")
            cur = merged.get(name)
            if kind == "scalar":
                value = stat["value"]
                reduce = stat.get("reduce", "last")
                if cur is None:
                    merged[name] = {"kind": "scalar", "reduce": reduce, "value": value}
                elif reduce == "sum":
                    cur["value"] += value
                elif reduce == "max":
                    cur["value"] = value if value > cur["value"] else cur["value"]
                elif reduce == "last":
                    cur["value"] = value
                else:
                    raise ValueError(f"Unsupported profiler scalar reduction: {reduce}")
                continue

            total_ms = float(stat["total_ms"])
            total_bytes = int(stat["total_bytes"])
            if cur is None:
                merged[name] = {
                    "kind": "timing",
                    "count": int(stat["count"]),
                    "total_ms": total_ms,
                    "avg_ms": float(stat["avg_ms"]),
                    "total_bytes": total_bytes,
                }
                continue
            if total_ms > cur["total_ms"]:
                cur["count"] = int(stat["count"])
                cur["total_ms"] = total_ms
                cur["avg_ms"] = float(stat["avg_ms"])
            if total_bytes > cur["total_bytes"]:
                cur["total_bytes"] = total_bytes
    return merged


def _should_sync_step_timers(profile_sp_comm: bool, device: str) -> bool:
    return bool(profile_sp_comm) and torch.cuda.is_available() and torch.device(device).type == "cuda"


def _should_force_ckpt_sync_timers(model) -> bool:
    mgr = getattr(model, "_comm_ckpt", None)
    if mgr is None:
        return False
    fn = getattr(mgr, "needs_precise_timing", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _time_step_block(fn, *, device: str, synchronize_cuda: bool = False):
    if synchronize_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    result = fn()
    if synchronize_cuda:
        torch.cuda.synchronize(device)
    return result, time.perf_counter() - t0


def _apply_multi_tier_gpu_memory_limit(args, device: str) -> int:
    limit_mib = int(getattr(args, "multi_tier_gpu_memory_limit_mib", 0) or 0)
    args._multi_tier_effective_gpu_memory_limit_mib = 0.0
    if (
        limit_mib <= 0
        or getattr(args, "activation_checkpoint_mode", None) != "multi_tier"
        or not torch.cuda.is_available()
        or torch.device(device).type != "cuda"
    ):
        return 0

    dev = torch.device(device)
    real_total = int(torch.cuda.get_device_properties(dev).total_memory)
    requested = int(limit_mib * (1024 ** 2))
    effective = max(1, min(requested, real_total))
    fraction = min(max(effective / float(real_total), 0.0), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction, dev)
    args._multi_tier_effective_gpu_memory_limit_mib = effective / (1024 ** 2)
    if args.rank == 0:
        print(
            f"[gpu-cap] multi_tier per-rank GPU memory cap: "
            f"requested={limit_mib} MiB effective={effective / (1024 ** 2):.0f} MiB "
            f"(physical={real_total / (1024 ** 2):.0f} MiB, allocator_fraction={fraction:.4f})"
        )
    return effective


def _sp_group_max(value, *, device: str, group, dtype, initialized: bool) -> float | int:
    if not initialized:
        return value
    t = torch.tensor([value], device=device, dtype=dtype)
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    out = t.item()
    return float(out) if dtype.is_floating_point else int(out)


def _sp_group_min_int(value: int, *, device: str, group, initialized: bool) -> int:
    if not initialized:
        return int(value)
    t = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group)
    return int(t.item())


def _sp_group_barrier(*, group, initialized: bool) -> None:
    if not initialized:
        return
    dist.barrier(group=group)


def _edge_seed_for_epoch(epoch: int, *, args, use_epoch_seed: bool, adaptive_edge_budget_cfg):
    if not use_epoch_seed:
        return None
    if epoch <= adaptive_edge_budget_cfg.static_seed_epochs:
        return args.seed
    return args.seed + epoch


def _profile_multi_tier_edge_policy(
    *,
    args,
    policy: str,
    edge_index_global,
    num_nodes: int,
    device: str,
    rw_device,
    sp_group,
    sp_src_rank: int,
    sp_rank: int,
    local_num_nodes: int,
    edge_seed,
    edge_budget_state,
    adaptive_edge_budget_cfg,
    precise_step_timing: bool,
    sp_initialized: bool,
):
    if policy in (EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH, EDGE_POLICY_CPU_BROADCAST_PREFETCH):
        clear_merged_edge_cache()
        cpu_before = _get_process_rss_mib()
        gpu_peak_abs = 0
        build_time = 0.0
        stage_time = 0.0
        try:
            def _build_prefetch_payload():
                if policy == EDGE_POLICY_CPU_BROADCAST_PREFETCH and sp_rank != sp_src_rank:
                    return torch.zeros((2, 0), dtype=torch.long)
                return _build_attention_edges(
                    args,
                    edge_index_global,
                    num_nodes,
                    "cpu",
                    "cpu",
                    sp_group,
                    sp_src_rank,
                    sp_rank,
                    local_num_nodes,
                    edge_seed=edge_seed,
                    edge_budget_state=edge_budget_state,
                    adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                    edge_policy=policy,
                    force_broadcast=False,
                )

            build_probe, build_time = _time_step_block(
                _build_prefetch_payload,
                device="cpu",
                synchronize_cuda=False,
            )
            cpu_delta_bytes = max(
                int(((_get_process_rss_mib() - cpu_before) * (1024 ** 2))),
                0,
            )
            if torch.cuda.is_available() and device.startswith("cuda"):
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            if policy == EDGE_POLICY_CPU_BROADCAST_PREFETCH:
                # For broadcast-prefetch, only src_rank performs the CPU build.
                # Non-src ranks would otherwise enter the foreground broadcast
                # early and spend most of their measured "stage" time waiting
                # for src_rank to finish build_time. That wait is the
                # overlappable build portion and should not be double-counted in
                # stage_time.
                _sp_group_barrier(group=sp_group, initialized=sp_initialized)

            stage_probe, stage_time = _time_step_block(
                lambda: _build_attention_edges(
                    args,
                    edge_index_global,
                    num_nodes,
                    device,
                    rw_device,
                    sp_group,
                    sp_src_rank,
                    sp_rank,
                    local_num_nodes,
                    edge_seed=edge_seed,
                    edge_budget_state=edge_budget_state,
                    adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                    edge_policy=policy,
                    force_broadcast=(policy == EDGE_POLICY_CPU_BROADCAST_PREFETCH),
                ),
                device=device,
                synchronize_cuda=precise_step_timing,
            )
            if torch.cuda.is_available() and device.startswith("cuda"):
                gpu_peak_abs = int(torch.cuda.max_memory_allocated(device))
        finally:
            if "build_probe" in locals():
                del build_probe
            if "stage_probe" in locals():
                del stage_probe
            clear_merged_edge_cache()
            _cuda_empty_cache(args)

        build_time = _sp_group_max(
            float(build_time),
            device=device,
            group=sp_group,
            dtype=torch.float64,
            initialized=sp_initialized,
        )
        stage_time = _sp_group_max(
            float(stage_time),
            device=device,
            group=sp_group,
            dtype=torch.float64,
            initialized=sp_initialized,
        )
        gpu_peak_abs = _sp_group_max(
            int(gpu_peak_abs),
            device=device,
            group=sp_group,
            dtype=torch.long,
            initialized=sp_initialized,
        )
        cpu_delta_bytes = _sp_group_max(
            int(cpu_delta_bytes),
            device=device,
            group=sp_group,
            dtype=torch.long,
            initialized=sp_initialized,
        )
        return {
            "policy": policy,
            "prep_time_s": float(build_time + stage_time),
            "serial_time_s": float(stage_time),
            "overlap_time_s": float(build_time),
            "gpu_peak_bytes": int(gpu_peak_abs),
            "cpu_delta_bytes": int(cpu_delta_bytes),
            "live_edge_bytes": 0,
        }

    probe_edge_index_global = edge_index_global
    live_edge_bytes = 0
    probe_cached = False
    if policy == EDGE_POLICY_GPU_PERSIST:
        live_edge_bytes = int(edge_index_global.numel() * edge_index_global.element_size())
        if edge_index_global.device.type != "cuda":
            _cuda_empty_cache(args)
            probe_edge_index_global, can_cache_local = _try_cache_edge_index_on_gpu(
                edge_index_global,
                device,
                rank=args.rank,
                safety_factor=0.95,
            )
            can_cache = _sp_group_min_int(
                int(bool(can_cache_local)),
                device=device,
                group=sp_group,
                initialized=sp_initialized,
            )
            if can_cache == 0:
                if probe_edge_index_global is not None and can_cache_local:
                    del probe_edge_index_global
                _cuda_empty_cache(args)
                return None
            probe_cached = True
        else:
            probe_cached = True

    cpu_before = _get_process_rss_mib()
    gpu_peak_abs = 0
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    try:
        edge_index_probe, prep_time = _time_step_block(
            lambda: _build_attention_edges(
                args,
                probe_edge_index_global,
                num_nodes,
                device,
                rw_device,
                sp_group,
                sp_src_rank,
                sp_rank,
                local_num_nodes,
                edge_seed=edge_seed,
                edge_budget_state=edge_budget_state,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                edge_policy=policy,
            ),
            device=device,
            synchronize_cuda=precise_step_timing,
        )
        if torch.cuda.is_available() and device.startswith("cuda"):
            gpu_peak_abs = int(torch.cuda.max_memory_allocated(device))
        cpu_delta_bytes = max(int(((_get_process_rss_mib() - cpu_before) * (1024 ** 2))), 0)
    finally:
        if "edge_index_probe" in locals():
            del edge_index_probe
        if probe_cached and probe_edge_index_global is not edge_index_global:
            del probe_edge_index_global
        _cuda_empty_cache(args)

    prep_time = _sp_group_max(
        float(prep_time),
        device=device,
        group=sp_group,
        dtype=torch.float64,
        initialized=sp_initialized,
    )
    gpu_peak_abs = _sp_group_max(
        int(gpu_peak_abs),
        device=device,
        group=sp_group,
        dtype=torch.long,
        initialized=sp_initialized,
    )
    cpu_delta_bytes = _sp_group_max(
        int(cpu_delta_bytes),
        device=device,
        group=sp_group,
        dtype=torch.long,
        initialized=sp_initialized,
    )
    return {
        "policy": policy,
        "prep_time_s": float(prep_time),
        "serial_time_s": float(prep_time),
        "overlap_time_s": 0.0,
        "gpu_peak_bytes": int(gpu_peak_abs),
        "cpu_delta_bytes": int(cpu_delta_bytes),
        "live_edge_bytes": int(live_edge_bytes),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full-Graph Node-Level Training with Sequence Parallel."
    )
    add_node_common_args(
        parser,
        defaults={
            "peak_lr": 1e-3,
            "warmup_updates": 20,
            "sequence_parallel_size": 1,
            "num_global_node": 1,
        },
    )
    add_node_fullgraph_sp_args(parser)
    parser.add_argument(
        "--multi_tier_probe_cpu_broadcast",
        action="store_true",
        help=(
            "Deprecated compatibility flag. CPU sync broadcast is no longer an ACTIVE "
            "candidate for multi_tier; CPU edge policies are modelled via prefetchable "
            "rank-local / broadcast variants instead."
        ),
    )
    args = normalize_main_node_fullgraph_sp_args(parser.parse_args())
    fixed_edge_budget_state = _fixed_edge_budget_state_from_args(args)
    if _adaptive_edge_budget_enabled(args) and fixed_edge_budget_state is not None:
        raise ValueError("--fixed_real_edges_per_query/--fixed_rw_edges_per_query are not compatible with --adaptive_edge_budget.")

    initialize_distributed(args)

    sp_initialized = sequence_parallel_is_initialized()
    sp_world_size = get_sequence_parallel_world_size() if sp_initialized else 1
    sp_rank = get_sequence_parallel_rank() if sp_initialized else 0
    sp_src_rank = get_sequence_parallel_src_rank() if sp_initialized else 0
    sp_group = get_sequence_parallel_group() if sp_initialized else None

    device = _resolve_device()
    amp_dtype = _resolve_amp_dtype(args)
    rw_device = resolve_edge_build_device(args, device)
    _set_seed(args.seed)
    _apply_multi_tier_gpu_memory_limit(args, device)
    profile_sp_comm = bool(getattr(args, "profile_sp_comm", False))
    precise_step_timing = _should_sync_step_timers(profile_sp_comm, device)
    enable_comm_profiler(profile_sp_comm)

    if args.rank == 0:
        os.makedirs(args.model_dir, exist_ok=True)

    feature, y, edge_index_global, num_nodes, split_idx = _load_data(args)
    valid_idx = split_idx.get("valid")
    valid_size = int(valid_idx.numel()) if valid_idx is not None else 0
    adaptive_edge_budget_cfg = _resolve_adaptive_edge_budget_config(args, valid_size)

    # Build CSR indexed by dst once for fast probe-node edge filtering.
    # Replaces O(E) torch.isin scans in bootstrap and adaptive budget updates
    # with O(probe_size × avg_degree) lookups.
    edge_index_csr = None
    if _adaptive_edge_budget_enabled(args) and sp_rank == sp_src_rank:
        if args.rank == 0:
            print("[dst-csr] Building dst-indexed CSR for probe edge filtering...")
        _t_csr = time.time()
        edge_index_csr = _build_dst_csr(edge_index_global, num_nodes)
        if args.rank == 0:
            print(f"[dst-csr] Built in {time.time() - _t_csr:.2f}s  "
                  f"(E={edge_index_global.shape[1]:,}, N={num_nodes:,})")

    pad_num_nodes = ((num_nodes + sp_world_size - 1) // sp_world_size) * sp_world_size
    if pad_num_nodes > num_nodes:
        pad_rows = pad_num_nodes - num_nodes
        feature = torch.cat([feature, feature.new_zeros(pad_rows, feature.shape[1])], dim=0)
        y = torch.cat([y, y.new_full((pad_rows,) + y.shape[1:], -1)], dim=0)

    # NAGphormer: pre-compute K-hop neighborhood features once before training.
    # Done after padding so padded rows are zero-initialised and propagate cleanly.
    # feature becomes (N_padded, hops+1, d); all downstream slices are correct.
    _is_nagphormer = (args.model == "nagphormer")
    _is_exphormer = (args.model == "exphormer")
    _nag_rw_mode = _is_nagphormer and int(getattr(args, "nagphormer_rw_walks", 0)) > 0
    _nag_rw_ptr = None
    _nag_rw_col = None
    _nag_feature_cpu = None   # full (N_padded, d) CPU features kept for per-epoch RW
    if _is_nagphormer:
        _nag_pe_dim = int(getattr(args, "pe_dim", 0))
        _nag_hops = int(args.hops)
        if _nag_rw_mode:
            _nag_rw_walks = int(args.nagphormer_rw_walks)
            if args.rank == 0:
                print(
                    f"[nagphormer-rw] Building RW CSR "
                    f"(N={feature.shape[0]:,}, hops={_nag_hops}, walks={_nag_rw_walks}) …"
                )
            _nag_rw_ptr, _nag_rw_col, _row_aug_rw, _col_aug_rw = _build_nagphormer_rw_csr(
                edge_index_global, feature.shape[0]
            )
            # Apply Laplacian PE once (static), then keep the 2-D feature tensor on CPU
            _nag_feature_cpu = feature.float().cpu()
            if _nag_pe_dim > 0:
                if args.rank == 0:
                    print(f"[nagphormer-rw] Computing Laplacian PE (pe_dim={_nag_pe_dim}) …")
                lpe = _compute_laplacian_pe(_row_aug_rw, _col_aug_rw, feature.shape[0], _nag_pe_dim)
                _nag_feature_cpu = torch.cat([_nag_feature_cpu, lpe], dim=1)
            del _row_aug_rw, _col_aug_rw
            # feature stays 2-D so _build_model reads feature.shape[-1] = d+pe_dim correctly
            feature = _nag_feature_cpu
        else:
            feature = _compute_multihop_features(
                edge_index_global, feature.shape[0], feature, _nag_hops,
                pe_dim=_nag_pe_dim,
                rank=args.rank,
            )

    # Graphormer / GT NAGphormer-style preprocessing:
    #   hops > 0 : K-hop mean feature smoothing (bidirected+self-loop, same as NAGphormer)
    #              (N, hops+1, d+pe_dim) → mean over hop dim → (N, d+pe_dim)
    #              input_dim stays d+pe_dim; cross-node attention is unchanged.
    #   hops == 0, pe_dim > 0 : Laplacian PE only — concatenated to raw features → (N, d+pe_dim)
    _is_graphormer_preprocess = (
        args.model in ("graphormer", "gt")
        and (int(getattr(args, "hops", 0)) > 0 or int(getattr(args, "pe_dim", 0)) > 0)
    )
    if _is_graphormer_preprocess:
        _gmh_hops = int(getattr(args, "hops", 0))
        _gmh_pe_dim = int(getattr(args, "pe_dim", 0))
        if _gmh_hops > 0:
            feat_3d = _compute_multihop_features(
                edge_index_global, feature.shape[0], feature, _gmh_hops,
                pe_dim=_gmh_pe_dim,
                rank=args.rank,
            )
            # Mean over hop dim: (N, hops+1, d+pe_dim) → (N, d+pe_dim)
            # Graphormer's cross-node attention refines these pre-smoothed features;
            # keeping input_dim = d+pe_dim avoids inflating the first linear layer.
            feature = feat_3d.mean(dim=1).contiguous()
            del feat_3d
        else:
            # hops == 0: add Laplacian PE to raw features only
            if args.rank == 0:
                print(f"[graphormer-preprocess] Adding Laplacian PE (pe_dim={_gmh_pe_dim}) to raw features …")
            _row_bi = torch.cat([edge_index_global[0].cpu(), edge_index_global[1].cpu()])
            _col_bi = torch.cat([edge_index_global[1].cpu(), edge_index_global[0].cpu()])
            _self = torch.arange(feature.shape[0], dtype=torch.long)
            _row_aug = torch.cat([_row_bi, _self])
            _col_aug = torch.cat([_col_bi, _self])
            lpe = _compute_laplacian_pe(_row_aug, _col_aug, feature.shape[0], _gmh_pe_dim)
            feature = torch.cat([feature.float().cpu(), lpe], dim=1)

    # Exphormer: generate fixed expander edges once (same seed every run).
    # Stored on CPU; merged into edge_index per epoch as edge type = 1.
    _expander_edge_index_cpu = None
    _expander_degree = int(getattr(args, "expander_degree", 0))
    if _expander_degree > 0:
        if args.rank == 0:
            print(
                f"[exphormer] Generating expander edges "
                f"(N={pad_num_nodes:,}, degree={_expander_degree}) …"
            )
        _expander_edge_index_cpu = _generate_expander_edges(
            pad_num_nodes, _expander_degree, seed=int(getattr(args, "seed", 0))
        )
        if args.rank == 0:
            print(f"[exphormer] Expander edges: {_expander_edge_index_cpu.shape[1]:,}")

    nodes_per_rank = pad_num_nodes // sp_world_size
    rank_start = sp_rank * nodes_per_rank
    rank_end = rank_start + nodes_per_rank
    local_num_nodes = nodes_per_rank

    if args.model == "graphormer" and args.num_global_node > 0:
        sub_real_seq_len = nodes_per_rank + args.num_global_node
        global_token_indices = list(range(0, sp_world_size * sub_real_seq_len, sub_real_seq_len))
        set_global_token_indices(global_token_indices)
    else:
        set_global_token_indices([])
    set_last_batch_global_token_indices(None)

    train_idx_global = split_idx["train"]
    local_train_mask = (train_idx_global >= rank_start) & (train_idx_global < min(rank_end, num_nodes))
    local_train_idx = (train_idx_global[local_train_mask] - rank_start).to(device=device, dtype=torch.long)
    local_y = y[rank_start:rank_end].to(device)

    edge_budget_controller = _AdaptiveEdgeBudgetController(adaptive_edge_budget_cfg)
    if fixed_edge_budget_state is not None:
        edge_budget_controller.set_state(fixed_edge_budget_state)
        edge_budget_controller.frozen = True
    total_bootstrap_time = 0.0
    total_adjustment_time = 0.0
    probe_idx_global = None
    local_probe_idx = None
    if edge_budget_controller.enabled:
        probe_idx_global = _select_probe_nodes(
            split_idx,
            adaptive_edge_budget_cfg.probe_size,
            getattr(args, "seed", 0),
        )
        if probe_idx_global is not None:
            local_probe_mask = (probe_idx_global >= rank_start) & (probe_idx_global < min(rank_end, num_nodes))
            local_probe_idx = (probe_idx_global[local_probe_mask] - rank_start).to(
                device=device,
                dtype=torch.long,
            )

    model = _build_model(args, feature, y, device)
    args._runtime_edge_policy = EDGE_POLICY_GPU_EPHEMERAL
    args._runtime_force_edge_broadcast = False
    args._runtime_edge_policy_locked = False
    sync_params_and_buffers(model)
    grad_reducer = build_gradient_reducer(
        model,
        process_group=sp_group,
        world_size=sp_world_size,
    )

    if _nag_rw_mode:
        # In RW mode feature is 2-D (N, d+pe_dim); x_local is computed per epoch.
        # We use a dummy 3-D slice here so the model is initialised with the right
        # shape, but it will be overwritten at the start of every epoch.
        _local_nodes_cpu = torch.arange(rank_start, rank_end, dtype=torch.long)
        x_local = _build_rw_hop_features(
            _nag_feature_cpu, _nag_rw_ptr, _nag_rw_col,
            _local_nodes_cpu, _nag_hops, _nag_rw_walks,
            seed=int(getattr(args, "seed", 0)),
        ).float().to(device)
    else:
        x_local = feature[rank_start:rank_end].float().to(device)
    y_eval = y if args.rank == 0 else None
    split_idx_eval = split_idx if args.rank == 0 else None
    if not _nag_rw_mode:
        del feature
    if args.rank != 0:
        del y
        del split_idx
    if edge_budget_controller.enabled:
        t_bs_start = time.time()
        initial_budget_state = _bootstrap_initial_edge_budget(
            args,
            adaptive_edge_budget_cfg,
            edge_budget_controller,
            model,
            x_local,
            local_y,
            local_train_idx,
            probe_idx_global,
            local_probe_idx,
            edge_index_global,
            num_nodes,
            device,
            rw_device,
            sp_group,
            sp_src_rank,
            sp_rank,
            local_num_nodes,
            amp_dtype,
            sp_world_size,
            grad_reducer,
            edge_index_csr=edge_index_csr,
        )
        total_bootstrap_time += (time.time() - t_bs_start)
        edge_budget_controller.set_state(initial_budget_state)

    random_blocks_dynamic = (
        _random_block_sampling_enabled(args, adaptive_edge_budget_cfg)
        and (
            (
                _use_real_edges(args, adaptive_edge_budget_cfg)
                and max(adaptive_edge_budget_cfg.max_total_edges_per_query, edge_budget_controller.real_budget) > 0
            )
            or (
                int(getattr(args, "head_hop_walks_per_node", 0)) > 0
                and max(adaptive_edge_budget_cfg.max_total_edges_per_query, edge_budget_controller.rw_budget) > 0
            )
        )
    )
    dynamic_edges = _use_rw_edges(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg) or random_blocks_dynamic
    real_edge_sampling_device = _resolve_real_edge_sampling_device(device, rw_device)
    cpu_real_edge_sampling = real_edge_sampling_device.type == "cpu" and device.startswith("cuda")
    allow_cpu_rank_local_prefetch = bool(
        getattr(args, "allow_cpu_rank_local_prefetch", False)
    )

    def _set_runtime_edge_policy_for_phase(*, budget_phase_active: bool) -> None:
        # Once multi_tier finishes calibration and applies its synced ACTIVE
        # plan, preserve that runtime edge policy across later epochs.
        if bool(getattr(args, "_runtime_edge_policy_locked", False)):
            return
        if budget_phase_active:
            args._runtime_edge_policy = EDGE_POLICY_CPU_BROADCAST
            args._runtime_force_edge_broadcast = True
            return
        if cpu_real_edge_sampling:
            if allow_cpu_rank_local_prefetch:
                # Experimental path for measuring whether CPU-side prefetch can
                # hide edge construction. Real-edge sampling and RW stay on CPU,
                # but we preserve the deterministic rank-local path so the
                # prefetched CPU cache entry can be reused next epoch.
                args._runtime_edge_policy = EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH
                args._runtime_force_edge_broadcast = False
                return
            args._runtime_edge_policy = EDGE_POLICY_CPU_BROADCAST
            args._runtime_force_edge_broadcast = True
            return
        args._runtime_edge_policy = EDGE_POLICY_GPU_EPHEMERAL
        args._runtime_force_edge_broadcast = False

    _set_runtime_edge_policy_for_phase(
        budget_phase_active=_adaptive_edge_budget_enabled(args)
    )

    # ------------------------------------------------------------------
    # Edge-index GPU cache decision.
    #
    # adaptive checkpoint mode: the _CommAwareCheckpointer jointly decides
    #   whether caching the edge index is worthwhile vs giving that memory
    #   to keep_mha layers.  We measure t_h2d now (freeing the GPU copy
    #   immediately so peak_warmup stays clean), register the info with the
    #   checkpointer, and defer the actual cache to after calibration.
    #
    # all other modes: simple memory-check – cache if safety_factor × free
    #   GPU memory ≥ edge_bytes, else fall back to broadcast path.
    # ------------------------------------------------------------------
    _is_adaptive_ckpt = getattr(args, "activation_checkpoint_mode", None) == "adaptive"
    _is_multi_tier = getattr(args, "activation_checkpoint_mode", None) == "multi_tier"
    edge_index_gpu_cached = False
    _adaptive_edge_decision_done = True  # True = no deferred work needed

    _cuda_empty_cache(args)
    if _is_nagphormer:
        if args.rank == 0:
            print(
                "[edge-cache] Disabled: NAGphormer uses pre-computed multi-hop features; "
                "edge_index_global is not passed to the model."
            )
    elif cpu_real_edge_sampling:
        if args.rank == 0:
            print(
                "[edge-cache] Disabled: edge_build_device=cpu keeps real-edge sampling on CPU, "
                "so edge_index_global will not be cached on GPU."
            )
    elif _is_multi_tier:
        # ---- multi_tier: measure t_h2d, register with planner, defer --------
        # The _MultiTierResourceManager jointly decides topology cache vs
        # per-layer tier allocation after profiling.  We must not cache here:
        # an early cache would consume HBM before the planner can account for
        # it, making its budget estimate incorrect.
        _t_h2d = _measure_edge_h2d_time(edge_index_global, device)
        _edge_bytes_for_cache = edge_index_global.numel() * edge_index_global.element_size()
        _mt_ckpt_pre = getattr(model, "_comm_ckpt", None)
        if _mt_ckpt_pre is not None:
            _mt_ckpt_pre.set_edge_info(_edge_bytes_for_cache, _t_h2d)
        if args.rank == 0:
            print(
                f"[edge-cache] multi_tier mode: t_h2d={_t_h2d * 1000:.1f} ms, "
                f"edge={_edge_bytes_for_cache / (1024 ** 2):.1f} MiB. "
                "Edge-policy decision deferred to after planner calibration."
            )
    elif not _is_adaptive_ckpt:
        # ---- Non-adaptive: simple memory-check path ----------------------
        _ei_gpu, _ei_cached_local = _try_cache_edge_index_on_gpu(
                edge_index_global, device, rank=args.rank
            )
        if sp_world_size > 1 and dist.is_initialized():
            _can_cache_t = torch.tensor(
                [int(_ei_cached_local)], device=device, dtype=torch.long
            )
            dist.all_reduce(_can_cache_t, op=dist.ReduceOp.MIN, group=sp_group)
            if int(_can_cache_t.item()) == 0 and _ei_cached_local:
                del _ei_gpu
                _ei_gpu = None
                _ei_cached_local = False
        edge_index_gpu_cached = _ei_cached_local
        if edge_index_gpu_cached:
            edge_index_global = _ei_gpu
            clear_random_walk_graph_cache()
            if args.rank == 0:
                _ei_mib = edge_index_global.numel() * edge_index_global.element_size() / (1024 ** 2)
                _free_mib = torch.cuda.mem_get_info(device)[0] / (1024 ** 2)
                print(
                    f"[edge-cache] edge_index_global cached on GPU "
                    f"({_ei_mib:.1f} MiB; GPU free after: {_free_mib:.1f} MiB). "
                    "Rank-local edge-build active (no per-epoch H2D transfer)."
                )
        else:
            if _ei_gpu is not None:
                del _ei_gpu
            if not getattr(args, "force_edge_broadcast", False):
                args.force_edge_broadcast = True
                if args.rank == 0:
                    print(
                        "[edge-cache] edge_index_global not cached on GPU. "
                        "force_edge_broadcast enabled to avoid per-epoch H2D transfers."
                    )
    else:
        # ---- Adaptive (comm_aware): measure t_h2d, register, defer ----------
        _t_h2d = _measure_edge_h2d_time(edge_index_global, device)
        _edge_bytes_for_cache = edge_index_global.numel() * edge_index_global.element_size()
        _comm_ckpt = getattr(model, "_comm_ckpt", None)
        if _comm_ckpt is not None:
            _comm_ckpt.set_edge_info(_edge_bytes_for_cache, _t_h2d)
        if args.rank == 0:
            print(
                f"[edge-cache] Adaptive mode: t_h2d={_t_h2d * 1000:.1f} ms, "
                f"edge={_edge_bytes_for_cache / (1024 ** 2):.1f} MiB. "
                "Cache decision deferred to after calibration."
            )
        _adaptive_edge_decision_done = False  # applied inside the training loop

    if args.rank == 0:
        print(f"\n{'=' * 72}")
        print(f"Full-Graph SP Training  (sp_world_size={sp_world_size})")
        print(f"  Nodes per rank: {nodes_per_rank:,}  (rank {sp_rank}: {rank_start:,}–{rank_end:,})")
        print(f"  Edge_index: full graph E={edge_index_global.shape[1]:,}")
        print(
            f"  Sparse edges: real={int(_use_real_edges(args, adaptive_edge_budget_cfg))} "
            f"rw={int(_use_rw_edges(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg))}"
        )
        print(f"  Edge build device: {rw_device}")
        print(f"  Real-edge sampling device: {real_edge_sampling_device}")
        print(f"  Force edge broadcast: {int(bool(getattr(args, 'force_edge_broadcast', False)))}")
        print(f"  Runtime edge policy: {getattr(args, '_runtime_edge_policy', EDGE_POLICY_GPU_EPHEMERAL)}")
        print(f"  Edge index GPU cached: {int(edge_index_gpu_cached)}")
        print(f"  AMP dtype: {args.amp_dtype}")
        print(f"  Activation CPU offload: {int(bool(getattr(args, 'activation_cpu_offload', False)))}")
        if getattr(args, "_multi_tier_effective_gpu_memory_limit_mib", 0.0) > 0.0:
            print(
                f"  Multi-tier GPU cap: "
                f"{getattr(args, '_multi_tier_effective_gpu_memory_limit_mib', 0.0):.0f} MiB per rank"
            )
        print(
            f"  Random edge blocks: {int(_random_block_sampling_enabled(args, adaptive_edge_budget_cfg))} "
            f"(max_total={adaptive_edge_budget_cfg.max_total_edges_per_query})"
        )
        if fixed_edge_budget_state is not None:
            print(
                "  Fixed edge budget: "
                f"real={fixed_edge_budget_state['real_edges_per_query']} "
                f"rw={fixed_edge_budget_state['rw_edges_per_query']} "
                f"walk_length={fixed_edge_budget_state.get('walk_length', getattr(args, 'head_hop_walk_length', 4))}"
            )
        print(
            f"  Adaptive edge budget: {int(edge_budget_controller.enabled)} "
            f"(probe={adaptive_edge_budget_cfg.probe_size} "
            f"block={adaptive_edge_budget_cfg.block_size} "
            f"warmup={adaptive_edge_budget_cfg.warmup_epochs if adaptive_edge_budget_cfg.warmup_epochs is not None else 'none'} "
            f"patience={adaptive_edge_budget_cfg.patience} "
            f"bootstrap_search={adaptive_edge_budget_cfg.bootstrap_search_epochs} "
            f"static_seed={adaptive_edge_budget_cfg.static_seed_epochs})"
        )
        if edge_budget_controller.enabled:
            print(
                "  Budget-phase baseline: cpu_build_broadcast + all-recompute "
                "(until the edge budget stabilizes)"
            )
        print(f"{'=' * 72}\n")

    if args.rank == 0:
        print(f"Model params: {sum(param.numel() for param in model.parameters()):,}")

    optimizer, lr_scheduler, scaler = _build_optimizer_bundle(args, model, device, amp_dtype)

    best_val = 0.0
    best_test = 0.0
    best_epoch = -1
    loss_ema = None
    epoch_wall_time_sum = 0.0
    epoch_wall_time_count = 0
    edge_broadcast_ms_sum = 0.0
    edge_broadcast_bytes_sum = 0
    edge_broadcast_epoch_count = 0
    _multi_tier_edge_profiled = not _is_multi_tier
    # Track whether we have applied the post-ACTIVE topology decision.
    _multi_tier_decision_done = not _is_multi_tier  # True = no deferred work needed
    # For adaptive/multi_tier checkpoint + adaptive edge budget: track whether we
    # have already notified the checkpointer that the edge budget is frozen.
    # For all other cases (no deferred calibration), treat as already done.
    _budget_frozen_notified = not (
        (_is_adaptive_ckpt or _is_multi_tier) and _adaptive_edge_budget_enabled(args)
    )

    use_epoch_seed = dynamic_edges

    # Async edge prefetch: while the GPU runs epoch N, a background thread builds
    # and caches the merged edge_index for epoch N+1.  The main thread's
    # _build_edges_for_epoch call at epoch N+1 then hits _MERGED_EDGE_CACHE and
    # returns almost instantly.
    #
    # Prefetch is intentionally CPU-only.  The background thread always builds
    # CPU tensors, so cache reuse requires that the foreground path also uses
    # the CPU graph object identity and CPU random-walk build device.  Once the
    # graph is cached on GPU or random walks build on GPU, the prefetch result
    # no longer matches the foreground cache key and only adds hidden wait time.
    _prefetch_pool = ThreadPoolExecutor(max_workers=1)
    _prefetch_future = None

    def _prefetch_cache_compatible() -> bool:
        if edge_index_global.device.type != "cpu":
            return False
        edge_policy = str(
            getattr(args, "_runtime_edge_policy", EDGE_POLICY_GPU_EPHEMERAL)
        )
        if edge_policy in (
            EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
            EDGE_POLICY_CPU_BROADCAST_PREFETCH,
        ):
            return True
        return torch.device(rw_device).type == "cpu"

    def _prefetch_topology_plan_stable() -> bool:
        if _is_adaptive_ckpt and not _adaptive_edge_decision_done:
            return False
        if _is_multi_tier and not _multi_tier_decision_done:
            return False
        return True

    def _next_epoch_budget_known_at_epoch_start(epoch_idx: int) -> bool:
        if not edge_budget_controller.enabled:
            return True
        if edge_budget_controller.frozen:
            return True
        if epoch_idx <= int(adaptive_edge_budget_cfg.bootstrap_search_epochs):
            return True
        warmup_epochs = adaptive_edge_budget_cfg.warmup_epochs
        if warmup_epochs is not None and epoch_idx > int(warmup_epochs):
            return True
        return False

    def _submit_prefetch(next_epoch: int, budget_state) -> bool:
        nonlocal _prefetch_future
        if _prefetch_future is not None:
            return False
        if next_epoch > args.epochs:
            return False
        if (
            not use_epoch_seed
            or bool(getattr(args, "disable_edge_prefetch", False))
            or profile_sp_comm
            or not _prefetch_cache_compatible()
            or not _prefetch_topology_plan_stable()
        ):
            return False

        edge_policy_snap = str(
            getattr(args, "_runtime_edge_policy", EDGE_POLICY_GPU_EPHEMERAL)
        )
        force_broadcast_snap = bool(
            getattr(args, "_runtime_force_edge_broadcast", False)
        )
        is_cpu_broadcast_prefetch = edge_policy_snap == EDGE_POLICY_CPU_BROADCAST_PREFETCH
        if force_broadcast_snap and not is_cpu_broadcast_prefetch:
            return False
        if is_cpu_broadcast_prefetch and sp_rank != sp_src_rank:
            return False

        next_seed = _edge_seed_for_epoch(
            next_epoch,
            args=args,
            use_epoch_seed=use_epoch_seed,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        )
        if next_seed is None:
            return False

        # Keep the same CPU graph object identity so the next epoch's foreground
        # cache lookup can reuse the prefetched entry directly.
        edge_index_cpu = edge_index_global
        budget_snap = copy.copy(budget_state)

        def _prefetch_fn(
            _seed=next_seed,
            _ei=edge_index_cpu,
            _budget=budget_snap,
            _edge_policy=edge_policy_snap,
            _force_broadcast=(False if is_cpu_broadcast_prefetch else force_broadcast_snap),
        ):
            _build_attention_edges(
                args,
                _ei,
                num_nodes,
                "cpu",
                "cpu",
                sp_group,
                sp_src_rank,
                sp_rank,
                local_num_nodes,
                edge_seed=_seed,
                edge_budget_state=_budget,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                edge_policy=_edge_policy,
                force_broadcast=_force_broadcast,
            )

        _prefetch_future = _prefetch_pool.submit(_prefetch_fn)
        return True

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        prefetch_wait_time = 0.0
        if profile_sp_comm:
            reset_comm_profiler()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _set_runtime_edge_policy_for_phase(
            budget_phase_active=_adaptive_edge_budget_enabled(args) and not _budget_frozen_notified
        )
        epoch_edge_seed = _edge_seed_for_epoch(
            epoch,
            args=args,
            use_epoch_seed=use_epoch_seed,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        )
        epoch_budget_state = dict(edge_budget_controller.current_state())
        sync_step_timing = precise_step_timing or _should_force_ckpt_sync_timers(model)

        if _is_multi_tier and dynamic_edges and not _multi_tier_edge_profiled and _budget_frozen_notified:
            _mt_prof_mgr = getattr(model, "_comm_ckpt", None)
            if _mt_prof_mgr is not None:
                candidate_policies = []
                if device.startswith("cuda"):
                    candidate_policies.extend(
                        [
                            EDGE_POLICY_GPU_EPHEMERAL,
                            EDGE_POLICY_GPU_PERSIST,
                        ]
                    )
                if device.startswith("cuda") and edge_index_global.device.type == "cpu":
                    candidate_policies.extend(
                        [
                            EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
                            EDGE_POLICY_CPU_BROADCAST_PREFETCH,
                        ]
                    )
                if args.rank == 0:
                    print(
                        f"[multi_tier] Profiling edge policies at epoch {epoch} "
                        f"(budget={edge_budget_controller.current_state()})"
                    )
                    if device.startswith("cuda") and edge_index_global.device.type == "cpu":
                        print(
                            "[multi_tier] CPU prefetch policies enabled for ACTIVE planning "
                            "regardless of the run's default edge_build_device."
                        )
                    elif device.startswith("cuda"):
                        print(
                            "[multi_tier] CPU prefetch policies disabled because edge_index_global "
                            "is not CPU-resident for this run."
                        )
                for policy in (
                    EDGE_POLICY_GPU_PERSIST,
                    EDGE_POLICY_GPU_EPHEMERAL,
                    EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
                    EDGE_POLICY_CPU_BROADCAST_PREFETCH,
                ):
                    if policy not in candidate_policies:
                        _mt_prof_mgr.set_edge_policy_profile(
                            policy,
                            prep_time_s=0.0,
                            gpu_peak_bytes=0,
                            cpu_delta_bytes=0,
                            live_edge_bytes=0,
                            enabled=False,
                        )
                        continue
                    metrics = _profile_multi_tier_edge_policy(
                        args=args,
                        policy=policy,
                        edge_index_global=edge_index_global,
                        num_nodes=num_nodes,
                        device=device,
                        rw_device=rw_device,
                        sp_group=sp_group,
                        sp_src_rank=sp_src_rank,
                        sp_rank=sp_rank,
                        local_num_nodes=local_num_nodes,
                        edge_seed=epoch_edge_seed,
                        edge_budget_state=edge_budget_controller.current_state(),
                        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                        precise_step_timing=sync_step_timing,
                        sp_initialized=sp_world_size > 1 and dist.is_initialized(),
                    )
                    if metrics is None:
                        _mt_prof_mgr.set_edge_policy_profile(
                            policy,
                            prep_time_s=0.0,
                            gpu_peak_bytes=0,
                            cpu_delta_bytes=0,
                            live_edge_bytes=0,
                            enabled=False,
                        )
                        if args.rank == 0:
                            print(f"[multi_tier] edge_policy={policy} infeasible during probe; skipping.")
                        continue
                    _mt_prof_mgr.set_edge_policy_profile(
                        policy,
                        prep_time_s=metrics["prep_time_s"],
                        serial_time_s=metrics.get("serial_time_s"),
                        overlap_time_s=metrics.get("overlap_time_s", 0.0),
                        gpu_peak_bytes=metrics["gpu_peak_bytes"],
                        cpu_delta_bytes=metrics["cpu_delta_bytes"],
                        live_edge_bytes=metrics["live_edge_bytes"],
                        enabled=True,
                    )
                    if args.rank == 0:
                        print(
                            f"[multi_tier] edge_policy={policy}: "
                            f"prep={metrics['prep_time_s'] * 1000:.1f} ms "
                            f"serial={metrics.get('serial_time_s', metrics['prep_time_s']) * 1000:.1f} ms "
                            f"overlap={metrics.get('overlap_time_s', 0.0) * 1000:.1f} ms "
                            f"gpu_peak={metrics['gpu_peak_bytes'] / (1024 ** 2):.1f} MiB "
                            f"cpu_delta={metrics['cpu_delta_bytes'] / (1024 ** 2):.1f} MiB "
                            f"live_edge={metrics['live_edge_bytes'] / (1024 ** 2):.1f} MiB"
                        )
                _multi_tier_edge_profiled = True
                if profile_sp_comm:
                    reset_comm_profiler()

        def _build_edges_for_epoch():
            return _build_attention_edges(
                args,
                edge_index_global,
                num_nodes,
                device,
                rw_device,
                sp_group,
                sp_src_rank,
                sp_rank,
                local_num_nodes,
                edge_seed=epoch_edge_seed,
                edge_budget_state=edge_budget_controller.current_state(),
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            )
        # If a prefetch future exists, wait for it before calling _build_edges_for_epoch.
        # The prefetch populates _MERGED_EDGE_CACHE, so the build call below is a
        # near-instant cache hit.  If the prefetch failed or produced a stale result
        # (budget changed), _build_edges_for_epoch falls back to a normal build.
        if _prefetch_future is not None:
            _prefetch_wait_start = time.perf_counter()
            try:
                _prefetch_future.result()
            except Exception as exc:
                if args.rank == 0:
                    print(f"[edge-prefetch] background build failed; foreground build will be used: {exc}")
            prefetch_wait_time = time.perf_counter() - _prefetch_wait_start
            _prefetch_future = None

        if _is_nagphormer:
            # NAGphormer uses pre-computed multi-hop features; no edge building needed.
            edge_index_rw, rw_time = None, 0.0
            if _nag_rw_mode:
                # Re-sample walks every epoch; different seed gives stochastic regularisation.
                _rw_seed = int(getattr(args, "seed", 0)) + epoch
                x_local = _build_rw_hop_features(
                    _nag_feature_cpu, _nag_rw_ptr, _nag_rw_col,
                    _local_nodes_cpu, _nag_hops, _nag_rw_walks,
                    seed=_rw_seed,
                ).float().to(device)
        else:
            edge_index_rw, rw_time = _time_step_block(
                _build_edges_for_epoch,
                device=device,
                synchronize_cuda=sync_step_timing,
            )

        # Exphormer: merge expander edges into edge_index_rw and attach edge-type
        # labels in row-2 (0 = real/RW edge, 1 = expander edge).
        # ExphormerCoreAttention reads row-2 as edge type for score modulation.
        if _expander_edge_index_cpu is not None and edge_index_rw is not None:
            _exp = _expander_edge_index_cpu.to(edge_index_rw.device)
            # Strip existing row-2 (hop counts) if present — Exphormer uses row-2 for type
            _ei_base = edge_index_rw[:2] if edge_index_rw.size(0) == 3 else edge_index_rw
            _n_real = _ei_base.size(1)
            _n_exp = _exp.size(1)
            _real_type = torch.zeros(_n_real, dtype=torch.long, device=_ei_base.device)
            _exp_type = torch.ones(_n_exp, dtype=torch.long, device=_ei_base.device)
            edge_index_rw = torch.cat([
                torch.cat([_ei_base, _exp], dim=1),
                torch.cat([_real_type, _exp_type]).unsqueeze(0),
            ], dim=0)  # (3, E_real + E_exp)
        # Submit N+1 prefetch as soon as the current epoch's edge build is done.
        # This allows the CPU worker to overlap with the current epoch's GPU
        # forward/backward when the next epoch's budget and edge policy are
        # already known at epoch start (fixed budget, frozen adaptive budget,
        # or post-warmup stable phase).
        _prefetch_submitted = False
        if _next_epoch_budget_known_at_epoch_start(epoch):
            _prefetch_submitted = _submit_prefetch(
                epoch + 1,
                budget_state=epoch_budget_state,
            )

        def _forward_step():
            # multi_tier models per-layer offload explicitly inside Graphormer;
            # do not wrap the full forward in save_on_cpu, otherwise the
            # single-layer OFFLOAD calibration becomes impossible to interpret.
            _global_offload_enabled = (
                bool(getattr(args, "activation_cpu_offload", False))
                and not _is_multi_tier
            )
            offload_ctx = (
                torch.autograd.graph.save_on_cpu(pin_memory=True)
                if _global_offload_enabled
                else contextlib.nullcontext()
            )
            with _autocast_context(device, amp_dtype), offload_ctx:
                out_local = model(x_local, None, edge_index_rw, attn_type=args.attn_type)
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
                    loss = out_local.sum() * 0.0
            return out_local, out_rows, local_y_eff, valid_train_mask, local_train_idx_eff, loss

        def _run_fwd_bwd_opt():
            fwd_result, fwd_t = _time_step_block(
                _forward_step,
                device=device,
                synchronize_cuda=sync_step_timing,
            )
            (
                out_local_,
                out_rows_,
                local_y_eff_,
                valid_train_mask_,
                local_train_idx_eff_,
                loss_,
            ) = fwd_result

            def _autograd_backward_step():
                grad_reducer.prepare_backward()
                if scaler.is_enabled():
                    scaler.scale(loss_).backward()
                else:
                    loss_.backward()
            _, autograd_bwd_t = _time_step_block(
                _autograd_backward_step,
                device=device,
                synchronize_cuda=sync_step_timing,
            )

            _, grad_sync_t = _time_step_block(
                lambda: grad_reducer.finalize_backward(),
                device=device,
                synchronize_cuda=sync_step_timing,
            )
            bwd_t = autograd_bwd_t + grad_sync_t

            def _optimizer_step():
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
            _, opt_t = _time_step_block(
                _optimizer_step,
                device=device,
                synchronize_cuda=sync_step_timing,
            )
            return (
                out_local_,
                out_rows_,
                local_y_eff_,
                valid_train_mask_,
                local_train_idx_eff_,
                loss_,
                fwd_t,
                autograd_bwd_t,
                grad_sync_t,
                bwd_t,
                opt_t,
            )

        def _sp_any_oom(local_flag: bool) -> bool:
            if sp_world_size > 1 and dist.is_initialized():
                t = torch.tensor([int(local_flag)], device=device, dtype=torch.long)
                dist.all_reduce(t, op=dist.ReduceOp.MAX, group=sp_group)
                return bool(t.item())
            return bool(local_flag)

        def _multi_tier_manager():
            mgr = getattr(model, "_comm_ckpt", None)
            if mgr is None:
                return None
            if getattr(args, "activation_checkpoint_mode", None) != "multi_tier":
                return None
            return mgr

        local_oom = False
        _oom_exc = None
        try:
            (
                out_local,
                out_rows,
                local_y_eff,
                valid_train_mask,
                local_train_idx_eff,
                loss,
                fwd_time,
                autograd_bwd_time,
                grad_sync_time,
                bwd_time,
                opt_time,
            ) = _run_fwd_bwd_opt()
        except torch.cuda.OutOfMemoryError as _e:
            local_oom = True
            _oom_exc = _e
        # Every rank must vote on OOM status: if one rank OOM'd inside an NCCL
        # collective, others are blocked there and can't reach this point — they
        # will surface as a timeout rather than a local OOM flag.
        if _sp_any_oom(local_oom):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
            # Adaptive budget verified this graph fits with all-recompute, so
            # OOM during calibration is unexpected and unrecoverable.
            if local_oom:
                raise _oom_exc
            raise RuntimeError(
                "multi_tier: peer rank reported OOM; aborting."
            )

        # Notify comm-aware checkpointer that this step is complete.
        # Must happen after optimizer.step() so that optimizer state (lazily
        # allocated on the first step) is included in the peak measurement.
        if hasattr(model, "comm_aware_notify_step_end"):
            model.comm_aware_notify_step_end(device, t_bwd=bwd_time, t_fwd=fwd_time)

        # Adaptive mode: apply edge-cache decision once calibration completes.
        # The checkpointer transitions to ACTIVE after the calibrate step, so
        # this block runs exactly once (at the start of epoch 3 effective work).
        if _is_adaptive_ckpt and not _adaptive_edge_decision_done:
            _comm_ckpt_check = getattr(model, "_comm_ckpt", None)
            if _comm_ckpt_check is not None and _comm_ckpt_check.is_active():
                _adaptive_edge_decision_done = True
                _should_cache = int(_comm_ckpt_check.cache_edge)
                # Synchronize: require all SP ranks to agree on CACHE.
                # Using MIN so a single NO-CACHE vote overrides.
                if sp_world_size > 1 and dist.is_initialized():
                    _cache_agree = torch.tensor(
                        [_should_cache], device=device, dtype=torch.long
                    )
                    dist.all_reduce(_cache_agree, op=dist.ReduceOp.MIN, group=sp_group)
                    _should_cache = int(_cache_agree.item())
                # Update _modes on every rank to match the agreed decision.
                # Critical: without this, ranks that voted differently end up
                # with inconsistent keep_mha assignments, causing A2A deadlock
                # in the backward pass when checkpoint modes disagree.
                _comm_ckpt_check.override_cache_decision(bool(_should_cache))
                if _should_cache:
                    _cuda_empty_cache(args)
                    _ei_adp, _ei_adp_ok = _try_cache_edge_index_on_gpu(
                        edge_index_global, device, rank=args.rank
                    )
                    if _ei_adp_ok:
                        edge_index_global = _ei_adp
                        clear_merged_edge_cache()
                        clear_random_walk_graph_cache()
                        edge_index_gpu_cached = True
                        if args.rank == 0:
                            _mib = (
                                edge_index_global.numel()
                                * edge_index_global.element_size()
                                / (1024 ** 2)
                            )
                            print(
                                f"[edge-cache] Applied: edge_index_global cached on GPU "
                                f"({_mib:.1f} MiB). Rank-local build active."
                            )
                    else:
                        if not getattr(args, "force_edge_broadcast", False):
                            args.force_edge_broadcast = True
                        if args.rank == 0:
                            print(
                                "[edge-cache] OOM when applying cache decision; "
                                "falling back to broadcast path."
                            )
                else:
                    if not getattr(args, "force_edge_broadcast", False):
                        args.force_edge_broadcast = True
                    if args.rank == 0:
                        print(
                            "[edge-cache] Decision: no cache. "
                            "force_edge_broadcast enabled."
                        )
                _cuda_empty_cache(args)

        # multi_tier: apply topology-cache decision once ACTIVE.
        # Runs exactly once, after the planner finishes its last probe step.
        # Mirrors the adaptive-checkpoint sync pattern above but serves the
        # _MultiTierResourceManager instead of _CommAwareCheckpointer.
        if _is_multi_tier and not _multi_tier_decision_done:
            _mt_ckpt = getattr(model, "_comm_ckpt", None)
            if _mt_ckpt is not None and _mt_ckpt.is_active():
                _multi_tier_decision_done = True
                _policy_to_id = {
                    EDGE_POLICY_GPU_PERSIST: 0,
                    EDGE_POLICY_GPU_EPHEMERAL: 1,
                    EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH: 2,
                    EDGE_POLICY_CPU_BROADCAST_PREFETCH: 3,
                    EDGE_POLICY_CPU_BROADCAST: 4,
                }
                _id_to_policy = {v: k for k, v in _policy_to_id.items()}
                _mode_to_sync_id = {
                    _mt_ckpt.T4_RETAIN: 0,
                    _mt_ckpt.T1_KEEP_MHA: 1,
                    _mt_ckpt.T0_RECOMPUTE: 2,
                    _mt_ckpt.T2_OFFLOAD: 3,
                }
                _sync_id_to_mode = {v: k for k, v in _mode_to_sync_id.items()}
                _mt_policy_id = _policy_to_id.get(_mt_ckpt.edge_policy, 1)
                _local_mode_ids = torch.tensor(
                    [_mode_to_sync_id.get(_mt_ckpt.mode(i), 2) for i in range(_mt_ckpt.n_layers)],
                    device=device,
                    dtype=torch.long,
                )
                if sp_world_size > 1 and dist.is_initialized():
                    _mt_policy_min = torch.tensor(
                        [_mt_policy_id], device=device, dtype=torch.long
                    )
                    _mt_policy_max = torch.tensor(
                        [_mt_policy_id], device=device, dtype=torch.long
                    )
                    dist.all_reduce(_mt_policy_min, op=dist.ReduceOp.MIN, group=sp_group)
                    dist.all_reduce(_mt_policy_max, op=dist.ReduceOp.MAX, group=sp_group)
                    if int(_mt_policy_min.item()) != int(_mt_policy_max.item()) and args.rank == 0:
                        print(
                            "[multi_tier] WARNING: edge-policy disagreement across ranks; "
                            "falling back to the most conservative synced policy."
                        )
                    _mt_policy_id = int(_mt_policy_max.item())
                    _synced_mode_ids = _local_mode_ids.clone()
                    dist.all_reduce(_synced_mode_ids, op=dist.ReduceOp.MAX, group=sp_group)
                    if not torch.equal(_synced_mode_ids, _local_mode_ids) and args.rank == 0:
                        print(
                            "[multi_tier] WARNING: tier-plan disagreement across ranks; "
                            "forcing the most conservative synced plan."
                        )
                    _local_mode_ids = _synced_mode_ids
                _mt_policy = _id_to_policy.get(_mt_policy_id, EDGE_POLICY_GPU_EPHEMERAL)
                _synced_modes = [
                    _sync_id_to_mode.get(int(_id), _mt_ckpt.T0_RECOMPUTE)
                    for _id in _local_mode_ids.tolist()
                ]

                if _mt_policy == EDGE_POLICY_GPU_PERSIST:
                    _cuda_empty_cache(args)
                    _ei_mt, _ei_mt_ok = _try_cache_edge_index_on_gpu(
                        edge_index_global, device, rank=args.rank, safety_factor=0.95
                    )
                    if sp_world_size > 1 and dist.is_initialized():
                        _cache_ok_t = torch.tensor(
                            [int(_ei_mt_ok)], device=device, dtype=torch.long
                        )
                        dist.all_reduce(_cache_ok_t, op=dist.ReduceOp.MIN, group=sp_group)
                        _ei_mt_ok = bool(int(_cache_ok_t.item()))
                    if _ei_mt_ok:
                        edge_index_global = _ei_mt
                        clear_merged_edge_cache()
                        clear_random_walk_graph_cache()
                        edge_index_gpu_cached = True
                        args._runtime_edge_policy = EDGE_POLICY_GPU_PERSIST
                        args._runtime_force_edge_broadcast = False
                        args._runtime_edge_policy_locked = True
                        _mt_ckpt.override_active_plan(EDGE_POLICY_GPU_PERSIST, _synced_modes)
                        if args.rank == 0:
                            _mib = (
                                edge_index_global.numel()
                                * edge_index_global.element_size()
                                / (1024 ** 2)
                            )
                            print(
                                f"[multi_tier] edge_index_global cached on GPU "
                                f"({_mib:.1f} MiB). edge_policy={EDGE_POLICY_GPU_PERSIST}."
                            )
                    else:
                        if _ei_mt is not None:
                            del _ei_mt
                        _fallback_policy = _mt_ckpt.best_nonpersistent_edge_policy
                        _fallback_policy_id = _policy_to_id.get(_fallback_policy, 1)
                        # Fetch the tier plan the planner computed for the fallback
                        # policy (edge_peak=0), which is less conservative than the
                        # GPU_PERSIST plan stored in _synced_modes.
                        _fallback_tier_ids = torch.tensor(
                            [_mode_to_sync_id.get(m, 2) for m in _mt_ckpt.best_nonpersistent_tiers],
                            device=device,
                            dtype=torch.long,
                        )
                        if sp_world_size > 1 and dist.is_initialized():
                            _fallback_policy_t = torch.tensor(
                                [_fallback_policy_id], device=device, dtype=torch.long
                            )
                            dist.all_reduce(_fallback_policy_t, op=dist.ReduceOp.MAX, group=sp_group)
                            _fallback_policy_id = int(_fallback_policy_t.item())
                            dist.all_reduce(_fallback_tier_ids, op=dist.ReduceOp.MAX, group=sp_group)
                        _fallback_policy = _id_to_policy.get(
                            _fallback_policy_id, EDGE_POLICY_GPU_EPHEMERAL
                        )
                        _fallback_modes = [
                            _sync_id_to_mode.get(int(i), _mt_ckpt.T0_RECOMPUTE)
                            for i in _fallback_tier_ids.tolist()
                        ]
                        args._runtime_edge_policy = _fallback_policy
                        args._runtime_force_edge_broadcast = (
                            _fallback_policy in (
                                EDGE_POLICY_CPU_BROADCAST,
                                EDGE_POLICY_CPU_BROADCAST_PREFETCH,
                            )
                        )
                        args._runtime_edge_policy_locked = True
                        _mt_ckpt.override_active_plan(_fallback_policy, _fallback_modes)
                        if args.rank == 0:
                            print(
                                "[multi_tier] OOM caching edge_index; "
                                f"falling back to edge_policy={_fallback_policy}."
                            )
                else:
                    args._runtime_edge_policy = _mt_policy
                    args._runtime_force_edge_broadcast = (
                        _mt_policy in (
                            EDGE_POLICY_CPU_BROADCAST,
                            EDGE_POLICY_CPU_BROADCAST_PREFETCH,
                        )
                    )
                    args._runtime_edge_policy_locked = True
                    _mt_ckpt.override_active_plan(_mt_policy, _synced_modes)
                    if args.rank == 0:
                        print(
                            f"[multi_tier] Applying edge_policy={_mt_policy} "
                            "(no persistent full-edge cache)."
                        )
                _cuda_empty_cache(args)

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        del out_local, loss, local_y_eff, valid_train_mask, local_train_idx_eff
        if edge_index_rw is not None:
            del edge_index_rw

        cpu_rss = _get_process_rss_mib()
        cpu_rss_peak = _get_process_peak_rss_mib()
        comm_profile = None
        if profile_sp_comm:
            comm_profile = _aggregate_comm_profile(get_comm_profile_summary(reset=True))
            edge_broadcast = comm_profile.get("edge_broadcast") if comm_profile else None
            if edge_broadcast and edge_broadcast.get("kind", "timing") == "timing":
                edge_broadcast_ms_sum += float(edge_broadcast["total_ms"])
                edge_broadcast_bytes_sum += int(edge_broadcast["total_bytes"])
                edge_broadcast_epoch_count += 1

        eval_time = 0.0
        if epoch % args.eval_every == 0:
            t_eval = time.time()
            accs = _eval_sp(
                args,
                model,
                x_local,
                y_eval,
                split_idx_eval,
                edge_index_global,
                num_nodes,
                device,
                rw_device,
                sp_group,
                sp_src_rank,
                sp_rank,
                sp_world_size,
                rank_start,
                rank_end,
                local_num_nodes,
                amp_dtype=amp_dtype,
                edge_budget_state=edge_budget_controller.current_state(),
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            )
            eval_time = time.time() - t_eval

            if args.rank == 0 and accs is not None:
                train_acc = accs.get("train", 0.0)
                val_acc = accs.get("valid", 0.0)
                test_acc = accs.get("test", 0.0)
                _use_rocauc = str(getattr(args, "dataset", "")).lower() == "genius"
                _fmt = (lambda v: f"{v:.4f}") if _use_rocauc else (lambda v: f"{v:.2%}")
                _metric_name = "ROC-AUC" if _use_rocauc else "Acc"
                print(f"  ↳ Eval ({eval_time:.2f}s) [{_metric_name}] | Train={_fmt(train_acc)}  Val={_fmt(val_acc)}  Test={_fmt(test_acc)}")
                if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = epoch
                    if args.save_model:
                        torch.save(model.state_dict(), os.path.join(args.model_dir, f"{args.dataset}_fg_sp.pkl"))
                if test_acc > best_test:
                    best_test = test_acc
                print(f"  ↳ Best: epoch={best_epoch}  val={_fmt(best_val)}  test={_fmt(best_test)}")
            del accs

        t_adjust_start = time.time()
        _maybe_update_edge_budget(
            args,
            adaptive_edge_budget_cfg,
            edge_budget_controller,
            epoch,
            model,
            x_local,
            local_y,
            probe_idx_global,
            local_probe_idx,
            edge_index_global,
            num_nodes,
            device,
            rw_device,
            sp_group,
            sp_src_rank,
            sp_rank,
            local_num_nodes,
            amp_dtype,
            sp_world_size,
            edge_index_csr=edge_index_csr,
        )
        adjust_time = time.time() - t_adjust_start
        total_adjustment_time += adjust_time

        # Notify the comm-aware checkpointer when the edge budget stabilises.
        # "Stable" means: patience exhausted (controller.frozen), past the
        # declared warmup window, or adaptive budget not enabled.
        # We call this once; notify_budget_frozen() is idempotent (no-op if
        # not in DEFERRED state).
        if not _budget_frozen_notified:
            _post_adjust_budget_state = dict(edge_budget_controller.current_state())
            _budget_is_stable = (
                edge_budget_controller.frozen
                or (
                    adaptive_edge_budget_cfg.warmup_epochs is not None
                    and epoch >= adaptive_edge_budget_cfg.warmup_epochs
                )
            )
            if _budget_is_stable:
                _budget_frozen_notified = True
                _set_runtime_edge_policy_for_phase(budget_phase_active=False)
                if hasattr(model, "comm_aware_notify_budget_frozen"):
                    model.comm_aware_notify_budget_frozen(
                        reuse_deferred_baseline=(
                            _adaptive_edge_budget_enabled(args)
                            and epoch_budget_state == _post_adjust_budget_state
                        )
                    )
                if args.rank == 0:
                    print(
                        f"[edge-cache] Edge budget stable at epoch {epoch} "
                        f"(real={edge_budget_controller.real_budget}, "
                        f"rw={edge_budget_controller.rw_budget}). "
                        "Checkpoint calibration will start next epoch."
                    )

        # Fallback for adaptive-budget epochs whose next-step budget is only
        # known after the end-of-epoch probe/update.
        # Submit as early as possible so the background thread overlaps with
        # dist.barrier and any remaining epoch overhead.
        if not _prefetch_submitted:
            _prefetch_submitted = _submit_prefetch(
                epoch + 1,
                budget_state=edge_budget_controller.current_state(),
            )

        if sp_world_size > 1:
            dist.barrier(group=sp_group)

        epoch_wall_time = time.time() - t_epoch - eval_time
        if sp_world_size > 1 and dist.is_initialized():
            epoch_wall_time_t = torch.tensor(
                [epoch_wall_time], device=device, dtype=torch.float64
            )
            dist.all_reduce(epoch_wall_time_t, op=dist.ReduceOp.MAX, group=sp_group)
            epoch_wall_time = float(epoch_wall_time_t.item())
        if epoch > 1:
            epoch_wall_time_sum += epoch_wall_time
            epoch_wall_time_count += 1

        if args.rank == 0:
            timer_fields = (
                f"rw={rw_time:.2f}s fwd={fwd_time:.2f}s bwd={bwd_time:.2f}s"
                f" opt={opt_time:.2f}s adj={adjust_time:.2f}s"
                f" prefetch_wait={prefetch_wait_time:.2f}s"
            )
            if profile_sp_comm:
                timer_fields += (
                    f" (autograd={autograd_bwd_time:.2f}s grad_sync={grad_sync_time:.2f}s)"
                )
            print(
                f"Epoch {epoch:04d} | loss={loss_val:.4f} (ema={loss_ema:.4f}) "
                f"| t={epoch_wall_time:.2f}s "
                f"({timer_fields}) "
                f"| cpu_rss={cpu_rss:.1f}/{cpu_rss_peak:.1f} MiB"
            )
            if comm_profile:
                for line in _format_comm_profile(comm_profile, rw_time + fwd_time + bwd_time):
                    print(line)
                for line in _format_edge_cardinality(comm_profile):
                    print(line)

    _prefetch_pool.shutdown(wait=False)

    if args.rank == 0:
        print(f"\n{'=' * 72}")
        if profile_sp_comm and edge_broadcast_epoch_count > 0:
            avg_broadcast_ms = edge_broadcast_ms_sum / edge_broadcast_epoch_count
            avg_broadcast_mib = (edge_broadcast_bytes_sum / edge_broadcast_epoch_count) / (1024.0 ** 2)
            naive_fanout_mib = avg_broadcast_mib * max(sp_world_size - 1, 0)
            print(
                "[comm-summary] edge_broadcast avg/epoch: "
                f"{avg_broadcast_ms:.2f} ms, "
                f"source_payload={avg_broadcast_mib:.2f} MiB, "
                f"naive_fanout_payload={naive_fanout_mib:.2f} MiB"
            )
        _use_rocauc = str(getattr(args, "dataset", "")).lower() == "genius"
        _fmt = (lambda v: f"{v:.4f}") if _use_rocauc else (lambda v: f"{v:.2%}")
        print(f"Done.  Best epoch={best_epoch}  Val={_fmt(best_val)}  Test={_fmt(best_test)}")
        if epoch_wall_time_count > 0:
            print(
                f"Avg epoch wall time (excluding epoch 1): "
                f"{epoch_wall_time_sum / epoch_wall_time_count:.2f}s"
            )
        else:
            print("Avg epoch wall time (excluding epoch 1): n/a")
        if edge_budget_controller.enabled:
            print(f"Timing: bootstrap={total_bootstrap_time:.2f}s  adjustment={total_adjustment_time:.2f}s")
        print(f"{'=' * 72}")

    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        if dist.is_initialized():
            mem = torch.tensor([alloc, reserved], device=device)
            gathered = [torch.zeros_like(mem) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, mem)
            if args.rank == 0:
                print("Peak GPU memory per rank (MiB):")
                for rank_id, tensor in enumerate(gathered):
                    print(f"  rank {rank_id}: allocated={tensor[0]:.1f}  reserved={tensor[1]:.1f}")
        elif args.rank == 0:
            print(f"Peak GPU memory: allocated={alloc:.1f} MiB  reserved={reserved:.1f} MiB")

    cpu_rss = _get_process_rss_mib()
    cpu_rss_peak = _get_process_peak_rss_mib()
    cpu_mem_device = device if torch.cuda.is_available() else "cpu"
    if dist.is_initialized():
        mem = torch.tensor([cpu_rss, cpu_rss_peak], device=cpu_mem_device, dtype=torch.float32)
        gathered = [torch.zeros_like(mem) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, mem)
        if args.rank == 0:
            print("CPU RSS per rank (MiB):")
            for rank_id, tensor in enumerate(gathered):
                print(f"  rank {rank_id}: current={tensor[0]:.1f}  peak={tensor[1]:.1f}")
    elif args.rank == 0:
        print(f"CPU RSS: current={cpu_rss:.1f} MiB  peak={cpu_rss_peak:.1f} MiB")


if __name__ == "__main__":
    main()
