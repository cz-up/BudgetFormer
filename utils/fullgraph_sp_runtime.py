"""Runtime helpers for ``main_node_fullgraph_sp.py``.

This module keeps the entry script focused on orchestration.  The helpers here
are intentionally thin wrappers around existing support functions; they should
not own model or training semantics.
"""

from __future__ import annotations

import gc
import time

import torch
import torch.distributed as dist

from gt_sp.utils import clear_random_walk_graph_cache, _clear_rw_device_cache
from utils.fullgraph_sp_support import (
    EDGE_POLICY_CPU_BROADCAST,
    EDGE_POLICY_CPU_BROADCAST_PREFETCH,
    EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
    EDGE_POLICY_GPU_EPHEMERAL,
    EDGE_POLICY_GPU_PERSIST,
    _build_attention_edges,
    _compute_laplacian_pe,
    _compute_multihop_features,
    _cuda_empty_cache,
    _generate_expander_edges,
    _get_process_peak_rss_mib,
    _get_process_rss_mib,
    _random_block_sampling_enabled,
    _try_cache_edge_index_on_gpu,
    _use_real_edges,
    _use_rw_edges,
    clear_merged_edge_cache,
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


def _pre_profile_gpu_cleanup(device: str) -> None:
    """Best-effort GPU cleanup before profiling a policy.

    Clears the PyTorch caching allocator and resets the RW-device decision
    cache so that each policy sees an uncontaminated GPU memory estimate.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    _clear_rw_device_cache()


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
    cpu_delta_bytes = 0
    prep_time = 0.0
    policy_oom = False
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
    except torch.cuda.OutOfMemoryError as oom:
        policy_oom = True
        if args.rank == 0:
            print(
                f"[multi_tier] OOM while profiling edge_policy={policy!r}; "
                f"marking infeasible and continuing with other policies. "
                f"({oom})"
            )
    finally:
        if "edge_index_probe" in locals():
            del edge_index_probe
        if probe_cached and probe_edge_index_global is not edge_index_global:
            del probe_edge_index_global
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        _cuda_empty_cache(args)
        if policy_oom:
            return None

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


def _apply_adaptive_edge_cache_decision(
    model,
    args,
    edge_index_global,
    edge_index_gpu_cached,
    *,
    device,
    sp_group,
    sp_world_size,
):
    # Sync the comm-aware checkpointer's CACHE/NO-CACHE vote across SP ranks
    # (MIN: a single NO-CACHE vote overrides) and apply it. On CACHE, move
    # edge_index to GPU for rank-local builds (broadcast fallback on OOM).
    # Returns the (possibly updated) (edge_index_global, edge_index_gpu_cached).
    _comm_ckpt_check = getattr(model, "_comm_ckpt", None)
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
    return edge_index_global, edge_index_gpu_cached


def _apply_multi_tier_active_plan(
    model,
    args,
    edge_index_global,
    edge_index_gpu_cached,
    *,
    device,
    sp_group,
    sp_world_size,
):
    # Sync the planner's ACTIVE edge-policy + tier plan across SP ranks and apply it.
    # For GPU_PERSIST, cache edge_index on GPU (falling back to the planner's best
    # non-persistent policy on OOM). Returns the (possibly updated)
    # (edge_index_global, edge_index_gpu_cached).
    _mt_ckpt = getattr(model, "_comm_ckpt", None)
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
    return edge_index_global, edge_index_gpu_cached


def _pad_to_sequence_parallel_world(feature, y, num_nodes: int, sp_world_size: int):
    pad_num_nodes = ((num_nodes + sp_world_size - 1) // sp_world_size) * sp_world_size
    if pad_num_nodes <= num_nodes:
        return feature, y, pad_num_nodes
    pad_rows = pad_num_nodes - num_nodes
    feature = torch.cat([feature, feature.new_zeros(pad_rows, feature.shape[1])], dim=0)
    y = torch.cat([y, y.new_full((pad_rows,) + y.shape[1:], -1)], dim=0)
    return feature, y, pad_num_nodes


def _preprocess_graphormer_features(args, feature, edge_index_global):
    should_preprocess = (
        args.model in ("graphormer", "gt")
        and (int(getattr(args, "hops", 0)) > 0 or int(getattr(args, "pe_dim", 0)) > 0)
    )
    if not should_preprocess:
        return feature

    hops = int(getattr(args, "hops", 0))
    pe_dim = int(getattr(args, "pe_dim", 0))
    if hops > 0:
        feat_3d = _compute_multihop_features(
            edge_index_global,
            feature.shape[0],
            feature,
            hops,
            pe_dim=pe_dim,
            rank=args.rank,
        )
        feature = feat_3d.mean(dim=1).contiguous()
        del feat_3d
        return feature

    if args.rank == 0:
        print(f"[graphormer-preprocess] Adding Laplacian PE (pe_dim={pe_dim}) to raw features ...")
    row_bi = torch.cat([edge_index_global[0].cpu(), edge_index_global[1].cpu()])
    col_bi = torch.cat([edge_index_global[1].cpu(), edge_index_global[0].cpu()])
    self_nodes = torch.arange(feature.shape[0], dtype=torch.long)
    row_aug = torch.cat([row_bi, self_nodes])
    col_aug = torch.cat([col_bi, self_nodes])
    lpe = _compute_laplacian_pe(row_aug, col_aug, feature.shape[0], pe_dim)
    return torch.cat([feature.float().cpu(), lpe], dim=1)


def _prepare_exphormer_edges(args, edge_index_global, pad_num_nodes: int, num_nodes: int):
    expander_edge_index_cpu = None
    exphormer_eval_ei = None
    expander_degree = int(getattr(args, "expander_degree", 0))
    if expander_degree <= 0:
        return expander_edge_index_cpu, exphormer_eval_ei

    if args.rank == 0:
        print(
            f"[exphormer] Generating expander edges "
            f"(N={pad_num_nodes:,}, degree={expander_degree}) ..."
        )
    expander_edge_index_cpu = _generate_expander_edges(
        pad_num_nodes,
        expander_degree,
        seed=int(getattr(args, "seed", 0)),
    )
    if args.rank == 0:
        print(f"[exphormer] Expander edges: {expander_edge_index_cpu.shape[1]:,}")

    self_nodes = torch.arange(num_nodes, dtype=torch.long)
    self_loops_cpu = torch.stack([self_nodes, self_nodes], dim=0)
    n_real = edge_index_global.shape[1]
    n_self = num_nodes
    n_exp = expander_edge_index_cpu.shape[1]
    real_self_type = torch.zeros(n_real + n_self, dtype=torch.long)
    exp_type_cpu = torch.ones(n_exp, dtype=torch.long)
    exphormer_eval_ei = torch.cat(
        [
            torch.cat([edge_index_global.cpu(), self_loops_cpu, expander_edge_index_cpu], dim=1),
            torch.cat([real_self_type, exp_type_cpu]).unsqueeze(0),
        ],
        dim=0,
    )
    if args.rank == 0:
        print(
            f"[exphormer] Eval edge_index: real={n_real:,} + self-loops={n_self:,} "
            f"+ expander={n_exp:,} = {exphormer_eval_ei.shape[1]:,}"
        )
    return expander_edge_index_cpu, exphormer_eval_ei


def _print_training_header(
    *,
    args,
    sp_world_size: int,
    sp_rank: int,
    nodes_per_rank: int,
    rank_start: int,
    rank_end: int,
    edge_index_global,
    rw_device,
    real_edge_sampling_device,
    edge_index_gpu_cached: bool,
    model,
    adaptive_edge_budget_cfg,
    edge_budget_controller,
    fixed_edge_budget_state,
):
    if args.rank != 0:
        return

    print(f"\n{'=' * 72}")
    print(f"Full-Graph SP Training  (sp_world_size={sp_world_size})")
    print(f"  Nodes per rank: {nodes_per_rank:,}  (rank {sp_rank}: {rank_start:,}-{rank_end:,})")
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
    ckpt_mgr = getattr(model, "_comm_ckpt", None)
    print(
        f"  Activation checkpoint: requested={getattr(args, 'activation_checkpoint_mode', None)} "
        f"model={getattr(model, 'activation_checkpoint_mode', 'n/a')} "
        f"enabled={int(bool(getattr(model, 'activation_checkpoint', False)))}"
    )
    if ckpt_mgr is not None:
        print(
            f"  Multi-tier manager: state={getattr(ckpt_mgr, 'state_name', 'unknown')} "
            f"active={int(bool(ckpt_mgr.is_active()))}"
        )
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
        f"static_seed={adaptive_edge_budget_cfg.static_seed_epochs})"
    )
    if edge_budget_controller.enabled:
        print(
            "  Budget-phase baseline: cpu_build_broadcast + all-recompute "
            "(until the edge budget stabilizes)"
        )
    print(f"{'=' * 72}\n")


def _metric_formatter(args):
    use_rocauc = str(getattr(args, "dataset", "")).lower() == "genius"
    return (lambda v: f"{v:.4f}") if use_rocauc else (lambda v: f"{v:.2%}")


def _print_rank0_training_summary(
    *,
    args,
    profile_sp_comm: bool,
    edge_broadcast_epoch_count: int,
    edge_broadcast_ms_sum: float,
    edge_broadcast_bytes_sum: int,
    sp_world_size: int,
    best_epoch: int,
    best_train_at_best_val: float,
    best_val: float,
    best_test_at_best_val: float,
    epoch_wall_time_sum: float,
    epoch_wall_time_count: int,
    eval_time_sum: float,
    eval_time_count: int,
    num_nodes: int,
    edge_budget_controller,
    total_bootstrap_time: float,
    total_adjustment_time: float,
):
    if args.rank != 0:
        return

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

    fmt = _metric_formatter(args)
    print(
        f"Done.  Best by Val epoch={best_epoch}  "
        f"Train={fmt(best_train_at_best_val)}  "
        f"Val={fmt(best_val)}  "
        f"Test={fmt(best_test_at_best_val)}"
    )
    if epoch_wall_time_count > 0:
        print(
            f"Avg epoch wall time (excluding epoch 1): "
            f"{epoch_wall_time_sum / epoch_wall_time_count:.2f}s"
        )
    else:
        print("Avg epoch wall time (excluding epoch 1): n/a")
    if eval_time_count > 0:
        mean_eval_s = eval_time_sum / eval_time_count
        fullgraph_thr = num_nodes / mean_eval_s if mean_eval_s > 0 else 0.0
        print(
            f"Avg inference time per eval (excluding warm-up): "
            f"{mean_eval_s * 1000:.2f} ms  "
            f"({num_nodes} nodes, throughput={fullgraph_thr:,.0f} nodes/s)"
        )
    else:
        print("Avg inference time per eval: n/a")
    if edge_budget_controller.enabled:
        print(f"Timing: bootstrap={total_bootstrap_time:.2f}s  adjustment={total_adjustment_time:.2f}s")
    print(f"{'=' * 72}")


def _print_peak_gpu_memory(args, device: str):
    if not torch.cuda.is_available():
        return
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


def _print_cpu_rss(args, device: str):
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
