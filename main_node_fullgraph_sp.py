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
from utils.fullgraph_sp_runtime import (
    _aggregate_comm_profile,
    _apply_adaptive_edge_cache_decision,
    _apply_multi_tier_active_plan,
    _apply_multi_tier_gpu_memory_limit,
    _edge_seed_for_epoch,
    _format_comm_profile,
    _format_edge_cardinality,
    _pad_to_sequence_parallel_world,
    _prepare_exphormer_edges,
    _preprocess_graphormer_features,
    _print_cpu_rss,
    _print_peak_gpu_memory,
    _print_rank0_training_summary,
    _print_training_header,
    _pre_profile_gpu_cleanup,
    _profile_multi_tier_edge_policy,
    _should_force_ckpt_sync_timers,
    _should_sync_step_timers,
    _time_step_block,
)
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
    _build_model,
    _build_optimizer_bundle,
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
    _release_probe_caches_on_freeze,
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


def _parse_force_multi_tier_plan(spec: str, n_layers: int):
    """Parse --force_multi_tier_plan spec into (edge_policy, modes) or None on error.

    Format: "<edge_policy>:<tier_config>"
    tier_config: "recompute" | "keep_mha=N" | "retain=N"
    N = number of layers (from the back) to apply the non-recompute tier.
    """
    _VALID_POLICIES = {
        EDGE_POLICY_GPU_PERSIST,
        EDGE_POLICY_GPU_EPHEMERAL,
        EDGE_POLICY_CPU_BROADCAST_PREFETCH,
        EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
    }
    try:
        parts = spec.strip().split(":", 1)
        if len(parts) != 2:
            raise ValueError("expected '<edge_policy>:<tier_config>'")
        edge_policy, tier_spec = parts[0].strip(), parts[1].strip()
        if edge_policy not in _VALID_POLICIES:
            raise ValueError(f"unknown edge_policy {edge_policy!r}")
        if tier_spec == "recompute":
            modes = ["recompute"] * n_layers
        elif tier_spec.startswith("keep_mha="):
            k = int(tier_spec.split("=", 1)[1])
            if not (0 <= k <= n_layers):
                raise ValueError(f"keep_mha={k} out of range [0, {n_layers}]")
            modes = ["recompute"] * (n_layers - k) + ["keep_mha"] * k
        elif tier_spec.startswith("retain="):
            k = int(tier_spec.split("=", 1)[1])
            if not (0 <= k <= n_layers):
                raise ValueError(f"retain={k} out of range [0, {n_layers}]")
            modes = ["recompute"] * (n_layers - k) + ["retain"] * k
        else:
            raise ValueError(f"unknown tier_config {tier_spec!r}")
        print(
            f"[force_multi_tier_plan] Overriding planner: "
            f"edge_policy={edge_policy}  tiers={modes}"
        )
        return edge_policy, modes
    except Exception as exc:
        print(f"[force_multi_tier_plan] WARNING: could not parse {spec!r}: {exc}; ignoring override.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Full-Graph Node-Level Training with Sequence Parallel."
    )
    add_node_common_args(
        parser,
        defaults={
            "peak_lr": 1e-3,
            "warmup_updates": 5,
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

    feature, y, pad_num_nodes = _pad_to_sequence_parallel_world(
        feature, y, num_nodes, sp_world_size
    )
    feature = _preprocess_graphormer_features(args, feature, edge_index_global)
    _expander_edge_index_cpu, _ = _prepare_exphormer_edges(
        args, edge_index_global, pad_num_nodes, num_nodes
    )

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
        # Fixed budget path never builds probe caches, but call the helper to
        # keep this branch symmetric with the adaptive-freeze paths.
        _release_probe_caches_on_freeze(edge_budget_controller, args)
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

    x_local = feature[rank_start:rank_end].float().to(device)
    y_eval = y if args.rank == 0 else None
    split_idx_eval = split_idx if args.rank == 0 else None
    del feature
    if args.rank != 0:
        del y
        del split_idx

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

    def _set_runtime_edge_policy_for_phase(*, budget_phase_active: bool) -> None:
        # Once multi_tier finishes calibration and applies its synced ACTIVE
        # plan, preserve that runtime edge policy across later epochs.
        if bool(getattr(args, "_runtime_edge_policy_locked", False)):
            return
        if budget_phase_active or cpu_real_edge_sampling:
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
    if _is_multi_tier and getattr(model, "_comm_ckpt", None) is None:
        raise RuntimeError(
            "--activation_checkpoint_mode multi_tier was requested, but the model "
            "does not have a _MultiTierResourceManager attached."
        )
    edge_index_gpu_cached = False
    _adaptive_edge_decision_done = True  # True = no deferred work needed

    _cuda_empty_cache(args)
    if _is_multi_tier:
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
    elif cpu_real_edge_sampling:
        if args.rank == 0:
            print(
                "[edge-cache] Disabled: edge_build_device=cpu keeps real-edge sampling on CPU, "
                "so edge_index_global will not be cached on GPU."
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

    _print_training_header(
        args=args,
        sp_world_size=sp_world_size,
        sp_rank=sp_rank,
        nodes_per_rank=nodes_per_rank,
        rank_start=rank_start,
        rank_end=rank_end,
        edge_index_global=edge_index_global,
        rw_device=rw_device,
        real_edge_sampling_device=real_edge_sampling_device,
        edge_index_gpu_cached=edge_index_gpu_cached,
        model=model,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        edge_budget_controller=edge_budget_controller,
        fixed_edge_budget_state=fixed_edge_budget_state,
    )

    if args.rank == 0:
        print(f"Model params: {sum(param.numel() for param in model.parameters()):,}")

    optimizer, lr_scheduler, scaler = _build_optimizer_bundle(
        args, model, device, amp_dtype, n_train=int(train_idx_global.numel())
    )

    best_train_at_best_val = 0.0
    best_val = float("-inf")
    best_test_at_best_val = 0.0
    best_epoch = -1
    loss_ema = None
    epoch_wall_time_sum = 0.0
    epoch_wall_time_count = 0
    eval_time_sum = 0.0
    eval_time_count = 0
    # First eval skipped as warm-up (cuDNN autotune / allocator).
    _eval_warmup_done = False
    edge_broadcast_ms_sum = 0.0
    edge_broadcast_bytes_sum = 0
    edge_broadcast_epoch_count = 0
    _multi_tier_edge_profiled = not _is_multi_tier
    # Track whether we have applied the post-ACTIVE topology decision.
    _multi_tier_decision_done = not _is_multi_tier  # True = no deferred work needed
    # Adaptive-timing feedback: accumulate actual fwd/bwd/prefetch_wait after
    # the ACTIVE plan is set to detect cases where the profiling-time overlap
    # estimate was wrong (e.g. actual CPU build is slower under training load).
    _mt_adapt_fwd_sum = 0.0
    _mt_adapt_bwd_sum = 0.0
    _mt_adapt_wait_sum = 0.0
    # Track the single largest observed prefetch wait so we can exclude it from
    # the average passed to reconsider_with_actual_timing().  The first epoch
    # after ACTIVE has a one-time CSR+DGL build cost (~5s on snap-patents) that
    # would otherwise dominate a small-sample mean and spuriously trigger the
    # CPU→GPU switch; this max-sample outlier rejection gives steady-state.
    _mt_adapt_wait_max = 0.0
    _mt_adapt_count = 0
    _MT_ADAPT_WARMUP = 3   # epochs to collect before first check
    _MT_ADAPT_INTERVAL = 5 # re-check every N epochs after that
    # Disable further adaptive-timing checks once we have N consecutive
    # confirmations that the current CPU-prefetch policy is fine.  Repeated
    # re-checks every INTERVAL epochs after that are pure overhead — the
    # decision has stabilised.  Switching to GPU also disables checks (the
    # mechanism is one-way CPU→GPU only).
    _MT_ADAPT_KEEP_CONFIRM = 2
    _mt_adapt_keep_streak = 0
    _mt_adapt_check_disabled = False
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
        if epoch_idx <= 0:
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
            # When --force_multi_tier_plan is set, the edge-policy choice is
            # predetermined and the per-candidate profile is pure overhead
            # (each cpu_*_prefetch probe can add several GiB to CPU peak on
            # large graphs).  Inject zeroed profiles so the planner state
            # machine still reaches ACTIVE, then skip the costly profiling loop.
            _force_plan_str = str(getattr(args, "force_multi_tier_plan", "") or "")
            _mt_prof_mgr = getattr(model, "_comm_ckpt", None)
            if _force_plan_str and _mt_prof_mgr is not None:
                if args.rank == 0:
                    print(
                        f"[multi_tier] --force_multi_tier_plan={_force_plan_str!r} set; "
                        "skipping edge-policy profile to avoid the CPU peak it would cause."
                    )
                for _p in (
                    EDGE_POLICY_GPU_EPHEMERAL,
                    EDGE_POLICY_GPU_PERSIST,
                    EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
                    EDGE_POLICY_CPU_BROADCAST_PREFETCH,
                ):
                    _mt_prof_mgr.set_edge_policy_profile(
                        _p,
                        prep_time_s=0.0,
                        gpu_peak_bytes=0,
                        cpu_delta_bytes=0,
                        live_edge_bytes=0,
                        enabled=False,
                    )
                _multi_tier_edge_profiled = True
                if profile_sp_comm:
                    reset_comm_profiler()
                # Skip the remainder of the profile block.
                _mt_prof_mgr = None
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
                # Profile ephemeral before persist so the ephemeral measurement
                # sees the cleanest possible GPU state (persist pre-caches 4+ GiB
                # which, even after cleanup, can leave driver accounting stale and
                # trigger a spurious CPU-RW fallback in the ephemeral probe).
                _GPU_POLICIES = (EDGE_POLICY_GPU_EPHEMERAL, EDGE_POLICY_GPU_PERSIST)
                # cpu_delta threshold: a GPU policy whose cpu_delta exceeds half
                # the estimated RW working set was likely contaminated by a CPU
                # RW fallback and gets one clean retry.
                _RW_WORKING_SET_MIB = 12000  # rough estimate; guards retry logic
                _CPU_CONTAMINATION_THRESHOLD_MIB = _RW_WORKING_SET_MIB * 0.5
                for policy in (
                    EDGE_POLICY_GPU_EPHEMERAL,
                    EDGE_POLICY_GPU_PERSIST,
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
                    # For GPU policies: flush caching allocator + reset RW cache
                    # before each probe so prior policy's memory footprint does
                    # not skew the device-selection check.
                    if policy in _GPU_POLICIES and device.startswith("cuda"):
                        _pre_profile_gpu_cleanup(device)
                    _profile_kwargs = dict(
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
                    metrics = _profile_multi_tier_edge_policy(**_profile_kwargs)
                    # Contamination check: a GPU policy with large cpu_delta
                    # indicates the RW fell back to CPU (GPU was still dirty from
                    # the previous probe).  Retry once after a second cleanup pass.
                    if (
                        metrics is not None
                        and policy in _GPU_POLICIES
                        and device.startswith("cuda")
                        and metrics["cpu_delta_bytes"] / (1024 ** 2) > _CPU_CONTAMINATION_THRESHOLD_MIB
                    ):
                        if args.rank == 0:
                            print(
                                f"[multi_tier] edge_policy={policy}: cpu_delta="
                                f"{metrics['cpu_delta_bytes'] / (1024**2):.0f} MiB suggests "
                                f"CPU-RW fallback during probe; retrying after cleanup."
                            )
                        _pre_profile_gpu_cleanup(device)
                        metrics_retry = _profile_multi_tier_edge_policy(**_profile_kwargs)
                        if metrics_retry is not None:
                            metrics = metrics_retry
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

        # Release reserved-but-unused GPU allocator blocks before edge build so
        # the RW device check sees a realistic free-memory estimate.  This is the
        # main-training path only — the probe inside _maybe_update_edge_budget
        # intentionally does NOT get this flush, keeping its device check
        # conservative (model activations + probe RW + probe forward must all fit).
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        edge_index_rw, rw_time = _time_step_block(
            _build_edges_for_epoch,
            device=device,
            synchronize_cuda=sync_step_timing,
        )

        # Exphormer edge-type assembly.
        # Mode A (--expander_degree > 0): fixed expander as type 1, RW/real as type 0.
        # Mode B (default, no explicit expander): RW as type 1, real as type 0
        #   — edge_index_rw already has row-2 type info from _build_merged_edges;
        #   only self-loops need to be appended here as type 0.
        if _expander_edge_index_cpu is not None and edge_index_rw is not None:
            _exp = _expander_edge_index_cpu.to(edge_index_rw.device)
            # Strip existing row-2 (hop counts) if present — Exphormer uses row-2 for type
            _ei_base = edge_index_rw[:2] if edge_index_rw.size(0) == 3 else edge_index_rw
            # Add self-loops so every node can attend to itself (original: add_self_loops=True)
            _self_nn = torch.arange(num_nodes, dtype=torch.long, device=_ei_base.device)
            _self_loops_d = torch.stack([_self_nn, _self_nn], dim=0)
            _ei_with_self = torch.cat([_ei_base, _self_loops_d], dim=1)
            _n_real_self = _ei_with_self.size(1)
            _n_exp = _exp.size(1)
            _real_type = torch.zeros(_n_real_self, dtype=torch.long, device=_ei_base.device)
            _exp_type = torch.ones(_n_exp, dtype=torch.long, device=_ei_base.device)
            edge_index_rw = torch.cat([
                torch.cat([_ei_with_self, _exp], dim=1),
                torch.cat([_real_type, _exp_type]).unsqueeze(0),
            ], dim=0)  # (3, E_real + N_self + E_exp)
        elif (
            args.model == "exphormer"
            and int(getattr(args, "expander_degree", 0)) <= 0
            and edge_index_rw is not None
        ):
            # Mode B: edge_index_rw is already [3, E_real+E_rw] (real=0, rw=1).
            # Append self-loops as type 0 so every node attends to itself.
            _ei = edge_index_rw[:2]
            _existing_types = edge_index_rw[2]
            _self_nn = torch.arange(num_nodes, dtype=torch.long, device=_ei.device)
            _self_loops_d = torch.stack([_self_nn, _self_nn], dim=0)
            _self_types = torch.zeros(num_nodes, dtype=torch.long, device=_ei.device)
            edge_index_rw = torch.cat([
                torch.cat([_ei, _self_loops_d], dim=1),
                torch.cat([_existing_types, _self_types]).unsqueeze(0),
            ], dim=0)  # (3, E_real + E_rw + N_self)
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
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        def _step_with_oom_recovery():
            # Run one fwd/bwd/opt step, retrying with multi_tier fallbacks on CUDA OOM.
            # Returns (step_outputs, skipped): step_outputs is the 11-tuple from
            # _run_fwd_bwd_opt() on success (None if skipped); skipped is True when the
            # step was dropped after exhausting fallbacks.
            _oom_exc = None
            _MAX_OOM_RETRIES = 3
            _oom_retries = 0
            while True:
                local_oom = False
                try:
                    step_outputs = _run_fwd_bwd_opt()
                except torch.cuda.OutOfMemoryError as _e:
                    local_oom = True
                    _oom_exc = _e
                # Every rank must vote on OOM status: if one rank OOM'd inside an NCCL
                # collective, others are blocked there and can't reach this point — they
                # will surface as a timeout rather than a local OOM flag.
                if not _sp_any_oom(local_oom):
                    return step_outputs, False   # success
                # Aggressive cleanup before retry. The failed step may have left
                # autograd graph nodes, partial forward activations, and stray
                # .grad tensors alive in the CUDA caching allocator's reserved
                # segments. Plain empty_cache() only reclaims unused reserved
                # blocks — for the retry to actually find a large enough
                # contiguous block we have to:
                #   1. drop every gradient reference (autograd backward graphs
                #      hold cyclic refs to saved tensors via grad accumulators);
                #   2. drain in-flight CUDA work via synchronize so half-allocated
                #      tensors from the failed step are actually released;
                #   3. run gc twice (cyclic ref breakdown) with empty_cache in
                #      between to flush both Python objects and CUDA reservations;
                #   4. ipc_collect to release any cross-process IPC handles
                #      (no-op in single-process, free in distributed).
                optimizer.zero_grad(set_to_none=True)
                for _p in model.parameters():
                    if _p.grad is not None:
                        _p.grad = None
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        # Stream may carry secondary errors from the OOM; we've
                        # already captured the original exception, so swallow
                        # noise here and continue cleanup.
                        pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not _is_multi_tier:
                    if args.rank == 0:
                        reason = "local CUDA OOM" if local_oom else "peer rank reported CUDA OOM"
                        print(
                            f"[oom] {reason} at epoch {epoch}; "
                            "activation_checkpoint_mode is not multi_tier, aborting this run."
                        )
                    if local_oom and _oom_exc is not None:
                        raise _oom_exc
                    raise RuntimeError("peer rank reported CUDA OOM")
                # OOM fallback: in ACTIVE state, demote one keep_mha/retain tier;
                # in CALIBRATE_* states, disable the tier currently under probe
                # and advance the state machine. WARMUP_RECOMPUTE / DEFERRED
                # have no fallback — even all-recompute does not fit.
                _mt_mgr = getattr(model, "_comm_ckpt", None) if _is_multi_tier else None
                _can_fallback = (
                    _mt_mgr is not None
                    and _oom_retries < _MAX_OOM_RETRIES
                )
                if not _can_fallback:
                    # No multi_tier fallback or retries exhausted. Final safety
                    # net: drop this step (no gradient update), continue training
                    # at the next epoch. Allocator state often recovers across
                    # epochs, and adaptive budget probe may produce a smaller
                    # edge_index_eval shortly. Worst case we lose a few steps.
                    if args.rank == 0:
                        _reason = ("OOM with no further fallback"
                                   if local_oom else "peer rank reported OOM")
                        print(
                            f"[multi_tier] {_reason} at epoch {epoch} "
                            f"after {_oom_retries} retries; dropping this step."
                        )
                    return None, True
                if _mt_mgr.is_active():
                    _recovered = _mt_mgr.apply_oom_fallback(
                        device, sp_group=sp_group, sp_world_size=sp_world_size
                    )
                    _recovery_kind = "tier-demote"
                else:
                    _recovered = _mt_mgr.mark_current_calibration_infeasible(device)
                    _recovery_kind = "calibration-skip"
                if not _recovered:
                    # All tiers already at T0 (or OOM in WARMUP/DEFERRED) — no
                    # plan adjustment can help. Drop the step and continue.
                    if args.rank == 0:
                        print(
                            f"[multi_tier] OOM at epoch {epoch} with no remaining "
                            f"tier fallback (state={_mt_mgr.state_name}); "
                            f"dropping this step."
                        )
                    return None, True
                _oom_retries += 1
                if args.rank == 0:
                    print(
                        f"[multi_tier] OOM at epoch {epoch} ({_recovery_kind}); "
                        f"retry {_oom_retries}/{_MAX_OOM_RETRIES}; "
                        f"new state={_mt_mgr.state_name}."
                    )

        _step_outputs, _epoch_skipped_due_to_oom = _step_with_oom_recovery()

        # OOM safety net: if the retry loop exhausted its budget without
        # being able to complete a step, _run_fwd_bwd_opt() never returned
        # valid outputs. Skip the rest of this epoch (no eval, no prefetch,
        # no logging) and let the next epoch try again with a (hopefully)
        # cleaner allocator state.
        if _epoch_skipped_due_to_oom:
            if sp_world_size > 1 and dist.is_initialized():
                # Re-sync ranks so the next epoch starts together.
                try:
                    dist.barrier(group=sp_group)
                except Exception:
                    pass
            continue

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
        ) = _step_outputs

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
                edge_index_global, edge_index_gpu_cached = _apply_adaptive_edge_cache_decision(
                    model, args, edge_index_global, edge_index_gpu_cached,
                    device=device, sp_group=sp_group, sp_world_size=sp_world_size,
                )

        # multi_tier: apply topology-cache decision once ACTIVE.
        # Runs exactly once, after the planner finishes its last probe step.
        # Mirrors the adaptive-checkpoint sync pattern above but serves the
        # _MultiTierResourceManager instead of _CommAwareCheckpointer.
        if _is_multi_tier and not _multi_tier_decision_done:
            _mt_ckpt = getattr(model, "_comm_ckpt", None)
            if _mt_ckpt is not None and _mt_ckpt.is_active():
                _multi_tier_decision_done = True
                edge_index_global, edge_index_gpu_cached = _apply_multi_tier_active_plan(
                    model, args, edge_index_global, edge_index_gpu_cached,
                    device=device, sp_group=sp_group, sp_world_size=sp_world_size,
                )
                # --force_multi_tier_plan override (for ablation / manual testing).
                # Format: "<edge_policy>:<tier_config>"
                # Examples: "gpu_persist:recompute"  "gpu_ephemeral:keep_mha=2"
                _force_plan = str(getattr(args, "force_multi_tier_plan", "") or "")
                if _force_plan and _mt_ckpt is not None:
                    _forced = _parse_force_multi_tier_plan(_force_plan, _mt_ckpt.n_layers)
                    if _forced is not None:
                        _forced_policy, _forced_modes = _forced
                        _mt_ckpt.override_active_plan(
                            edge_policy=_forced_policy,
                            modes=_forced_modes,
                        )
                        # Keep edge_index_global/gpu_cached consistent with forced policy.
                        edge_index_global, edge_index_gpu_cached = _apply_multi_tier_active_plan(
                            model, args, edge_index_global, edge_index_gpu_cached,
                            device=device, sp_group=sp_group, sp_world_size=sp_world_size,
                        )

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
            training_epoch_edge_seed=epoch_edge_seed,
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
                # Drop probe-only caches the moment we leave budget-search.  This
                # is also done inside _maybe_update_edge_budget at the various
                # internal freeze points, but here it also covers the warmup-cap
                # stable branch where the controller itself never set .frozen.
                _release_probe_caches_on_freeze(edge_budget_controller, args)
                _ckpt_mgr = getattr(model, "_comm_ckpt", None)
                if _ckpt_mgr is not None and hasattr(model, "comm_aware_notify_budget_frozen"):
                    model.comm_aware_notify_budget_frozen(
                        reuse_deferred_baseline=(
                            _adaptive_edge_budget_enabled(args)
                            and epoch_budget_state == _post_adjust_budget_state
                        )
                    )
                if args.rank == 0:
                    ckpt_msg = (
                        "Checkpoint calibration will start next epoch."
                        if _ckpt_mgr is not None
                        else "No activation-checkpoint calibration is active."
                    )
                    print(
                        f"[edge-cache] Edge budget stable at epoch {epoch} "
                        f"(real={edge_budget_controller.real_budget}, "
                        f"rw={edge_budget_controller.rw_budget}). {ckpt_msg}"
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

        # eval_time is 0.0 here (eval has not run yet), so epoch_wall_time
        # correctly measures training-only time excluding eval.
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

        # Adaptive-timing feedback for multi_tier CPU-prefetch policies.
        # Accumulate actual timing and periodically re-evaluate whether the
        # profiling-time overlap estimate is still accurate.
        if _is_multi_tier and _multi_tier_decision_done and not _mt_adapt_check_disabled:
            _mt_adapt_fwd_sum += fwd_time
            _mt_adapt_bwd_sum += bwd_time
            _mt_adapt_wait_sum += prefetch_wait_time
            if prefetch_wait_time > _mt_adapt_wait_max:
                _mt_adapt_wait_max = prefetch_wait_time
            _mt_adapt_count += 1
            _do_adapt_check = (
                _mt_adapt_count == _MT_ADAPT_WARMUP
                or (_mt_adapt_count > _MT_ADAPT_WARMUP and (_mt_adapt_count - _MT_ADAPT_WARMUP) % _MT_ADAPT_INTERVAL == 0)
            )
            if _do_adapt_check:
                _mt_ckpt_adapt = getattr(model, "_comm_ckpt", None)
                if _mt_ckpt_adapt is not None and _mt_ckpt_adapt.is_active():
                    # Only the src SP rank submits the CPU prefetch, so only
                    # it accumulates non-zero prefetch_wait_time.  Non-src
                    # ranks have wait_sum=0 and would compute gain < 0 →
                    # wouldn't switch, causing a NCCL deadlock when the src
                    # rank enters _apply_multi_tier_active_plan and the others
                    # don't.  Fix: MAX all_reduce the wait sum so every rank
                    # uses the src's true observed wait before deciding.
                    _synced_wait_sum = _mt_adapt_wait_sum
                    _synced_wait_max = _mt_adapt_wait_max
                    if sp_world_size > 1 and dist.is_initialized():
                        _wt = torch.tensor(
                            [_mt_adapt_wait_sum, _mt_adapt_wait_max],
                            device=device,
                            dtype=torch.float64,
                        )
                        dist.all_reduce(_wt, op=dist.ReduceOp.MAX, group=sp_group)
                        _synced_wait_sum = float(_wt[0].item())
                        _synced_wait_max = float(_wt[1].item())
                    _avg_t_model = (_mt_adapt_fwd_sum + _mt_adapt_bwd_sum) / _mt_adapt_count
                    # Exclude the largest single wait observation (one-time
                    # cold-start cost for CPU CSR/DGL build).  Falls back to
                    # the plain mean when only one sample exists.
                    if _mt_adapt_count > 1:
                        _avg_wait = (_synced_wait_sum - _synced_wait_max) / (_mt_adapt_count - 1)
                    else:
                        _avg_wait = _synced_wait_sum / _mt_adapt_count
                    _pre_check_policy = _mt_ckpt_adapt.edge_policy
                    _switched = _mt_ckpt_adapt.reconsider_with_actual_timing(
                        actual_t_model_s=_avg_t_model,
                        actual_prefetch_wait_s=_avg_wait,
                    )
                    if _switched:
                        # Drain any in-flight CPU-prefetch future — it was
                        # built for the old policy and will be a cache miss
                        # after the switch. Waiting here (at most ~10s) avoids
                        # wasting a full epoch stall at the next epoch's start.
                        if _prefetch_future is not None:
                            try:
                                _prefetch_future.result()
                            except Exception:
                                pass
                            _prefetch_future = None
                        # Re-apply edge placement to match the updated policy.
                        edge_index_global, edge_index_gpu_cached = _apply_multi_tier_active_plan(
                            model, args, edge_index_global, edge_index_gpu_cached,
                            device=device, sp_group=sp_group, sp_world_size=sp_world_size,
                        )
                        # We just left a CPU-prefetch policy for a GPU one; the
                        # adaptive-timing mechanism is one-way (no GPU→CPU
                        # switch path exists), so future checks would be pure
                        # overhead.  Disable them.
                        _mt_adapt_check_disabled = True
                        _mt_adapt_keep_streak = 0
                        # Reset accumulators so next window reflects the new policy.
                        _mt_adapt_fwd_sum = 0.0
                        _mt_adapt_bwd_sum = 0.0
                        _mt_adapt_wait_sum = 0.0
                        _mt_adapt_wait_max = 0.0
                        _mt_adapt_count = 0
                    elif _pre_check_policy in (
                        EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
                        EDGE_POLICY_CPU_BROADCAST_PREFETCH,
                    ):
                        # Genuine "CPU is fine" decision (vs the early-return
                        # path inside reconsider for non-CPU policies which
                        # also returns False but does not represent a real
                        # check).  Count consecutive confirmations.
                        _mt_adapt_keep_streak += 1
                        if _mt_adapt_keep_streak >= _MT_ADAPT_KEEP_CONFIRM:
                            _mt_adapt_check_disabled = True
                            if args.rank == 0:
                                print(
                                    f"[MultiTierManager] adaptive-timing checks disabled: "
                                    f"{_mt_adapt_keep_streak} consecutive checks confirm "
                                    f"{_pre_check_policy} is optimal; stopping further re-evaluation."
                                )

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

        if epoch % args.eval_every == 0:
            if args.model == "exphormer":
                _expander_deg = int(getattr(args, "expander_degree", 0))
                _ei_cpu = (
                    edge_index_global
                    if edge_index_global.device.type == "cpu"
                    else edge_index_global.cpu()
                )
                _self_nn = torch.arange(num_nodes, dtype=torch.long)
                _self_loops = torch.stack([_self_nn, _self_nn], dim=0)
                if _expander_deg <= 0:
                    _n_real_self = _ei_cpu.shape[1] + num_nodes
                    _all_eval = torch.cat([_ei_cpu, _self_loops], dim=1)
                    _eval_types = torch.zeros(_n_real_self, dtype=torch.long)
                else:
                    _n_real_self = _ei_cpu.shape[1] + num_nodes
                    _n_exp = _expander_edge_index_cpu.shape[1]
                    _all_eval = torch.cat([_ei_cpu, _self_loops, _expander_edge_index_cpu], dim=1)
                    _eval_types = torch.cat([
                        torch.zeros(_n_real_self, dtype=torch.long),
                        torch.ones(_n_exp, dtype=torch.long),
                    ])
                _eval_ei_cpu = torch.cat([_all_eval, _eval_types.unsqueeze(0)], dim=0)
                del _all_eval, _eval_types
                _eval_cached_ei = _eval_ei_cpu.to(device)
                del _eval_ei_cpu
            else:
                _eval_cached_ei = None
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            t_eval = time.perf_counter()
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
                cached_edge_index=_eval_cached_ei,
                edge_budget_state=edge_budget_controller.current_state(),
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            eval_time = time.perf_counter() - t_eval
            if args.rank == 0:
                # Skip first eval as warm-up.
                if not _eval_warmup_done:
                    _eval_warmup_done = True
                else:
                    eval_time_sum += eval_time
                    eval_time_count += 1

            if args.rank == 0 and accs is not None:
                train_acc = accs.get("train", 0.0)
                val_acc = accs.get("valid", 0.0)
                test_acc = accs.get("test", 0.0)
                _use_rocauc = str(getattr(args, "dataset", "")).lower() == "genius"
                _fmt = (lambda v: f"{v:.4f}") if _use_rocauc else (lambda v: f"{v:.2%}")
                _metric_name = "ROC-AUC" if _use_rocauc else "Acc"
                print(f"  ↳ Eval ({eval_time:.2f}s) [{_metric_name}] | Train={_fmt(train_acc)}  Val={_fmt(val_acc)}  Test={_fmt(test_acc)}")
                if val_acc > best_val:
                    best_train_at_best_val = train_acc
                    best_val = val_acc
                    best_test_at_best_val = test_acc
                    best_epoch = epoch
                    if args.save_model:
                        torch.save(model.state_dict(), os.path.join(args.model_dir, f"{args.dataset}_fg_sp.pkl"))
                print(
                    f"  ↳ Best by Val: epoch={best_epoch}  "
                    f"train={_fmt(best_train_at_best_val)}  "
                    f"val={_fmt(best_val)}  "
                    f"test={_fmt(best_test_at_best_val)}"
                )
            del accs

    _prefetch_pool.shutdown(wait=False)

    _print_rank0_training_summary(
        args=args,
        profile_sp_comm=profile_sp_comm,
        edge_broadcast_epoch_count=edge_broadcast_epoch_count,
        edge_broadcast_ms_sum=edge_broadcast_ms_sum,
        edge_broadcast_bytes_sum=edge_broadcast_bytes_sum,
        sp_world_size=sp_world_size,
        best_epoch=best_epoch,
        best_train_at_best_val=best_train_at_best_val,
        best_val=best_val,
        best_test_at_best_val=best_test_at_best_val,
        epoch_wall_time_sum=epoch_wall_time_sum,
        epoch_wall_time_count=epoch_wall_time_count,
        eval_time_sum=eval_time_sum,
        eval_time_count=eval_time_count,
        num_nodes=num_nodes,
        edge_budget_controller=edge_budget_controller,
        total_adjustment_time=total_adjustment_time,
    )
    _print_peak_gpu_memory(args, device)
    _print_cpu_rss(args, device)


if __name__ == "__main__":
    main()
