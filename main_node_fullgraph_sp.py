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
import os
import time

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
from gt_sp.utils import resolve_edge_build_device
from utils.fullgraph_sp_support import (
    _AdaptiveEdgeBudgetController,
    _adaptive_edge_budget_enabled,
    _autocast_context,
    _build_attention_edges,
    _build_dst_csr,
    _build_model,
    _build_optimizer_bundle,
    _bootstrap_initial_edge_budget,
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
        "edge_broadcast",
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
    enable_comm_profiler(bool(getattr(args, "profile_sp_comm", False)))

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
    nodes_per_rank = pad_num_nodes // sp_world_size
    rank_start = sp_rank * nodes_per_rank
    rank_end = rank_start + nodes_per_rank
    local_num_nodes = nodes_per_rank

    if args.model in ("graphormer", "acm") and args.num_global_node > 0:
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
    edge_index_gpu_cached = False
    _adaptive_edge_decision_done = True  # True = no deferred work needed

    _cuda_empty_cache(args)
    if cpu_real_edge_sampling:
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
        # ---- Adaptive: measure t_h2d, register with checkpointer, defer --
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
        print(f"  Edge index GPU cached: {int(edge_index_gpu_cached)}")
        print(f"  AMP dtype: {args.amp_dtype}")
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
    # For adaptive checkpoint + adaptive edge budget: track whether we have
    # already notified the checkpointer that the edge budget is frozen.
    # For all other cases (no deferred calibration), treat as already done.
    _budget_frozen_notified = not (
        _is_adaptive_ckpt and _adaptive_edge_budget_enabled(args)
    )

    cached_edge_index = None
    if not dynamic_edges:
        cached_edge_index = _build_attention_edges(
            args,
            edge_index_global,
            num_nodes,
            device,
            rw_device,
            sp_group,
            sp_src_rank,
            sp_rank,
            local_num_nodes,
            edge_budget_state=edge_budget_controller.current_state(),
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        )

    use_epoch_seed = (
        random_blocks_dynamic
    )

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        if getattr(args, "profile_sp_comm", False):
            reset_comm_profiler()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        t_rw = time.time()
        if cached_edge_index is not None:
            edge_index_rw = cached_edge_index
        else:
            if use_epoch_seed:
                if epoch <= adaptive_edge_budget_cfg.static_seed_epochs:
                    edge_seed = args.seed
                else:
                    edge_seed = args.seed + epoch
            else:
                edge_seed = None
            edge_index_rw = _build_attention_edges(
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
                edge_budget_state=edge_budget_controller.current_state(),
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            )
        rw_time = time.time() - t_rw

        t_fwd = time.time()
        with _autocast_context(device, amp_dtype):
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
        fwd_time = time.time() - t_fwd

        t_bwd = time.time()
        grad_reducer.prepare_backward()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_reducer.finalize_backward()
        bwd_time = time.time() - t_bwd

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step()

        # Notify comm-aware checkpointer that this step is complete.
        # Must happen after optimizer.step() so that optimizer state (lazily
        # allocated on the first step) is included in the peak measurement.
        if hasattr(model, "comm_aware_notify_step_end"):
            model.comm_aware_notify_step_end(device, t_bwd=bwd_time)

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

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val

        del out_local, loss, local_y_eff, valid_train_mask, local_train_idx_eff
        if cached_edge_index is None:
            del edge_index_rw

        cpu_rss = _get_process_rss_mib()
        cpu_rss_peak = _get_process_peak_rss_mib()
        comm_profile = None
        if getattr(args, "profile_sp_comm", False):
            comm_profile = _aggregate_comm_profile(get_comm_profile_summary(reset=True))
            edge_broadcast = comm_profile.get("edge_broadcast") if comm_profile else None
            if edge_broadcast and edge_broadcast.get("kind", "timing") == "timing":
                edge_broadcast_ms_sum += float(edge_broadcast["total_ms"])
                edge_broadcast_bytes_sum += int(edge_broadcast["total_bytes"])
                edge_broadcast_epoch_count += 1

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
                cached_edge_index=cached_edge_index,
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
        total_adjustment_time += (time.time() - t_adjust_start)

        # Notify the comm-aware checkpointer when the edge budget stabilises.
        # "Stable" means: patience exhausted (controller.frozen), past the
        # declared warmup window, or adaptive budget not enabled.
        # We call this once; notify_budget_frozen() is idempotent (no-op if
        # not in DEFERRED state).
        if not _budget_frozen_notified:
            _budget_is_stable = (
                edge_budget_controller.frozen
                or (
                    adaptive_edge_budget_cfg.warmup_epochs is not None
                    and epoch >= adaptive_edge_budget_cfg.warmup_epochs
                )
            )
            if _budget_is_stable:
                _budget_frozen_notified = True
                if hasattr(model, "comm_aware_notify_budget_frozen"):
                    model.comm_aware_notify_budget_frozen()
                if args.rank == 0:
                    print(
                        f"[edge-cache] Edge budget stable at epoch {epoch} "
                        f"(real={edge_budget_controller.real_budget}, "
                        f"rw={edge_budget_controller.rw_budget}). "
                        "Checkpoint calibration will start next epoch."
                    )

        if sp_world_size > 1:
            dist.barrier(group=sp_group)

        epoch_wall_time = time.time() - t_epoch
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
            print(
                f"Epoch {epoch:04d} | loss={loss_val:.4f} (ema={loss_ema:.4f}) "
                f"| t={epoch_wall_time:.2f}s "
                f"(rw={rw_time:.2f}s fwd={fwd_time:.2f}s bwd={bwd_time:.2f}s) "
                f"| cpu_rss={cpu_rss:.1f}/{cpu_rss_peak:.1f} MiB"
            )
            if comm_profile:
                for line in _format_comm_profile(comm_profile, rw_time + fwd_time + bwd_time):
                    print(line)
                for line in _format_edge_cardinality(comm_profile):
                    print(line)

    if args.rank == 0:
        print(f"\n{'=' * 72}")
        if getattr(args, "profile_sp_comm", False) and edge_broadcast_epoch_count > 0:
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
