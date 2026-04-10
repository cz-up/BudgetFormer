import contextlib
import gc
import os
import random
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import coalesce

from gt_sp.comm_profiler import profile_call, record_scalar
from gt_sp.reducer import build_gradient_reducer
from gt_sp.utils import (
    _merge_edge_index_list,
    adjust_edge_index_nomerge,
    build_head_hop_edges,
    fix_edge_index,
    fixed_random_seed,
    random_split_idx,
)
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from utils.lr import PolynomialDecayLR
from utils.split_utils import load_default_split

_STREAM_EDGE_CHUNK_EDGES = 1 << 19
_MINHASH_SAMPLE_K_LIMIT = 8


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def _resolve_amp_dtype(args):
    amp_mode = getattr(args, "amp_dtype", "none")
    if amp_mode == "none" or not torch.cuda.is_available():
        return None
    if amp_mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if args.rank == 0:
            print("[amp] bf16 is not supported on this GPU; falling back to fp16.")
        return torch.float16
    if amp_mode == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_mode}")


def _autocast_context(device: str, amp_dtype):
    if amp_dtype is None or not device.startswith("cuda"):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_process_rss_mib() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            fields = handle.readline().strip().split()
        if len(fields) >= 2:
            rss_pages = int(fields[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return rss_pages * page_size / (1024 ** 2)
    except Exception:
        pass
    return _get_process_peak_rss_mib()


def _get_process_peak_rss_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 ** 2)
    return usage / 1024.0


def _cuda_empty_cache(args) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _try_cache_edge_index_on_gpu(
    edge_index_global: torch.Tensor,
    device: str,
    rank: int = 0,
    safety_factor: float = 0.5,
):
    """Attempt to move *edge_index_global* to GPU for persistent caching.

    Avoids repeated host-to-device transfers in the rank-local edge-build path.
    The check uses *safety_factor* as a guard: we only cache when the tensor
    fits within ``safety_factor * free_GPU_bytes`` so that enough headroom
    remains for optimizer state and forward/backward activations.

    Returns:
        (gpu_tensor, True)   – allocation succeeded; caller should replace
                               edge_index_global with the returned tensor.
        (None, False)        – not enough memory or CUDA unavailable; caller
                               should keep the original CPU tensor and enable
                               the broadcast path (force_edge_broadcast=True).
    """
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return None, False
    if edge_index_global.device.type == "cuda":
        return edge_index_global, True  # already resident – nothing to do

    edge_bytes = edge_index_global.numel() * edge_index_global.element_size()
    gc.collect()
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    budget_bytes = int(free_bytes * safety_factor)

    if edge_bytes > budget_bytes:
        if rank == 0:
            print(
                f"[edge-cache] Insufficient GPU memory to cache edge_index_global "
                f"({edge_bytes / (1024 ** 2):.1f} MiB needed, "
                f"{budget_bytes / (1024 ** 2):.1f} MiB available "
                f"[free={free_bytes / (1024 ** 2):.1f} MiB × safety={safety_factor:.0%}]). "
                "Will use rank-0 broadcast path."
            )
        return None, False

    try:
        gpu_tensor = edge_index_global.to(device=device)
        return gpu_tensor, True
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(
                "[edge-cache] OOM while caching edge_index_global on GPU; "
                "falling back to broadcast path."
            )
        return None, False


def _measure_edge_h2d_time(edge_index_global: torch.Tensor, device: str) -> float:
    """Measure host-to-device transfer time for edge_index_global.

    Performs a single timed .to(device) call with CUDA synchronisation for
    accuracy, then immediately frees the GPU copy.  This keeps the warmup
    peak_warmup measurement (taken later) free of edge-index bytes.

    Returns elapsed seconds (0.0 when CUDA is unavailable or the tensor is
    already resident on the target device).
    """
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        return 0.0
    if edge_index_global.device.type == "cuda":
        return 0.0
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    _tmp = edge_index_global.to(device=device)
    torch.cuda.synchronize(device)
    t_h2d = time.perf_counter() - t0
    del _tmp
    torch.cuda.empty_cache()
    return t_h2d


def _to_bidirected_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index_bi = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return coalesce(edge_index_bi, num_nodes=num_nodes)


def _load_default_split(dataset_name: str, root_dir: str, wait_for_rank0: bool = False):
    return load_default_split(
        dataset_name,
        root_dir,
        dist_module=dist,
        wait_for_rank0=wait_for_rank0,
    )


def _load_data(args):
    data_path = args.dataset_dir + args.dataset
    feature = _torch_load_cpu(data_path + "/x.pt")
    y = _torch_load_cpu(data_path + "/y.pt")
    edge_index_global = _torch_load_cpu(data_path + "/edge_index.pt")
    num_nodes = feature.shape[0]

    if args.to_bidirected:
        edge_index_global = _to_bidirected_edge_index(edge_index_global, num_nodes=num_nodes)

    if args.dataset == "pokec":
        y = torch.clamp(y, min=0)

    split_idx = _load_default_split(args.dataset, args.dataset_dir, wait_for_rank0=True)
    if split_idx is None:
        if args.rank == 0:
            print("[split] No default split found, falling back to random 60/20/20 split.")
        split_idx = random_split_idx(y, 0.6, 0.2, 0.2, args.seed)
    else:
        if args.rank == 0:
            print("[split] Loaded official dataset split.")

    if args.rank == 0:
        print(args)
        if args.to_bidirected:
            print("[graph] Converted edge_index to bidirected after loading.")
        print(f"Dataset loaded: N={num_nodes:,}  E={edge_index_global.shape[1]:,}")
        print(
            f"  train={split_idx['train'].shape[0]:,}  "
            f"val={split_idx['valid'].shape[0]:,}  "
            f"test={split_idx['test'].shape[0]:,}"
        )
    return feature, y, edge_index_global, num_nodes, split_idx


def _torch_load_cpu(path: str):
    attempts = (
        {"weights_only": True, "mmap": True},
        {"weights_only": False, "mmap": True},
        {"weights_only": True},
        {"weights_only": False},
        {},
    )
    last_exc = None
    for extra_kwargs in attempts:
        kwargs = {"map_location": "cpu", **extra_kwargs}
        try:
            return torch.load(path, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
        except RuntimeError as exc:
            last_exc = exc
            if extra_kwargs:
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to load tensor from {path}")


def _build_model(args, feature, y, device):
    out_dim = int(y.max().item()) + 1
    common = dict(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=feature.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=out_dim,
        attn_bias_dim=args.attn_bias_dim,
        dropout_rate=args.dropout_rate,
        input_dropout_rate=args.input_dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        ffn_dim=args.ffn_dim,
    )
    if args.model == "graphormer":
        model = Graphormer(
            **common,
            num_global_node=args.num_global_node,
        ).to(device)
        _ckpt_mode = getattr(args, "activation_checkpoint_mode", None)
        # Defer checkpoint calibration when adaptive_edge_budget is also active:
        # peak_warmup measured before the budget stabilises is too small and
        # produces wrong keep_mha layer counts.
        _ckpt_deferred = (
            _ckpt_mode == "adaptive"
            and _adaptive_edge_budget_enabled(args)
        )
        model.set_activation_checkpoint(mode=_ckpt_mode, deferred=_ckpt_deferred)
    elif args.model == "gt":
        model = GT(
            **common,
            num_global_node=0,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model


@dataclass(frozen=True)
class _AdaptiveEdgeBudgetConfig:
    enabled: bool
    probe_size: int
    block_size: int
    warmup_epochs: Optional[int]
    patience: int
    gain_threshold: float
    max_total_edges_per_query: int
    bootstrap_search_epochs: int
    bootstrap_n_layers: int
    static_seed_epochs: int


def _pin_cpu_tensor(tensor):
    if tensor.device.type == "cpu" and torch.cuda.is_available():
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
    return tensor


def _random_block_sampling_enabled(args, adaptive_edge_budget_cfg=None) -> bool:
    if adaptive_edge_budget_cfg is not None and adaptive_edge_budget_cfg.enabled:
        return True
    if _fixed_edge_budget_enabled(args, adaptive_edge_budget_cfg):
        return True
    return bool(getattr(args, "random_edge_blocks", False))


def _adaptive_edge_budget_enabled(args) -> bool:
    return bool(getattr(args, "adaptive_edge_budget", False))


def _fixed_edge_budget_state_from_args(args):
    fixed_real = getattr(args, "fixed_real_edges_per_query", None)
    fixed_rw = getattr(args, "fixed_rw_edges_per_query", None)
    if fixed_real is None or fixed_rw is None:
        return None
    state = {
        "real_edges_per_query": int(fixed_real),
        "rw_edges_per_query": int(fixed_rw),
    }
    fixed_walk = getattr(args, "fixed_walk_length", None)
    if fixed_walk is not None:
        state["walk_length"] = int(fixed_walk)
    return state


def _fixed_edge_budget_enabled(args, adaptive_edge_budget_cfg=None) -> bool:
    if adaptive_edge_budget_cfg is not None and adaptive_edge_budget_cfg.enabled:
        return False
    return _fixed_edge_budget_state_from_args(args) is not None


def _get_real_edge_budget(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> int:
    state = edge_budget_state if edge_budget_state is not None else _fixed_edge_budget_state_from_args(args)
    if state is not None and "real_edges_per_query" in state:
        return int(state["real_edges_per_query"])
    if adaptive_edge_budget_cfg is not None:
        max_tot = int(adaptive_edge_budget_cfg.max_total_edges_per_query)
    else:
        max_tot = int(getattr(args, "max_total_edges_per_query", 0))
    if int(getattr(args, "head_hop_walks_per_node", 0)) <= 0:
        return max_tot
    return (max_tot + 1) // 2


def _get_rw_edge_budget(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> int:
    state = edge_budget_state if edge_budget_state is not None else _fixed_edge_budget_state_from_args(args)
    if state is not None and "rw_edges_per_query" in state:
        return int(state["rw_edges_per_query"])
    if int(getattr(args, "head_hop_walks_per_node", 0)) <= 0:
        return 0
    if adaptive_edge_budget_cfg is not None:
        max_tot = int(adaptive_edge_budget_cfg.max_total_edges_per_query)
    else:
        max_tot = int(getattr(args, "max_total_edges_per_query", 0))
    return max_tot // 2


def _use_real_edges(args, adaptive_edge_budget_cfg=None) -> bool:
    if _fixed_edge_budget_enabled(args, adaptive_edge_budget_cfg):
        return _get_real_edge_budget(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg) > 0
    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return _get_real_edge_budget(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg) > 0
    return bool(getattr(args, "include_real_edges", 0))


def _use_real_edges_for_state(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> bool:
    if _fixed_edge_budget_enabled(args, adaptive_edge_budget_cfg):
        return _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg) > 0
    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg) > 0
    return bool(getattr(args, "include_real_edges", 0))


def _use_rw_edges(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> bool:
    if int(getattr(args, "head_hop_walks_per_node", 0)) <= 0:
        return False
    if _fixed_edge_budget_enabled(args, adaptive_edge_budget_cfg):
        return _get_rw_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg) > 0
    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return _get_rw_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg) > 0
    return True


def _resolve_adaptive_edge_budget_config(args, valid_size: int) -> _AdaptiveEdgeBudgetConfig:
    enabled = _adaptive_edge_budget_enabled(args)

    probe_size = int(getattr(args, "adaptive_edge_budget_probe_size", 0))
    if enabled and probe_size <= 0:
        valid_n = max(0, int(valid_size))
        probe_size = min(512, valid_n) if valid_n > 0 else 0
    else:
        probe_size = max(0, probe_size)

    block_size = int(getattr(args, "adaptive_edge_budget_block_size", 0))
    if block_size <= 0:
        block_size = 2

    warmup_epochs_arg = int(getattr(args, "adaptive_edge_budget_warmup_epochs", 0))
    warmup_epochs = None if warmup_epochs_arg < 0 else warmup_epochs_arg

    patience = int(getattr(args, "adaptive_edge_budget_patience", 0))
    if patience <= 0:
        patience = 3
    patience = max(3, patience)

    bootstrap_search_epochs = int(getattr(args, "adaptive_edge_budget_bootstrap_search_epochs", 0))
    if enabled:
        if bootstrap_search_epochs == 0:
            bootstrap_search_epochs = 2
        elif bootstrap_search_epochs < 0:
            bootstrap_search_epochs = 0
    bootstrap_search_epochs = max(0, bootstrap_search_epochs)

    bootstrap_n_layers = int(getattr(args, "adaptive_edge_budget_bootstrap_n_layers", 0))
    bootstrap_n_layers = max(0, bootstrap_n_layers)

    static_seed_epochs = int(getattr(args, "adaptive_edge_budget_static_seed_epochs", 0))
    if enabled:
        if static_seed_epochs == 0:
            static_seed_epochs = bootstrap_search_epochs
        elif static_seed_epochs < 0:
            static_seed_epochs = 0
    static_seed_epochs = max(0, static_seed_epochs)

    return _AdaptiveEdgeBudgetConfig(
        enabled=enabled,
        probe_size=probe_size,
        block_size=max(1, block_size),
        warmup_epochs=warmup_epochs,
        patience=patience,
        gain_threshold=float(getattr(args, "adaptive_edge_budget_gain_threshold", 0.0)),
        max_total_edges_per_query=max(0, int(getattr(args, "max_total_edges_per_query", 0))),
        bootstrap_search_epochs=bootstrap_search_epochs,
        bootstrap_n_layers=bootstrap_n_layers,
        static_seed_epochs=static_seed_epochs,
    )


def _edge_block_seed(args, edge_seed, offset: int) -> int:
    base = getattr(args, "seed", 0) if edge_seed is None else int(edge_seed)
    return int(base) + int(offset)


def _sample_edges_per_query_random_sort(edge_index, max_edges_per_query: int, seed: int):
    max_edges_per_query = int(max_edges_per_query)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    if src.numel() <= max_edges_per_query:
        return edge_index

    scale = 1 << 31
    rand_key = torch.remainder(
        src * 1103515245 + dst * 214013 + int(seed) * 2654435761 + 12345,
        scale,
    )
    composite = dst * scale + rand_key
    perm = torch.argsort(composite)
    dst_sorted = dst[perm]

    arange = torch.arange(dst_sorted.numel(), device=dst_sorted.device, dtype=torch.long)
    is_new = torch.ones_like(dst_sorted, dtype=torch.bool)
    is_new[1:] = dst_sorted[1:] != dst_sorted[:-1]
    group_starts = torch.where(is_new, arange, torch.zeros_like(arange))
    group_starts = torch.cummax(group_starts, dim=0).values
    pos_in_group = arange - group_starts
    keep = pos_in_group < max_edges_per_query
    return edge_index[:, perm[keep]]


def _sample_edges_per_query_random_minhash(edge_index, max_edges_per_query: int, seed: int):
    max_edges_per_query = int(max_edges_per_query)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    if src.numel() <= max_edges_per_query:
        return edge_index
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        return _sample_edges_per_query_random_sort(edge_index, max_edges_per_query, seed)

    scale = 1 << 31
    rand_key = torch.remainder(
        src * 1103515245 + dst * 214013 + int(seed) * 2654435761 + 12345,
        scale,
    ).to(torch.long)
    num_groups = int(dst.max().item()) + 1
    if num_groups <= 0:
        return edge_index

    edge_count = int(dst.numel())
    active = torch.ones(edge_count, dtype=torch.bool, device=dst.device)
    keep_mask = torch.zeros(edge_count, dtype=torch.bool, device=dst.device)
    edge_pos = torch.arange(edge_count, dtype=torch.long, device=dst.device)
    key_fill = int(scale)
    pos_fill = int(edge_count)

    for _ in range(max_edges_per_query):
        if not bool(active.any().item()):
            break
        min_key = torch.full((num_groups,), key_fill, dtype=torch.long, device=dst.device)
        key_vals = rand_key.masked_fill(~active, key_fill)
        min_key.scatter_reduce_(0, dst, key_vals, reduce="amin", include_self=True)
        chosen = active & (rand_key == min_key.index_select(0, dst))
        if not bool(chosen.any().item()):
            break

        min_pos = torch.full((num_groups,), pos_fill, dtype=torch.long, device=dst.device)
        pos_vals = edge_pos.masked_fill(~chosen, pos_fill)
        min_pos.scatter_reduce_(0, dst, pos_vals, reduce="amin", include_self=True)
        chosen = chosen & (edge_pos == min_pos.index_select(0, dst))
        keep_mask |= chosen
        active &= ~chosen

    return edge_index[:, keep_mask]


def _sample_edges_per_query_random(edge_index, max_edges_per_query: int, seed: int):
    if edge_index is None or edge_index.numel() == 0 or int(max_edges_per_query) <= 0:
        return edge_index

    max_edges_per_query = int(max_edges_per_query)
    if max_edges_per_query <= _MINHASH_SAMPLE_K_LIMIT:
        return _sample_edges_per_query_random_minhash(edge_index, max_edges_per_query, seed)
    return _sample_edges_per_query_random_sort(edge_index, max_edges_per_query, seed)


def _sample_random_edge_blocks(args, real_edges, rw_edges, edge_seed=None, adaptive_edge_budget_cfg=None):
    if not _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return real_edges, rw_edges

    real_budget = _get_real_edge_budget(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg)
    rw_budget = _get_rw_edge_budget(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg)

    if real_edges is not None and real_budget > 0:
        real_edges = _sample_edges_per_query_random(
            real_edges,
            real_budget,
            _edge_block_seed(args, edge_seed, 17),
        )

    if rw_edges is not None and rw_budget > 0:
        rw_edges = _sample_edges_per_query_random(
            rw_edges,
            rw_budget,
            _edge_block_seed(args, edge_seed, 37),
        )

    return real_edges, rw_edges


def _sample_random_edge_blocks_with_state(
    args,
    real_edges,
    rw_edges,
    edge_seed=None,
    edge_budget_state=None,
    adaptive_edge_budget_cfg=None,
):
    if not _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return real_edges, rw_edges

    real_budget = _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)
    rw_budget = _get_rw_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)

    if real_edges is not None and real_budget > 0:
        real_edges = _sample_edges_per_query_random(
            real_edges,
            real_budget,
            _edge_block_seed(args, edge_seed, 17),
        )

    if rw_edges is not None and rw_budget > 0:
        rw_edges = _sample_edges_per_query_random(
            rw_edges,
            rw_budget,
            _edge_block_seed(args, edge_seed, 37),
        )

    return real_edges, rw_edges


def _resolve_real_edges_for_state(
    args,
    edge_index_global,
    device,
    edge_seed=None,
    edge_budget_state=None,
    adaptive_edge_budget_cfg=None,
):
    if not _use_real_edges_for_state(args, edge_budget_state, adaptive_edge_budget_cfg):
        return None

    real_edges = edge_index_global.to(device)
    if not _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return real_edges

    real_budget = _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)
    if real_budget <= 0:
        return None

    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        real_edges = profile_call(
            "edge_real_sample",
            lambda: _sample_edges_per_query_random(
                real_edges,
                real_budget,
                _edge_block_seed(args, edge_seed, 17),
            ),
            device=device,
        )
    return real_edges


def _resolve_rw_edges_for_state(
    args,
    rw_edges,
    edge_seed=None,
    edge_budget_state=None,
    adaptive_edge_budget_cfg=None,
):
    if rw_edges is None:
        return None
    if not _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return rw_edges

    rw_budget = _get_rw_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)
    if rw_budget <= 0:
        return None
    return profile_call(
        "edge_rw_sample",
        lambda: _sample_edges_per_query_random(
            rw_edges,
            rw_budget,
            _edge_block_seed(args, edge_seed, 37),
        ),
        device=rw_edges.device,
    )


def _filter_edge_index_by_dst(edge_index, dst_nodes):
    if edge_index is None:
        return None
    if dst_nodes is None:
        return edge_index
    if edge_index.numel() == 0:
        return edge_index
    dst_nodes = dst_nodes.to(device=edge_index.device, dtype=torch.long).view(-1)
    if dst_nodes.numel() == 0:
        return edge_index.new_zeros((edge_index.size(0), 0), dtype=edge_index.dtype)
    mask = torch.isin(edge_index[1].to(torch.long), dst_nodes)
    if not mask.any():
        return edge_index.new_zeros((edge_index.size(0), 0), dtype=edge_index.dtype)
    return edge_index[:, mask]


def _build_probe_edge_pools(
    args,
    edge_index_global,
    num_nodes,
    rw_device,
    probe_idx_global,
    edge_seed=None,
    adaptive_edge_budget_cfg=None,
    walk_length_override=None,
):
    probe_idx_global = probe_idx_global.to(dtype=torch.long, device="cpu").view(-1)
    seed_ctx = (
        _fixed_torch_cpu_seed(edge_seed)
        if edge_seed is not None and torch.device(rw_device).type == "cpu"
        else contextlib.nullcontext()
    )
    with seed_ctx:
        real_pool = None
        if _use_real_edges(args, adaptive_edge_budget_cfg):
            real_pool = _filter_edge_index_by_dst(edge_index_global.cpu(), probe_idx_global)
            if real_pool is not None:
                real_pool = real_pool.to(dtype=torch.long, device="cpu")

        rw_pool = None
        if _use_rw_edges(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg):
            wl = _resolve_walk_length(
                args,
                walk_length_override=walk_length_override,
            )
            rw_pool = build_head_hop_edges(
                edge_index=edge_index_global,
                num_nodes=num_nodes,
                num_heads=args.num_heads,
                num_groups=1,
                device=rw_device,
                walk_length=wl,
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            )
            if isinstance(rw_pool, list):
                rw_pool = _merge_edge_index_list(rw_pool)
            if rw_pool is not None:
                rw_pool = _filter_edge_index_by_dst(rw_pool, probe_idx_global)
                rw_pool = rw_pool.to(dtype=torch.long, device="cpu")

    return {
        "real": real_pool,
        "rw": rw_pool,
    }


def _assemble_edges_from_pools(
    args,
    edge_pools,
    num_nodes,
    nodes_per_rank,
    edge_seed=None,
    edge_budget_state=None,
    adaptive_edge_budget_cfg=None,
):
    parts = []
    real_edges = edge_pools.get("real")
    rw_edges = edge_pools.get("rw")
    real_edges, rw_edges = _sample_random_edge_blocks_with_state(
        args,
        real_edges,
        rw_edges,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
    )
    if real_edges is not None and real_edges.numel() > 0:
        parts.append(real_edges)
    if rw_edges is not None and rw_edges.numel() > 0:
        parts.append(rw_edges)

    merged = _merge_edge_index_list(parts)
    if merged is None:
        template = edge_pools.get("real")
        if template is None:
            template = edge_pools.get("rw")
        if template is None:
            merged = torch.zeros((2, 0), dtype=torch.long)
        else:
            merged = template.new_zeros((2, 0), dtype=torch.long)
    if args.model == "graphormer" and args.num_global_node > 0:
        merged = fix_edge_index(merged, num_nodes)
        merged = adjust_edge_index_nomerge(merged, nodes_per_rank)
    return merged.contiguous()


def _build_probe_attention_edges(
    args,
    edge_index_global,
    num_nodes,
    device,
    rw_device,
    sp_group,
    sp_src_rank,
    sp_rank,
    nodes_per_rank,
    probe_idx_global,
    edge_seed=None,
    edge_budget_state=None,
    adaptive_edge_budget_cfg=None,
    edge_pools=None,
    walk_length_override=None,
):
    if sp_rank == sp_src_rank:
        pools = edge_pools
        if pools is None:
            resolved_walk_length = _resolve_walk_length(
                args,
                edge_budget_state=edge_budget_state,
                walk_length_override=walk_length_override,
            )
            pools = _build_probe_edge_pools(
                args,
                edge_index_global,
                num_nodes,
                rw_device,
                probe_idx_global,
                edge_seed=edge_seed,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                walk_length_override=resolved_walk_length,
            )
        merged_cpu = _assemble_edges_from_pools(
            args,
            pools,
            num_nodes,
            nodes_per_rank,
            edge_seed=edge_seed,
            edge_budget_state=edge_budget_state,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        )
    else:
        merged_cpu = None

    return _broadcast_prefetched_edges(merged_cpu, device, sp_group, sp_src_rank, sp_rank)



class _AdaptiveEdgeBudgetController:
    def __init__(self, config: _AdaptiveEdgeBudgetConfig) -> None:
        self.enabled = config.enabled
        self.block_size = int(config.block_size)
        self.max_total = int(config.max_total_edges_per_query)
        self.warmup_epochs = None if config.warmup_epochs is None else int(config.warmup_epochs)
        self.gain_threshold = float(config.gain_threshold)
        self.patience = int(config.patience)
        self.bad_rounds = 0
        self.seen_positive_gain = False
        
        self.auto_hold_released = False
        self.baseline_acc = -1.0
        self.bootstrap_baseline_loss = float("inf")
        self.bootstrap_baseline_acc = -1.0
        
        self.frozen = not self.enabled
        if not self.enabled:
            self.real_budget = self.max_total
            self.rw_budget = 0
            self.walk_length = None
            return

        self.real_budget = 1 if self.max_total > 0 else 0
        self.rw_budget = 1 if self.max_total > 0 else 0
        self.walk_length = None

    def current_state(self):
        state = {
            "real_edges_per_query": int(self.real_budget),
            "rw_edges_per_query": int(self.rw_budget),
        }
        if self.walk_length is not None:
            state["walk_length"] = int(self.walk_length)
        return state

    def candidate_states(self):
        out = {}
        if self.real_budget + self.rw_budget + self.block_size <= self.max_total:
            real_up = {
                "real_edges_per_query": self.real_budget + self.block_size,
                "rw_edges_per_query": self.rw_budget,
            }
            rw_up = {
                "real_edges_per_query": self.real_budget,
                "rw_edges_per_query": self.rw_budget + self.block_size,
            }
            if self.walk_length is not None:
                real_up["walk_length"] = int(self.walk_length)
                rw_up["walk_length"] = int(self.walk_length)
            out["real_up"] = real_up
            out["rw_up"] = rw_up
        return out

    def set_state(self, state) -> None:
        if state is None:
            return
        self.real_budget = int(state.get("real_edges_per_query", self.real_budget))
        self.rw_budget = int(state.get("rw_edges_per_query", self.rw_budget))
        if "walk_length" in state:
            self.walk_length = int(state["walk_length"])

    def update(self, choice, best_gain, next_state=None):
        if best_gain > 0.0:
            self.seen_positive_gain = True
        if next_state is not None and choice is not None:
            self.real_budget = int(next_state["real_edges_per_query"])
            self.rw_budget = int(next_state["rw_edges_per_query"])
            if "walk_length" in next_state:
                self.walk_length = int(next_state["walk_length"])
            self.bad_rounds = 0
            return
        if not self.seen_positive_gain:
            return
        if best_gain <= self.gain_threshold:
            self.bad_rounds += 1
            if self.bad_rounds >= self.patience:
                self.frozen = True


def _round_budget_to_block(value: int, block_size: int) -> int:
    if value <= 0:
        return 0
    block_size = max(1, int(block_size))
    return max(block_size, ((int(value) + block_size - 1) // block_size) * block_size)


def _floor_budget_to_block(value: int, block_size: int) -> int:
    if value <= 0:
        return 0
    block_size = max(1, int(block_size))
    return max(block_size, (int(value) // block_size) * block_size)


def _auto_initial_budget_candidates(edge_index_global, num_nodes: int, config: _AdaptiveEdgeBudgetConfig):
    max_total = int(config.max_total_edges_per_query)
    if max_total <= 0:
        return []

    block_size = max(1, int(config.block_size))

    # Generate (real, rw) pairs anchored to block_size multiples.
    # For each real scale, vary rw from 0 up to rw-heavy; larger real scales
    # use proportionally more rw rather than a fixed 1x cap.
    raw_pairs = [
        (1 * block_size, 0),
        (1 * block_size, 1 * block_size),
        (1 * block_size, 2 * block_size),
        (2 * block_size, 0),
        (2 * block_size, 1 * block_size),
        (2 * block_size, 2 * block_size),
        (4 * block_size, 0),
        (4 * block_size, 2 * block_size),
    ]

    seen = set()
    candidates = []
    for real, rw in raw_pairs:
        if real + rw > max_total:
            continue
        key = (real, rw)
        if key not in seen:
            seen.add(key)
            candidates.append({"real_edges_per_query": real, "rw_edges_per_query": rw})

    if not candidates:
        candidates = [{"real_edges_per_query": min(max_total, block_size), "rw_edges_per_query": 0}]

    candidates.sort(
        key=lambda state: (
            state["real_edges_per_query"] + state["rw_edges_per_query"],
            state["rw_edges_per_query"],
        )
    )

    return candidates


def _clone_model_state_to_cpu(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


@contextlib.contextmanager
def _temporary_bootstrap_layer_subset(
    args,
    config: _AdaptiveEdgeBudgetConfig,
    model,
    sp_group,
    sp_world_size,
    grad_reducer,
):
    target_n_layers = int(config.bootstrap_n_layers)
    model_layers = getattr(model, "layers", None)
    if target_n_layers <= 0 or model_layers is None:
        yield model, grad_reducer, None
        return

    full_n_layers = len(model_layers)
    if target_n_layers >= full_n_layers:
        yield model, grad_reducer, None
        return

    original_layers = model.layers
    original_n_layers = getattr(model, "n_layers", None)
    original_ckpt_mode = getattr(model, "activation_checkpoint_mode", None)

    try:
        model.layers = torch.nn.ModuleList(list(original_layers[:target_n_layers]))
        if original_n_layers is not None:
            model.n_layers = target_n_layers
        if hasattr(model, "set_activation_checkpoint"):
            model.set_activation_checkpoint(mode=getattr(args, "activation_checkpoint_mode", None))
        bootstrap_grad_reducer = build_gradient_reducer(
            model,
            process_group=sp_group,
            world_size=sp_world_size,
        )
        yield model, bootstrap_grad_reducer, full_n_layers
    finally:
        model.layers = original_layers
        if original_n_layers is not None:
            model.n_layers = original_n_layers
        if hasattr(model, "set_activation_checkpoint"):
            model.set_activation_checkpoint(mode=original_ckpt_mode)


def _select_probe_nodes(split_idx, probe_size: int, seed: int):
    probe_size = max(0, int(probe_size))
    if probe_size <= 0:
        return None
    valid_idx = split_idx.get("valid")
    if valid_idx is None or valid_idx.numel() == 0:
        return None
    valid_idx = valid_idx.to(torch.long).cpu()
    if valid_idx.numel() <= probe_size:
        return valid_idx
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    perm = torch.randperm(valid_idx.numel(), generator=gen)[:probe_size]
    return valid_idx.index_select(0, perm).to(torch.long)


def _edge_count(attn_edges) -> int:
    if isinstance(attn_edges, dict):
        return int(attn_edges["src"].numel())
    return int(attn_edges.size(1))


def _probe_loss_sp(args, model, x_local, local_y, local_probe_idx, edge_index_probe,
                   device, amp_dtype, sp_group, sp_world_size):
    was_training = model.training
    model.train()
    drop_states = _set_dropout_eval(model)

    loss_sum = torch.zeros(1, device=device, dtype=torch.float32)
    acc_sum = torch.zeros(1, device=device, dtype=torch.float32)
    count = torch.zeros(1, device=device, dtype=torch.long)
    with torch.no_grad():
        with _autocast_context(device, amp_dtype):
            out_local = model(x_local, None, edge_index_probe, attn_type=args.attn_type)
        out_rows = int(out_local.size(0))
        if local_probe_idx is not None and local_probe_idx.numel() > 0:
            valid_probe = (local_probe_idx >= 0) & (local_probe_idx < out_rows)
            probe_idx_eff = local_probe_idx[valid_probe]
            if probe_idx_eff.numel() > 0:
                local_y_eff = local_y[:out_rows]
                logits = out_local.index_select(0, probe_idx_eff)
                targets = local_y_eff.index_select(0, probe_idx_eff).long()
                loss_sum = F.nll_loss(
                    logits,
                    targets,
                    reduction="sum",
                ).to(torch.float32).view(1)
                preds = logits.argmax(dim=-1)
                acc_sum = (preds == targets).sum().to(torch.float32).view(1)
                count = torch.tensor([probe_idx_eff.numel()], device=device, dtype=torch.long)

    if sp_world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=sp_group)

    _restore_dropout(drop_states)
    if not was_training:
        model.eval()

    mean_loss = float(loss_sum.item() / max(int(count.item()), 1))
    mean_acc = float(acc_sum.item() / max(int(count.item()), 1))
    del out_local
    _cuda_empty_cache(args)
    return mean_loss, mean_acc


def _build_optimizer_bundle(args, model, device, amp_dtype):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup=args.warmup_updates,
        tot=args.epochs,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(amp_dtype == torch.float16 and device.startswith("cuda"))
    )
    return optimizer, lr_scheduler, scaler


def _train_one_epoch_with_budget(
    args,
    adaptive_edge_budget_cfg,
    model,
    optimizer,
    lr_scheduler,
    scaler,
    x_local,
    local_y,
    local_train_idx,
    edge_index_global,
    num_nodes,
    device,
    rw_device,
    sp_group,
    sp_src_rank,
    sp_rank,
    local_nodes,
    amp_dtype,
    sp_world_size,
    grad_reducer,
    budget_state,
    edge_seed,
    walk_length_override=None,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    edge_index = _build_attention_edges(
        args,
        edge_index_global,
        num_nodes,
        device,
        rw_device,
        sp_group,
        sp_src_rank,
        sp_rank,
        local_nodes,
        edge_seed=edge_seed,
        edge_budget_state=budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        walk_length_override=walk_length_override,
    )

    with _autocast_context(device, amp_dtype):
        out_local = model(x_local, None, edge_index, attn_type=args.attn_type)
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

    grad_reducer.prepare_backward()
    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    grad_reducer.finalize_backward()

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    lr_scheduler.step()

    del edge_index, out_local, loss, local_y_eff, valid_train_mask, local_train_idx_eff
    _cuda_empty_cache(args)


def _bootstrap_initial_edge_budget(
    args,
    adaptive_edge_budget_cfg,
    controller,
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
    local_nodes,
    amp_dtype,
    sp_world_size,
    grad_reducer,
):
    if not controller.enabled:
        return None
    if (
        not adaptive_edge_budget_cfg.bootstrap_search_epochs
        or adaptive_edge_budget_cfg.bootstrap_search_epochs <= 0
    ):
        return None
    if probe_idx_global is None or local_probe_idx is None or probe_idx_global.numel() == 0:
        return None

    full_initial_model_state = _clone_model_state_to_cpu(model)
    try:
        with _temporary_bootstrap_layer_subset(
            args,
            adaptive_edge_budget_cfg,
            model,
            sp_group,
            sp_world_size,
            grad_reducer,
        ) as (bootstrap_model, bootstrap_grad_reducer, full_n_layers):
            if args.rank == 0 and full_n_layers is not None:
                print(
                    "  ↳ BootstrapBudgetSearch "
                    f"temporary_model_layers={len(bootstrap_model.layers)}/{full_n_layers}"
                )

            candidate_states = _auto_initial_budget_candidates(edge_index_global, num_nodes, adaptive_edge_budget_cfg)
            if not candidate_states:
                return None

            initial_model_state = _clone_model_state_to_cpu(bootstrap_model)
            probe_seed = int(getattr(args, "seed", 0))
            probe_edge_pools = None
            if sp_rank == sp_src_rank:
                probe_edge_pools = _build_probe_edge_pools(
                    args,
                    edge_index_global,
                    num_nodes,
                    rw_device,
                    probe_idx_global,
                    edge_seed=probe_seed,
                    adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                )

            candidate_specs = []
            for candidate_state in candidate_states:
                candidate_specs.append(
                    {
                        "budget_state": dict(candidate_state),
                        "full_state": dict(candidate_state),
                        "probe_edge_pools": probe_edge_pools,
                        "model_state": initial_model_state,
                        "trained_epochs": 0,
                        "loss": float("inf"),
                        "edge_count": float("inf"),
                    }
                )

            n_epochs = max(1, int(adaptive_edge_budget_cfg.bootstrap_search_epochs))

            def _fmt_summary(state, loss, trained_epochs):
                return f"({state['real_edges_per_query']},{state['rw_edges_per_query']})@{loss:.4f}[e={trained_epochs}]"

            # Flat search: train every candidate for the same n_epochs, then pick by loss.
            all_summaries = []
            for spec in candidate_specs:
                _set_seed(int(getattr(args, "seed", 0)))
                bootstrap_model.load_state_dict(initial_model_state, strict=True)
                optimizer, lr_scheduler, scaler = _build_optimizer_bundle(args, bootstrap_model, device, amp_dtype)

                for _ in range(n_epochs):
                    _train_one_epoch_with_budget(
                        args,
                        adaptive_edge_budget_cfg,
                        bootstrap_model,
                        optimizer,
                        lr_scheduler,
                        scaler,
                        x_local,
                        local_y,
                        local_train_idx,
                        edge_index_global,
                        num_nodes,
                        device,
                        rw_device,
                        sp_group,
                        sp_src_rank,
                        sp_rank,
                        local_nodes,
                        amp_dtype,
                        sp_world_size,
                        bootstrap_grad_reducer,
                        spec["budget_state"],
                        probe_seed,
                    )

                probe_edges = _build_probe_attention_edges(
                    args,
                    edge_index_global,
                    num_nodes,
                    device,
                    rw_device,
                    sp_group,
                    sp_src_rank,
                    sp_rank,
                    local_nodes,
                    probe_idx_global,
                    edge_seed=probe_seed,
                    edge_budget_state=spec["budget_state"],
                    adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                    edge_pools=spec["probe_edge_pools"],
                )
                candidate_loss, _ = _probe_loss_sp(
                    args,
                    bootstrap_model,
                    x_local,
                    local_y,
                    local_probe_idx,
                    probe_edges,
                    device,
                    amp_dtype,
                    sp_group,
                    sp_world_size,
                )
                candidate_edge_count = _edge_count(probe_edges)
                spec["loss"] = float(candidate_loss)
                spec["edge_count"] = int(candidate_edge_count)
                spec["trained_epochs"] = n_epochs
                all_summaries.append((spec, float(candidate_loss), int(candidate_edge_count)))
                del probe_edges, optimizer, lr_scheduler, scaler

            if not all_summaries:
                return None

            all_summaries.sort(key=lambda item: (float(item[1]), int(item[2])))
            best_spec, best_loss, best_edge_count = all_summaries[0]
            best_state = dict(best_spec["full_state"])

            bootstrap_model.load_state_dict(initial_model_state, strict=True)
            _set_seed(int(getattr(args, "seed", 0)))

            if args.rank == 0:
                formatted_candidates = ", ".join(
                    _fmt_summary(spec["full_state"], spec["loss"], spec["trained_epochs"])
                    for spec, _, _ in all_summaries
                )
                print(
                    "  ↳ BootstrapBudgetSearch "
                    f"strategy=flat "
                    f"epochs={n_epochs} "
                    f"walk_length={_resolve_walk_length(args)} "
                    f"candidates=[{formatted_candidates}] "
                    f"chosen=({best_state['real_edges_per_query']},{best_state['rw_edges_per_query']}) "
                    f"probe_loss={best_loss:.4f} probe_edges={best_edge_count}"
                )

            if args.rank == 0:
                print(
                    "  ↳ BudgetCtrl [ThresholdFixed] "
                    f"rel_gain_threshold remains {controller.gain_threshold:.8e} from args."
                )

            controller.bootstrap_baseline_loss = float(best_loss)
            return best_state
    finally:
        model.load_state_dict(full_initial_model_state, strict=True)


def _maybe_update_edge_budget(
    args,
    adaptive_edge_budget_cfg,
    controller,
    epoch: int,
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
    local_nodes,
    amp_dtype,
    sp_world_size,
):
    if (
        (not controller.enabled)
        or controller.frozen
        or epoch <= int(adaptive_edge_budget_cfg.bootstrap_search_epochs)
        or (controller.warmup_epochs is not None and epoch > controller.warmup_epochs)
    ):
        return
        
    if local_probe_idx is None or probe_idx_global is None or probe_idx_global.numel() == 0:
        controller.frozen = True
        return

    probe_seed = int(getattr(args, "seed", 0)) + 100000 + int(epoch)
    base_state = controller.current_state()
    probe_edge_pools = None
    if sp_rank == sp_src_rank:
        probe_edge_pools = _build_probe_edge_pools(
            args,
            edge_index_global,
            num_nodes,
            rw_device,
            probe_idx_global,
            edge_seed=probe_seed,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            walk_length_override=base_state.get("walk_length"),
        )
    base_edges = _build_probe_attention_edges(
        args,
        edge_index_global,
        num_nodes,
        device,
        rw_device,
        sp_group,
        sp_src_rank,
        sp_rank,
        local_nodes,
        probe_idx_global,
        edge_seed=probe_seed,
        edge_budget_state=base_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        edge_pools=probe_edge_pools,
    )
    # 1. Base Evaluation
    base_loss, base_acc = _probe_loss_sp(
        args, model, x_local, local_y, local_probe_idx, base_edges,
        device, amp_dtype, sp_group, sp_world_size,
    )
    del base_edges

    # 2. Candidate Evaluation
    best_kind = None
    best_gain = float("-inf")
    best_state = None
    best_loss = base_loss
    best_count = 0
    
    for kind, cand_state in controller.candidate_states().items():
        cand_edges = _build_probe_attention_edges(
            args,
            edge_index_global,
            num_nodes,
            device,
            rw_device,
            sp_group,
            sp_src_rank,
            sp_rank,
            local_nodes,
            probe_idx_global,
            edge_seed=probe_seed,
            edge_budget_state=cand_state,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            edge_pools=probe_edge_pools,
        )
        cand_loss, cand_acc = _probe_loss_sp(
            args, model, x_local, local_y, local_probe_idx, cand_edges,
            device, amp_dtype, sp_group, sp_world_size,
        )
        cand_count = _edge_count(cand_edges)
        
        # Relative improvement against the current budget; no edge-count normalization.
        gain = (base_loss - cand_loss) / max(base_loss, 1e-6)
        
        if gain > best_gain:
            best_kind = kind
            best_gain = gain
            best_state = cand_state
            best_loss = cand_loss
            best_count = cand_count
            
        del cand_edges

    # 3. Check Condition for Release or Continued Update
    if not controller.auto_hold_released:
        # Release trigger: Only if we find a significant relative gain
        if best_gain > controller.gain_threshold:
            controller.auto_hold_released = True
            if args.rank == 0:
                print(
                    "  ↳ BudgetCtrl [AutoHoldReleased] via significant rel_gain: "
                    f"{best_gain:.8e} > threshold={controller.gain_threshold:.8e}"
                )
        else:
            if args.rank == 0:
                print(
                    f"  ↳ BudgetCtrl [AutoHold] epoch={epoch} searching... "
                    f"best_rel_gain_found={best_gain:.8e} (threshold={controller.gain_threshold:.8e})"
                )
            return

    # 4. Standard Update (if released)
    controller.update(best_kind if best_gain > controller.gain_threshold else None,
                      best_gain, best_state if best_gain > controller.gain_threshold else None)

    if args.rank == 0:
        actual_move = best_kind if best_gain > controller.gain_threshold else "STAY"
        cur_budget = controller.current_state()
        print(
            f"  ↳ BudgetCtrl update epoch={epoch} move={actual_move} rel_gain={best_gain:.8e} "
            f"new_budget=({cur_budget['real_edges_per_query']},{cur_budget['rw_edges_per_query']}) "
            f"probe_loss={best_loss:.4f} probe_edges={best_count}"
        )


@contextlib.contextmanager
def _fixed_torch_cpu_seed(seed: int):
    torch_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(torch_state)


class _CpuRandomWalkEdgePrefetcher:
    def __init__(self, enabled: bool) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1) if enabled else None
        self._future = None
        self._epoch = None

    @property
    def enabled(self) -> bool:
        return self._executor is not None

    def submit(self, epoch: int, fn) -> None:
        if self._executor is None or epoch <= 0 or self._future is not None:
            return
        self._epoch = int(epoch)
        self._future = self._executor.submit(fn)

    def pop(self, epoch: int):
        if self._future is None or self._epoch != int(epoch):
            return None
        result = self._future.result()
        self._future = None
        self._epoch = None
        return result

    def close(self) -> None:
        if self._future is not None:
            self._future.result()
            self._future = None
            self._epoch = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def _build_merged_edges(args, edge_index_global, num_nodes, final_device, rw_device, nodes_per_rank,
                        edge_seed=None, edge_budget_state=None, adaptive_edge_budget_cfg=None,
                        walk_length_override=None):
    parts = []
    final_torch_device = torch.device(final_device)
    real_edges = _resolve_real_edges_for_state(
        args,
        edge_index_global,
        final_torch_device,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
    )
    record_scalar("edge_real_edges", int(real_edges.size(1)) if real_edges is not None else 0, reduce="max")
    rw_heads = None
    if _use_rw_edges(args, edge_budget_state, adaptive_edge_budget_cfg):
        wl = _resolve_walk_length(
            args,
            edge_budget_state=edge_budget_state,
            walk_length_override=walk_length_override,
        )
        rw_heads = profile_call(
            "edge_rw_build",
            lambda: build_head_hop_edges(
                edge_index=edge_index_global,
                num_nodes=num_nodes,
                num_heads=args.num_heads,
                num_groups=1,
                device=rw_device,
                walk_length=wl,
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            ),
            device=rw_device,
        )
        if isinstance(rw_heads, list):
            rw_heads = _merge_edge_index_list(rw_heads)
        if rw_heads is not None and rw_heads.device != final_torch_device:
            rw_heads = rw_heads.to(final_torch_device)
    rw_heads = _resolve_rw_edges_for_state(
        args,
        rw_heads,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
    )
    record_scalar("edge_rw_edges", int(rw_heads.size(1)) if rw_heads is not None else 0, reduce="max")
    if real_edges is not None:
        parts.append(real_edges)
    if rw_heads is not None:
        parts.append(rw_heads)
    def _assemble():
        merged_local = _merge_edge_index_list(parts)
        if merged_local is None:
            merged_local = edge_index_global.new_zeros((2, 0), dtype=torch.long).to(final_torch_device)
        if args.model == "graphormer" and args.num_global_node > 0:
            merged_local = fix_edge_index(merged_local, num_nodes)
            merged_local = adjust_edge_index_nomerge(merged_local, nodes_per_rank)
        return merged_local

    merged = profile_call("edge_merge", _assemble, device=final_torch_device)
    record_scalar("edge_merged_edges", int(merged.size(1)), reduce="max")
    return merged


def _resolve_walk_length(args, edge_budget_state=None, walk_length_override=None) -> int:
    if walk_length_override is not None:
        return max(1, int(walk_length_override))
    if edge_budget_state is not None and "walk_length" in edge_budget_state:
        return max(1, int(edge_budget_state["walk_length"]))
    fixed_walk_length = getattr(args, "fixed_walk_length", None)
    if fixed_walk_length is not None:
        return max(1, int(fixed_walk_length))
    return max(1, int(getattr(args, "head_hop_walk_length", 4)))


def _pack_streaming_edges(merged_cpu):
    if merged_cpu is None:
        merged_cpu = torch.zeros((2, 0), dtype=torch.long)
    edge_cpu = merged_cpu[:2].to(device="cpu", dtype=torch.long).contiguous()
    index_dtype = torch.int64
    if edge_cpu.numel() == 0:
        index_dtype = torch.int32
    else:
        max_index = int(edge_cpu.max().item())
        if max_index <= torch.iinfo(torch.int32).max:
            index_dtype = torch.int32

    src_cpu = edge_cpu[0].to(dtype=index_dtype).contiguous()
    dst_cpu = edge_cpu[1].to(dtype=index_dtype).contiguous()
    num_edges = int(src_cpu.numel())
    if num_edges <= 0:
        offsets = torch.zeros(1, dtype=torch.long)
    else:
        offsets = torch.arange(0, num_edges, _STREAM_EDGE_CHUNK_EDGES, dtype=torch.long)
        if int(offsets[-1].item()) != num_edges:
            offsets = torch.cat([offsets, offsets.new_tensor([num_edges])], dim=0)
        else:
            offsets = offsets.clone()

    return {
        "mode": "cpu_stream",
        "src": _pin_cpu_tensor(src_cpu),
        "dst": _pin_cpu_tensor(dst_cpu),
        "offsets": _pin_cpu_tensor(offsets.contiguous()),
    }


def _build_streaming_edges(args, edge_index_global, num_nodes, rw_device, nodes_per_rank,
                           edge_seed=None, edge_budget_state=None, adaptive_edge_budget_cfg=None,
                           walk_length_override=None):
    seed_ctx = fixed_random_seed(edge_seed) if edge_seed is not None else contextlib.nullcontext()
    with seed_ctx:
        merged_cpu = profile_call(
            "edge_build_local",
            lambda: _build_merged_edges(
                args,
                edge_index_global,
                num_nodes,
                "cpu",
                rw_device,
                nodes_per_rank,
                edge_seed=edge_seed,
                edge_budget_state=edge_budget_state,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                walk_length_override=walk_length_override,
            ),
            device=None,
        )
    return _pack_streaming_edges(merged_cpu)

def _build_prefetched_cpu_edges(args, edge_index_global, num_nodes, rw_device, nodes_per_rank, edge_seed,
                                edge_budget_state=None, adaptive_edge_budget_cfg=None):
    if getattr(args, "stream_edges_from_cpu", False):
        return _build_streaming_edges(
            args,
            edge_index_global,
            num_nodes,
            rw_device,
            nodes_per_rank,
            edge_seed=edge_seed,
            edge_budget_state=edge_budget_state,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        )
    seed_ctx = _fixed_torch_cpu_seed(edge_seed) if edge_seed is not None else contextlib.nullcontext()
    with seed_ctx:
        merged_cpu = _build_merged_edges(
            args, edge_index_global, num_nodes, "cpu", rw_device, nodes_per_rank,
            edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
    )
    if merged_cpu.device.type == "cpu":
        merged_cpu = _pin_cpu_tensor(merged_cpu.contiguous())
    return merged_cpu


def _broadcast_prefetched_edges(merged_cpu, device, sp_group, sp_src_rank, sp_rank):
    if sp_rank == sp_src_rank:
        merged = merged_cpu.to(device=device, dtype=torch.long, non_blocking=(merged_cpu.device.type == "cpu"))
        size_t = torch.tensor([merged.shape[1]], device=device, dtype=torch.long)
    else:
        size_t = torch.empty(1, device=device, dtype=torch.long)
        merged = None

    if dist.is_initialized():
        def _broadcast():
            nonlocal merged
            dist.broadcast(size_t, sp_src_rank, group=sp_group)
            if sp_rank != sp_src_rank:
                merged = torch.empty((2, int(size_t.item())), device=device, dtype=torch.long)
            dist.broadcast(merged, sp_src_rank, group=sp_group)

        payload_bytes = int(merged_cpu.numel() * merged_cpu.element_size()) if merged_cpu is not None else 0
        profile_call("edge_broadcast", _broadcast, device=device, payload_bytes=payload_bytes)
    return merged


def _build_and_broadcast_edges(args, edge_index_global, num_nodes, device, rw_device, sp_group,
                               sp_src_rank, sp_rank, nodes_per_rank, edge_seed=None,
                               edge_budget_state=None, adaptive_edge_budget_cfg=None,
                               walk_length_override=None):
    # ---------------------------------------------------------------------------
    # O3 Optimization: Rank-Local Edge Construction (eliminates edge broadcast).
    #
    # When edge_seed is fixed, every SP rank produces an identical edge_index by
    # running the same deterministic random walk under the same seed via
    # fixed_random_seed(), which resets both CPU and CUDA RNG states.
    # torch_cluster.random_walk is an integer-indexed operation with no
    # floating-point reductions, so it is bit-identical across ranks given the
    # same seed and graph topology.
    #
    # This removes the dist.broadcast() call from the hot path, eliminating
    # O(|E_sampled|) inter-rank communication per training step.
    #
    # Fallback: when edge_seed is None (un-seeded dynamic edges) the original
    # broadcast path is preserved to keep cross-rank consistency.
    # ---------------------------------------------------------------------------
    if edge_seed is not None and not bool(getattr(args, "force_edge_broadcast", False)):
        # Rank-local path: all ranks independently build the same edge_index.
        # fixed_random_seed handles both CPU and CUDA RNG (covers gpu rw_device).
        with fixed_random_seed(edge_seed):
            merged = profile_call(
                "edge_build_local",
                lambda: _build_merged_edges(
                    args,
                    edge_index_global,
                    num_nodes,
                    device,
                    rw_device,
                    nodes_per_rank,
                    edge_seed=edge_seed,
                    edge_budget_state=edge_budget_state,
                    adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                    walk_length_override=walk_length_override,
                ),
                device=device,
            )
        # Record a zero-byte entry so the comm profiler stats remain consistent.
        profile_call("edge_broadcast", lambda: None, device=device, payload_bytes=0)
        return merged

    # ---------------------------------------------------------------------------
    # Fallback: original broadcast path when edge_seed is None.
    # ---------------------------------------------------------------------------
    if sp_rank == sp_src_rank:
        merged = profile_call(
            "edge_build_local",
            lambda: _build_merged_edges(
                args,
                edge_index_global,
                num_nodes,
                device,
                rw_device,
                nodes_per_rank,
                edge_seed=edge_seed,
                edge_budget_state=edge_budget_state,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                walk_length_override=walk_length_override,
            ),
            device=device,
        )
        size_t = torch.tensor([merged.shape[1]], device=device, dtype=torch.long)
    else:
        size_t = torch.empty(1, device=device, dtype=torch.long)
        merged = None

    if dist.is_initialized():
        def _broadcast():
            nonlocal merged
            dist.broadcast(size_t, sp_src_rank, group=sp_group)
            if sp_rank != sp_src_rank:
                merged = torch.empty((2, int(size_t.item())), device=device, dtype=torch.long)
            dist.broadcast(merged, sp_src_rank, group=sp_group)

        payload_bytes = int(merged.numel() * merged.element_size()) if merged is not None else 0
        profile_call("edge_broadcast", _broadcast, device=device, payload_bytes=payload_bytes)

    return merged


def _build_attention_edges(args, edge_index_global, num_nodes, device, rw_device, sp_group,
                           sp_src_rank, sp_rank, nodes_per_rank, edge_seed=None,
                           edge_budget_state=None, adaptive_edge_budget_cfg=None,
                           walk_length_override=None):
    if getattr(args, "stream_edges_from_cpu", False):
        return _build_streaming_edges(
            args,
            edge_index_global,
            num_nodes,
            rw_device,
            nodes_per_rank,
            edge_seed=edge_seed,
            edge_budget_state=edge_budget_state,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            walk_length_override=walk_length_override,
        )
    return _build_and_broadcast_edges(
        args,
        edge_index_global,
        num_nodes,
        device,
        rw_device,
        sp_group,
        sp_src_rank,
        sp_rank,
        nodes_per_rank,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        walk_length_override=walk_length_override,
    )


def _set_dropout_eval(model):
    states = []
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            states.append((module, module.training))
            module.eval()
    return states


def _restore_dropout(states):
    for module, training_state in states:
        module.train(training_state)


def _eval_sp(args, model, x_local, y, split_idx, edge_index_global, num_nodes,
             device, rw_device, sp_group, sp_src_rank, sp_rank, sp_world_size,
             rank_start, rank_end, local_nodes, amp_dtype=None, cached_edge_index=None,
             edge_budget_state=None, adaptive_edge_budget_cfg=None):
    if cached_edge_index is None:
        with fixed_random_seed(args.seed):
            edge_index_eval = _build_attention_edges(
                args, edge_index_global, num_nodes, device, rw_device,
                sp_group, sp_src_rank, sp_rank, local_nodes,
                edge_budget_state=edge_budget_state,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            )
    else:
        edge_index_eval = cached_edge_index

    was_training = model.training
    model.train()
    drop_states = _set_dropout_eval(model)
    with torch.no_grad():
        with _autocast_context(device, amp_dtype):
            out_local = model(x_local, None, edge_index_eval, attn_type=args.attn_type)
        pred_local = out_local.argmax(dim=1)

    _restore_dropout(drop_states)
    if not was_training:
        model.eval()

    if sp_world_size > 1:
        local_pred_len = torch.tensor([pred_local.size(0)], dtype=torch.long, device=device)
        len_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(sp_world_size)]
        dist.all_gather(len_list, local_pred_len, group=sp_group)
        pred_lens = [int(t.item()) for t in len_list]
        max_pred_len = max(pred_lens) if pred_lens else 0

        if pred_local.size(0) < max_pred_len:
            padded = torch.zeros(max_pred_len, dtype=torch.long, device=device)
            padded[:pred_local.size(0)] = pred_local
            pred_local = padded

        gather_list = [torch.zeros(max_pred_len, dtype=torch.long, device=device) for _ in range(sp_world_size)]
        dist.all_gather(gather_list, pred_local, group=sp_group)
        pred_chunks = [gather_list[i][:pred_lens[i]] for i in range(sp_world_size)]
        pred_global = torch.cat(pred_chunks, dim=0)[:num_nodes].cpu()
    else:
        pred_global = pred_local.cpu()

    result = None
    if args.rank == 0:
        if y is None or split_idx is None:
            raise RuntimeError("Rank 0 requires full labels and split_idx for evaluation.")
        valid_n = min(int(pred_global.size(0)), int(num_nodes))
        pred_global = pred_global[:valid_n]
        y_cpu = y[:valid_n].cpu().view(-1)
        accs = {}
        for split_name, idx in split_idx.items():
            idx_valid = idx[idx < valid_n]
            correct = (pred_global[idx_valid] == y_cpu[idx_valid]).sum().item()
            accs[split_name] = float(correct) / max(1, len(idx_valid))
        result = accs

    del pred_local, pred_global, out_local, x_local
    if cached_edge_index is None:
        del edge_index_eval
    _cuda_empty_cache(args)
    return result
