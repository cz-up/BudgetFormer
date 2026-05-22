import contextlib
import gc
import os
import random
import resource
import sys
import time
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
    fixed_random_seed_cpu,
    fix_edge_index,
    fixed_random_seed,
    random_split_idx,
)
from models.exphormer_dist_node_level import Exphormer
from models.graphgps_dist_node_level import GraphGPS
from models.graphormer_dist_node_level import Graphormer
from models.gt_dist_node_level import GT
from utils.lr import CosineWithWarmupLR, PolynomialDecayLR
from utils.split_utils import load_default_split

_MINHASH_SAMPLE_K_LIMIT = 8

# Deterministic merged-edge cache.  When edge_seed is fixed across calls (e.g.
# during static-seed epochs or bootstrap), the full walk+merge computation
# produces an identical CPU tensor every time.  We cache it here to avoid
# re-running the expensive build; the value is always stored on CPU so that
# GPU-resident entries from the rank-local path don't pin HBM permanently.
# Size is bounded (FIFO, capacity=8) to prevent unbounded growth when seeds
# change every epoch after the static-seed phase.
_MERGED_EDGE_CACHE: dict = {}
# CPU real-edge sampling repeatedly touches the full graph with only the
# sampling budget and seed changing.  Cache per-dst counts so we can skip
# minhash work for low-degree query nodes whose full in-edge set is already
# within budget.
_EDGE_DST_STATS_CACHE: dict = {}

_GRAPHORMER_VARIANTS = {
    "graphormer": Graphormer,
}
_GRAPHORMER_VIRTUAL_NODE_MODELS = frozenset(_GRAPHORMER_VARIANTS)


def _compute_laplacian_pe(row_aug, col_aug, num_nodes, pe_dim):
    """Compute Laplacian Positional Encoding matching NAGphormer's utils.laplacian_positional_encoding.

    NAGphormer computes PE on the UNWEIGHTED 0/1 adjacency (dgl_graph_to_sparse_adj returns
    values=1 regardless of edge weights), then uses the integer in-degree for normalisation.
    The result is the standard normalised graph Laplacian eigenvectors of the
    bidirected+self-loop graph structure.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs

    n = int(num_nodes)
    row_np = row_aug.numpy().astype(np.int32)
    col_np = col_aug.numpy().astype(np.int32)

    # Unweighted 0/1 adjacency — matches dgl_graph_to_sparse_adj (data=np.ones)
    data = np.ones(len(row_np), dtype=np.float64)
    A_01 = sp.csr_matrix((data, (row_np, col_np)), shape=(n, n))

    # Integer in-degrees (edge count, not weighted sum)
    in_deg = np.array(A_01.sum(axis=0), dtype=np.float64).flatten().clip(1)
    D_inv_sqrt = sp.diags(in_deg ** -0.5)

    # L = I - D^{-1/2} A_01 D^{-1/2}  (standard normalised Laplacian)
    L = sp.eye(n, format="csr") - D_inv_sqrt @ A_01 @ D_inv_sqrt

    try:
        EigVal, EigVec = eigs(L, k=pe_dim + 1, which="SR", tol=1e-2)
    except Exception:
        # Fallback for very small or disconnected graphs
        EigVal, EigVec = np.linalg.eigh(L.toarray())
        EigVal, EigVec = EigVal[: pe_dim + 1], EigVec[:, : pe_dim + 1]

    order = EigVal.real.argsort()
    EigVec = EigVec[:, order]
    lap_pe = torch.from_numpy(EigVec[:, 1 : pe_dim + 1].real).float()
    return lap_pe


def _compute_multihop_features(edge_index_global, num_nodes, features, hops, pe_dim=0, rank=0):
    """Pre-compute K-hop neighborhood feature aggregations for NAGphormer.

    Faithfully reproduces NAGphormer's full data-preprocessing pipeline:
      1. Convert to bidirected — NAGphormer calls dgl.to_bidirected on every dataset.
      2. Add self-loops — normalize_sparse_adj does adj = adj + I before normalising.
      3. Laplacian PE (pe_dim > 0): compute eigenvectors of the normalised Laplacian on
         the UNWEIGHTED (0/1) bidirected+self-loop structure (matching dgl_graph_to_sparse_adj
         which returns ones regardless of edge weights), then cat to node features.
         PE is concatenated BEFORE propagation so it is also smoothed across hops.
      4. Symmetric normalisation D^{-1/2}(A_bi+I)D^{-1/2} with degree from augmented adj.
      5. Iterative propagation: h_0 = features, h_k = A_norm @ h_{k-1}.
      6. Stack [h_0, …, h_hops] → (num_nodes, hops+1, d+pe_dim).
    """
    from torch_geometric.utils import degree as pyg_degree

    raw_dim = features.shape[-1]
    if rank == 0:
        total_dim = raw_dim + pe_dim
        mem_mib = num_nodes * (hops + 1) * total_dim * 4 / (1024 ** 2)
        print(
            f"[nagphormer] Pre-computing {hops}-hop features "
            f"(N={num_nodes:,}, d={raw_dim}, pe_dim={pe_dim}, est.={mem_mib:.1f} MiB) …"
        )
    t0 = time.time()

    row = edge_index_global[0].cpu()
    col = edge_index_global[1].cpu()

    # Step 1: bidirected
    row_bi = torch.cat([row, col])
    col_bi = torch.cat([col, row])
    idx = row_bi * num_nodes + col_bi
    idx, perm = torch.sort(idx)
    mask = torch.ones(idx.size(0), dtype=torch.bool)
    mask[1:] = idx[1:] != idx[:-1]
    perm = perm[mask]
    row_bi = row_bi[perm]
    col_bi = col_bi[perm]

    # Step 2: self-loops
    self_nodes = torch.arange(num_nodes, dtype=torch.long)
    row_aug = torch.cat([row_bi, self_nodes])
    col_aug = torch.cat([col_bi, self_nodes])

    # Step 3: Laplacian PE — computed on the unweighted 0/1 graph topology,
    # then concatenated to features BEFORE propagation (matching NAGphormer data.py)
    x = features.float().cpu()
    if pe_dim > 0:
        if rank == 0:
            print(f"[nagphormer] Computing Laplacian PE (pe_dim={pe_dim}) …")
        lpe = _compute_laplacian_pe(row_aug, col_aug, num_nodes, pe_dim)
        x = torch.cat([x, lpe], dim=1)  # (num_nodes, raw_dim + pe_dim)

    # Step 4: weighted normalisation for propagation
    deg = pyg_degree(col_aug, num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    norm = deg_inv_sqrt[row_aug] * deg_inv_sqrt[col_aug]

    # Build a standard PyTorch sparse COO tensor (avoids torch_sparse C++ memory bugs).
    # adj[row_aug[k], col_aug[k]] = norm[k] — symmetric for bidirected+self-loop graphs,
    # so adj @ x is identical to the original SparseTensor(row=col_aug, col=row_aug) @ x.
    indices = torch.stack([row_aug, col_aug])  # (2, E)
    adj = torch.sparse_coo_tensor(
        indices, norm.float(), (num_nodes, num_nodes)
    ).coalesce()

    # Step 5: iterative propagation
    hop_feats = [x]
    for _ in range(hops):
        x = torch.sparse.mm(adj, x)
        hop_feats.append(x)

    result = torch.stack(hop_feats, dim=1)  # (num_nodes, hops+1, raw_dim+pe_dim)
    if rank == 0:
        actual_mib = result.numel() * result.element_size() / (1024 ** 2)
        print(
            f"[nagphormer] Multi-hop features ready in {time.time() - t0:.2f}s "
            f"({actual_mib:.1f} MiB)"
        )
    return result


def _generate_expander_edges(num_nodes, degree, seed=0):
    """Generate a random d-regular expander graph using Hamiltonian cycles.

    Matches Exphormer's default 'Hamiltonian' algorithm.  Each of the `degree`
    cycles is a random permutation of all nodes visited in order; edges are
    added in both directions.  Self-loops are removed.

    Returns edge_index (2, ~num_nodes * degree * 2) with global node indices.
    """
    import numpy as np
    rng = np.random.default_rng(int(seed))
    senders, receivers = [], []
    for _ in range(degree):
        perm = rng.permutation(num_nodes).tolist()
        for idx in range(num_nodes):
            v = perm[idx]
            u = perm[(idx - 1) % num_nodes]
            senders.extend([v, u])
            receivers.extend([u, v])
    row = torch.tensor(senders, dtype=torch.long)
    col = torch.tensor(receivers, dtype=torch.long)
    mask = row != col
    return torch.stack([row[mask], col[mask]], dim=0)


EDGE_POLICY_GPU_PERSIST = "gpu_persist"
EDGE_POLICY_GPU_EPHEMERAL = "gpu_ephemeral"
EDGE_POLICY_CPU_BROADCAST = "cpu_broadcast"
EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH = "cpu_rank_local_prefetch"
EDGE_POLICY_CPU_BROADCAST_PREFETCH = "cpu_broadcast_prefetch"
_EDGE_POLICY_CHOICES = frozenset(
    {
        EDGE_POLICY_GPU_PERSIST,
        EDGE_POLICY_GPU_EPHEMERAL,
        EDGE_POLICY_CPU_BROADCAST,
        EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH,
        EDGE_POLICY_CPU_BROADCAST_PREFETCH,
    }
)


def _make_edge_dst_stats_cache_key(edge_index: torch.Tensor) -> tuple:
    dev = edge_index.device
    return (
        int(edge_index.data_ptr()),
        int(edge_index.size(1)),
        dev.type,
        dev.index,
    )


def _get_edge_dst_stats(edge_index: torch.Tensor) -> Optional[dict]:
    if edge_index is None or edge_index.numel() == 0 or edge_index.device.type != "cpu":
        return None

    key = _make_edge_dst_stats_cache_key(edge_index)
    cached = _EDGE_DST_STATS_CACHE.get(key)
    if cached is not None:
        return cached

    dst = edge_index[1].to(torch.long)
    num_groups = int(dst.max().item()) + 1 if dst.numel() > 0 else 0
    counts = torch.bincount(dst, minlength=num_groups) if num_groups > 0 else dst.new_zeros((0,))
    cached = {
        "counts": counts,
        "max_count": int(counts.max().item()) if counts.numel() > 0 else 0,
        "heavy_group_maps": {},
    }
    if len(_EDGE_DST_STATS_CACHE) >= 2:
        _EDGE_DST_STATS_CACHE.pop(next(iter(_EDGE_DST_STATS_CACHE)))
    _EDGE_DST_STATS_CACHE[key] = cached
    return cached


def _get_heavy_group_map(dst_stats: Optional[dict], max_edges_per_query: int) -> Optional[torch.Tensor]:
    if dst_stats is None:
        return None

    k = int(max_edges_per_query)
    cached = dst_stats["heavy_group_maps"].get(k)
    if cached is not None:
        return cached

    counts = dst_stats["counts"]
    group_map = torch.full((counts.numel(),), -1, dtype=torch.long, device=counts.device)
    heavy_nodes = torch.nonzero(counts > k, as_tuple=False).view(-1)
    if heavy_nodes.numel() > 0:
        group_map[heavy_nodes] = torch.arange(
            heavy_nodes.numel(),
            dtype=torch.long,
            device=counts.device,
        )
    if len(dst_stats["heavy_group_maps"]) >= 4:
        dst_stats["heavy_group_maps"].pop(next(iter(dst_stats["heavy_group_maps"])))
    dst_stats["heavy_group_maps"][k] = group_map
    return group_map


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


def _resolve_real_edge_sampling_device(final_device, rw_device) -> torch.device:
    """Choose the device used to sample real edges before the final merge.

    When training runs on CUDA but random walks are explicitly placed on CPU,
    keep real-edge sampling on CPU as well so that only the sampled subset is
    transferred to the training GPU.
    """
    final_torch_device = torch.device(final_device)
    rw_torch_device = torch.device(rw_device)
    if final_torch_device.type == "cuda" and rw_torch_device.type == "cpu":
        return torch.device("cpu")
    return final_torch_device


def _resolve_edge_policy(args, edge_index_global: Optional[torch.Tensor] = None, edge_policy: Optional[str] = None) -> str:
    policy = edge_policy
    if policy is None:
        policy = getattr(args, "_runtime_edge_policy", None)
    if policy is None:
        if edge_index_global is not None and edge_index_global.device.type == "cuda":
            policy = EDGE_POLICY_GPU_PERSIST
        else:
            policy = EDGE_POLICY_GPU_EPHEMERAL
    policy = str(policy)
    if policy not in _EDGE_POLICY_CHOICES:
        raise ValueError(f"Unsupported edge policy: {policy!r}")
    return policy


def _edge_policy_build_devices(edge_policy: str, device: str, rw_device) -> tuple[torch.device, torch.device]:
    policy = str(edge_policy)
    if policy not in _EDGE_POLICY_CHOICES:
        raise ValueError(f"Unsupported edge policy: {policy!r}")
    if policy in (EDGE_POLICY_CPU_BROADCAST, EDGE_POLICY_CPU_BROADCAST_PREFETCH):
        return torch.device("cpu"), torch.device("cpu")
    if policy == EDGE_POLICY_CPU_RANK_LOCAL_PREFETCH:
        return torch.device(device), torch.device("cpu")
    return torch.device(device), torch.device(rw_device)


def _edge_policy_force_broadcast(args, edge_policy: str, force_broadcast: Optional[bool] = None) -> bool:
    if force_broadcast is not None:
        return bool(force_broadcast)
    return (
        bool(getattr(args, "force_edge_broadcast", False))
        or bool(getattr(args, "_runtime_force_edge_broadcast", False))
        or edge_policy in (EDGE_POLICY_CPU_BROADCAST, EDGE_POLICY_CPU_BROADCAST_PREFETCH)
    )


def _budget_phase_probe_rw_device(rw_device) -> str:
    return "cpu"


@contextlib.contextmanager
def _budget_phase_cpu_edge_baseline(args):
    prev_policy = getattr(args, "_runtime_edge_policy", None)
    prev_force = getattr(args, "_runtime_force_edge_broadcast", None)
    args._runtime_edge_policy = EDGE_POLICY_CPU_BROADCAST
    args._runtime_force_edge_broadcast = True
    try:
        yield
    finally:
        if prev_policy is None:
            if hasattr(args, "_runtime_edge_policy"):
                delattr(args, "_runtime_edge_policy")
        else:
            args._runtime_edge_policy = prev_policy
        if prev_force is None:
            if hasattr(args, "_runtime_force_edge_broadcast"):
                delattr(args, "_runtime_force_edge_broadcast")
        else:
            args._runtime_force_edge_broadcast = prev_force


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


def _cuda_empty_cache(args, full: bool = True) -> None:
    """Free the PyTorch CUDA caching allocator.

    ``full=True``  (default): also runs gc.collect() before flushing.
                  Use before large one-off allocations where an accurate
                  free-memory reading is required (e.g. edge-index caching,
                  H2D timing measurements).
    ``full=False``: skips gc.collect().  Use after routine tensor deletions
                  where the caching allocator will handle block reuse on its
                  own and a full GC pause is undesirable.
    """
    if full:
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
    return t_h2d


def _to_bidirected_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index_bi = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return coalesce(edge_index_bi, num_nodes=num_nodes)


def _load_default_split(dataset_name: str, root_dir: str, wait_for_rank0: bool = False, split_id: int = 0):
    return load_default_split(
        dataset_name,
        root_dir,
        dist_module=dist,
        wait_for_rank0=wait_for_rank0,
        split_id=split_id,
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

    if getattr(args, "force_random_split", False):
        split_idx = None
        if args.rank == 0:
            print("[split] --force_random_split set; skipping default split lookup.")
    else:
        split_idx = _load_default_split(
            args.dataset,
            args.dataset_dir,
            wait_for_rank0=True,
            split_id=int(getattr(args, "split_id", 0)),
        )
    if split_idx is None:
        if args.rank == 0:
            print("[split] Using random 60/20/20 split.")
        split_idx = random_split_idx(y, 0.6, 0.2, 0.2, args.seed)
    else:
        if args.rank == 0:
            print(f"[split] Loaded dataset split (split_id={int(getattr(args, 'split_id', 0))}).")

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
    # feature may be (N, d) for GT/Graphormer or (N, hops+1, d) for NAGphormer;
    # shape[-1] always gives the raw input feature dimension.
    input_dim = feature.shape[-1]
    common = dict(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=out_dim,
        attn_bias_dim=args.attn_bias_dim,
        dropout_rate=args.dropout_rate,
        input_dropout_rate=args.input_dropout_rate,
        attention_dropout_rate=args.attention_dropout_rate,
        ffn_dim=args.ffn_dim,
    )

    def _configure_activation_checkpoint(model, *, allow_adaptive: bool):
        _ckpt_mode = getattr(args, "activation_checkpoint_mode", None)
        if _ckpt_mode == "adaptive" and not allow_adaptive:
            raise ValueError(
                "--activation_checkpoint_mode adaptive is only implemented for Graphormer. "
                f"Use --activation_checkpoint_mode all, ffn_only, or multi_tier with --model {args.model}."
            )
        if not hasattr(model, "set_activation_checkpoint"):
            if _ckpt_mode is not None:
                import warnings
                warnings.warn(
                    f"--activation_checkpoint_mode {_ckpt_mode!r} was requested but "
                    f"{type(model).__name__} does not implement set_activation_checkpoint. "
                    "Activation checkpointing will be disabled.",
                    stacklevel=3,
                )
            return
        # Defer checkpoint calibration when adaptive_edge_budget is also active:
        # peak_warmup measured before the budget stabilises is too small and
        # produces wrong keep_mha layer counts. multi_tier also starts deferred
        # since its WARMUP/CALIBRATE phases use the same signal.
        _ckpt_deferred = (
            _ckpt_mode in ("adaptive", "multi_tier")
            and _adaptive_edge_budget_enabled(args)
        )
        model.set_activation_checkpoint(
            mode=_ckpt_mode,
            deferred=_ckpt_deferred,
        )
        if _ckpt_mode == "multi_tier":
            _mgr = getattr(model, "_comm_ckpt", None)
            if _mgr is None:
                raise RuntimeError(
                    f"{type(model).__name__}.set_activation_checkpoint(mode='multi_tier') was "
                    "called but model._comm_ckpt is still None.  The model's "
                    "set_activation_checkpoint implementation must create a "
                    "_MultiTierResourceManager and assign it to self._comm_ckpt."
                )
            _limit_mib = int(getattr(args, "multi_tier_gpu_memory_limit_mib", 0) or 0)
            if _limit_mib > 0 and hasattr(_mgr, "set_gpu_memory_limit_bytes"):
                _mgr.set_gpu_memory_limit_bytes(_limit_mib * (1024 ** 2))

    if args.model in _GRAPHORMER_VARIANTS:
        model_cls = _GRAPHORMER_VARIANTS[args.model]
        model = model_cls(
            **common,
            num_global_node=args.num_global_node,
        ).to(device)
        _configure_activation_checkpoint(model, allow_adaptive=True)
    elif args.model == "gt":
        model = GT(
            **common,
            num_global_node=0,
        ).to(device)
        _configure_activation_checkpoint(model, allow_adaptive=False)
    elif args.model == "exphormer":
        model = Exphormer(
            **common,
            num_global_node=0,
        ).to(device)
        _configure_activation_checkpoint(model, allow_adaptive=False)
    elif args.model == "graphgps":
        model = GraphGPS(
            **common,
            num_global_node=0,
            attn_type=getattr(args, "attn_type", "sparse"),
        ).to(device)
        _configure_activation_checkpoint(model, allow_adaptive=False)
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
    max_total_edges_per_query: int
    static_seed_epochs: int


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
        probe_size = valid_n if valid_n > 0 else 0
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

    static_seed_epochs = int(getattr(args, "adaptive_edge_budget_static_seed_epochs", 0))
    if enabled:
        if static_seed_epochs < 0:
            static_seed_epochs = 0
    static_seed_epochs = max(0, static_seed_epochs)

    return _AdaptiveEdgeBudgetConfig(
        enabled=enabled,
        probe_size=probe_size,
        block_size=max(1, block_size),
        warmup_epochs=warmup_epochs,
        patience=patience,
        max_total_edges_per_query=max(0, int(getattr(args, "max_total_edges_per_query", 0))),
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


def _sample_edges_per_query_random_minhash_mask(src, dst, max_edges_per_query: int, seed: int):
    max_edges_per_query = int(max_edges_per_query)
    if src.numel() <= max_edges_per_query:
        return torch.ones(src.numel(), dtype=torch.bool, device=src.device)

    scale = 1 << 31
    rand_key = torch.remainder(
        src * 1103515245 + dst * 214013 + int(seed) * 2654435761 + 12345,
        scale,
    ).to(torch.long)
    num_groups = int(dst.max().item()) + 1
    if num_groups <= 0:
        return torch.ones(src.numel(), dtype=torch.bool, device=src.device)

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

    return keep_mask


def _sample_edges_per_query_random_minhash(
    edge_index,
    max_edges_per_query: int,
    seed: int,
    *,
    dst_stats: Optional[dict] = None,
):
    max_edges_per_query = int(max_edges_per_query)
    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)
    if src.numel() <= max_edges_per_query:
        return edge_index
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        return _sample_edges_per_query_random_sort(edge_index, max_edges_per_query, seed)

    if (
        dst_stats is not None
        and dst.device.type == "cpu"
        and dst_stats.get("counts") is not None
        and int(dst_stats["counts"].numel()) > 0
    ):
        if int(dst_stats.get("max_count", 0)) <= max_edges_per_query:
            return edge_index

        edge_counts = dst_stats["counts"].index_select(0, dst)
        keep_mask = edge_counts <= max_edges_per_query
        if bool(keep_mask.all().item()):
            return edge_index

        heavy_mask = ~keep_mask
        heavy_group_map = _get_heavy_group_map(dst_stats, max_edges_per_query)
        if heavy_group_map is not None and heavy_group_map.numel() > 0:
            heavy_keep = _sample_edges_per_query_random_minhash_mask(
                src[heavy_mask],
                heavy_group_map.index_select(0, dst[heavy_mask]),
                max_edges_per_query,
                seed,
            )
            keep_mask = keep_mask.clone()
            keep_mask[heavy_mask] = heavy_keep
            return edge_index[:, keep_mask]

    keep_mask = _sample_edges_per_query_random_minhash_mask(
        src,
        dst,
        max_edges_per_query,
        seed,
    )
    return edge_index[:, keep_mask]


def _sample_edges_per_query_random(
    edge_index,
    max_edges_per_query: int,
    seed: int,
    *,
    dst_stats: Optional[dict] = None,
):
    if edge_index is None or edge_index.numel() == 0 or int(max_edges_per_query) <= 0:
        return edge_index

    max_edges_per_query = int(max_edges_per_query)
    if max_edges_per_query <= _MINHASH_SAMPLE_K_LIMIT:
        return _sample_edges_per_query_random_minhash(
            edge_index,
            max_edges_per_query,
            seed,
            dst_stats=dst_stats,
        )
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

    build_device = torch.device(device)

    # Proactive GPU-OOM guard for real-edge sampling. The minhash-mask
    # path inside _sample_edges_per_query_random_minhash_mask needs a few
    # single-shot int64 allocations of size E (E ≈ |edge_index_global|),
    # the largest being min_key.index_select(0, dst) and the rand_key
    # tensor — each ~8 × E bytes. On amazon (E = 264M) that's ~2 GiB
    # contiguous each. If usable large_pool memory can't fit a single
    # such allocation, demote sampling to CPU to avoid OOM partway
    # through the build. Uses the same per-large_pool accounting as the
    # RW pre-check (no magic multipliers).
    if build_device.type == "cuda":
        E_real = int(edge_index_global.size(1))
        single_alloc = max(0, E_real) * 8  # int64
        # Sampling allocates ~3-4 such tensors live concurrently; require
        # headroom for at least 4 of them so partway allocs don't OOM.
        sampling_estimate = 4 * single_alloc
        from gt_sp.utils import (
            _gpu_can_fit_rw as _gpu_can_fit,
            _gpu_concurrent_overhead_bytes,
            _rw_log_is_rank0,
            _RW_DEVICE_DECISION_CACHE,
        )
        if not _gpu_can_fit(build_device, sampling_estimate):
            cache_key = ("real_edge_sampling", str(build_device), E_real, "cpu")
            prev = _RW_DEVICE_DECISION_CACHE.get(cache_key)
            if prev is None and _rw_log_is_rank0():
                try:
                    free, _ = torch.cuda.mem_get_info(build_device)
                    free_mib = free / (1024 ** 2)
                except Exception:
                    free_mib = -1.0
                overhead_mib = _gpu_concurrent_overhead_bytes(build_device) / (1024 ** 2)
                print(
                    f"[rw-device] real-edge sampling: peak single alloc "
                    f"{single_alloc/(1024**2):.0f} MiB × 4 concurrent = "
                    f"{sampling_estimate/(1024**2):.0f} MiB exceeds usable GPU "
                    f"(free={free_mib:.0f} MiB − overhead {overhead_mib:.0f} MiB); "
                    f"forcing CPU sampling for E={E_real:,}. NOTE: this "
                    f"overrides any GPU-based edge_policy announced by "
                    f"multi_tier ACTIVE. (Further matching calls suppressed.)"
                )
            _RW_DEVICE_DECISION_CACHE[cache_key] = (sampling_estimate, 0)
            build_device = torch.device("cpu")

    if edge_index_global.device == build_device:
        real_edges = edge_index_global
    else:
        payload_bytes = int(edge_index_global.numel() * edge_index_global.element_size())
        real_edges = profile_call(
            "edge_full_h2d",
            lambda: edge_index_global.to(build_device),
            device=build_device,
            payload_bytes=payload_bytes,
        )
    if not _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return real_edges

    real_budget = _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)
    if real_budget <= 0:
        return None

    dst_stats = None
    if build_device.type == "cpu" and real_edges.device.type == "cpu":
        dst_stats = _get_edge_dst_stats(real_edges)

    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        real_edges = profile_call(
            "edge_real_sample",
            lambda: _sample_edges_per_query_random(
                real_edges,
                real_budget,
                _edge_block_seed(args, edge_seed, 17),
                dst_stats=dst_stats,
            ),
            device=build_device,
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


def _subtract_edge_index(a, b, num_nodes):
    # Drop edges in `a` whose (src, dst) pair also appears in `b`.
    # Used to deduplicate the probe-time RW pool against the real-edge pool so
    # the budget probe sees disjoint candidate sets.
    if a is None or a.numel() == 0:
        return a
    if b is None or b.numel() == 0:
        return a
    n = max(1, int(num_nodes))
    if b.device != a.device:
        b = b.to(a.device)
    a_keys = a[0].to(torch.long) * n + a[1].to(torch.long)
    b_keys = b[0].to(torch.long) * n + b[1].to(torch.long)
    keep = ~torch.isin(a_keys, b_keys)
    if keep.all():
        return a
    return a[:, keep].contiguous()


def _compute_sorted_edge_keys(edges, num_nodes):
    """Return ``sort(edges[0] * num_nodes + edges[1])``.

    Pre-computing once and caching this lets ``_subtract_edge_index_sorted``
    skip a ~2 GiB int64 keys materialisation and a 264M-element sort on
    every probe — the dominant cost in build_pools on large graphs.
    """
    if edges is None or edges.numel() == 0:
        return None
    n = max(1, int(num_nodes))
    keys = edges[0].to(torch.long) * n + edges[1].to(torch.long)
    sorted_keys, _ = keys.sort()
    return sorted_keys


def _subtract_edge_index_sorted(a, b_keys_sorted, num_nodes):
    """Same semantics as ``_subtract_edge_index`` but expects ``b``'s edge
    keys already computed and sorted ascending.

    Uses ``torch.searchsorted`` + equality test instead of ``torch.isin``:
      - skips materialising b_keys (~2 GiB for amazon's real_pool)
      - skips the internal isin sort on 264M elements
      - searchsorted on a_keys is O(|a| · log|b|), very fast on int64.
    """
    if a is None or a.numel() == 0:
        return a
    if b_keys_sorted is None or b_keys_sorted.numel() == 0:
        return a
    n = max(1, int(num_nodes))
    a_keys = a[0].to(torch.long) * n + a[1].to(torch.long)
    pos = torch.searchsorted(b_keys_sorted, a_keys)
    pos_clamped = pos.clamp(max=b_keys_sorted.numel() - 1)
    in_b = b_keys_sorted[pos_clamped] == a_keys
    keep = ~in_b
    if bool(keep.all().item()):
        return a
    return a[:, keep].contiguous()


@dataclass(frozen=True)
class _DstEdgeCSR:
    """CSR indexed by destination node for O(probe_size × avg_degree) edge filtering.

    Replaces O(E) torch.isin scans when repeatedly querying a small probe set
    against a large static edge_index (e.g. edge_index_global during bootstrap
    and adaptive budget updates).
    """
    sorted_edge_index: torch.Tensor  # [2, E] sorted by dst, on CPU
    row_ptr: torch.Tensor            # [num_nodes + 1] cumulative counts, on CPU


def _build_dst_csr(edge_index: torch.Tensor, num_nodes: int) -> _DstEdgeCSR:
    """Build a dst-indexed CSR from a COO edge_index.

    One-time O(E log E) cost at startup; subsequent per-probe filtering is
    O(probe_size × avg_degree) instead of O(E).
    """
    ei = edge_index.cpu().to(torch.long)
    perm = torch.argsort(ei[1], stable=True)
    sorted_ei = ei[:, perm].contiguous()
    counts = torch.bincount(sorted_ei[1], minlength=num_nodes)
    row_ptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    torch.cumsum(counts, dim=0, out=row_ptr[1:])
    return _DstEdgeCSR(sorted_edge_index=sorted_ei, row_ptr=row_ptr)


def _filter_by_dst_csr(csr: _DstEdgeCSR, dst_nodes: torch.Tensor) -> torch.Tensor:
    """Return edges whose dst ∈ dst_nodes using a pre-built CSR (CPU output)."""
    dst_nodes = dst_nodes.cpu().to(torch.long).view(-1)
    num_nodes = int(csr.row_ptr.numel()) - 1
    dst_nodes = dst_nodes[(dst_nodes >= 0) & (dst_nodes < num_nodes)]
    if dst_nodes.numel() == 0:
        return csr.sorted_edge_index.new_zeros((2, 0))
    ptr_starts = csr.row_ptr[dst_nodes]
    counts = csr.row_ptr[dst_nodes + 1] - ptr_starts
    total = int(counts.sum().item())
    if total == 0:
        return csr.sorted_edge_index.new_zeros((2, 0))
    # Build flat edge indices without Python loops.
    # For each segment i with length counts[i], produce ptr_starts[i]+0, +1, ..., +counts[i]-1.
    base = torch.repeat_interleave(ptr_starts, counts)
    cum_starts = torch.zeros(dst_nodes.numel(), dtype=torch.long)
    cum_starts[1:] = counts[:-1].cumsum(0)
    local_offset = torch.arange(total, dtype=torch.long) - torch.repeat_interleave(cum_starts, counts)
    return csr.sorted_edge_index[:, base + local_offset]


def _build_or_retrieve_full_rw_pool(
    args,
    edge_index_global,
    num_nodes,
    rw_device,
    walk_length,
    walks_per_node,
    num_heads,
    edge_seed,
):
    """Build the full-graph RW pool, or retrieve it from a single-slot cache
    when the same epoch's training step has already built it.

    Cache key (everything that influences the RW realization):
      (id(edge_index_global), num_nodes, walk_length, walks_per_node,
       num_heads, edge_seed, rw_device).

    Why this exists: training and probe were independently rebuilding the
    full pool every epoch (~40-60s on amazon-scale graphs) under different
    seeds. By sharing one seed per epoch and one cached pool, probe and
    training observe the same RW realization → candidate-vs-base
    comparisons become CRN-coupled at the pool level (not just the
    per-candidate sample level), maximising signal-to-noise.

    Cross-epoch resampling is preserved: the next epoch's training step
    overwrites the cache with a new ``edge_seed``-keyed entry.
    """
    cache_key = (
        id(edge_index_global),
        int(num_nodes),
        int(walk_length),
        int(walks_per_node),
        int(num_heads),
        int(edge_seed) if edge_seed is not None else None,
        str(rw_device),
    )
    cached = getattr(args, "_cached_full_rw_pool", None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["pool"]

    rw_torch_device = torch.device(rw_device)
    seed_ctx = (
        _fixed_torch_cpu_seed(edge_seed)
        if edge_seed is not None and rw_torch_device.type == "cpu"
        else contextlib.nullcontext()
    )
    with seed_ctx:
        pool = profile_call(
            "edge_rw_build",
            lambda: build_head_hop_edges(
                edge_index=edge_index_global,
                num_nodes=num_nodes,
                num_heads=num_heads,
                num_groups=1,
                device=rw_device,
                walk_length=walk_length,
                walks_per_node=walks_per_node,
            ),
            device=rw_device,
        )
    if isinstance(pool, list):
        pool = _merge_edge_index_list(pool)

    args._cached_full_rw_pool = {"key": cache_key, "pool": pool}
    return pool


def _build_probe_edge_pools(
    args,
    edge_index_global,
    num_nodes,
    rw_device,
    probe_idx_global,
    edge_seed=None,
    adaptive_edge_budget_cfg=None,
    walk_length_override=None,
    edge_index_csr=None,
    edge_dst_filter=None,
    precomputed_real_pool=None,
    precomputed_real_pool_sorted_keys=None,
):
    probe_idx_global = probe_idx_global.to(dtype=torch.long, device="cpu").view(-1)
    # edge_dst_filter is the expanded set (k-hop in-neighbors) used for edge
    # selection; falls back to probe_idx_global when not provided.
    dst_filter = edge_dst_filter if edge_dst_filter is not None else probe_idx_global
    dst_filter = dst_filter.to(dtype=torch.long, device="cpu").view(-1)
    seed_ctx = (
        _fixed_torch_cpu_seed(edge_seed)
        if edge_seed is not None and torch.device(rw_device).type == "cpu"
        else contextlib.nullcontext()
    )
    with seed_ctx:
        real_pool = None
        if _use_real_edges(args, adaptive_edge_budget_cfg):
            if precomputed_real_pool is not None:
                # Cross-epoch cache hit: skip the ~30-50s CSR gather.
                real_pool = precomputed_real_pool
            elif edge_index_csr is not None:
                real_pool = _filter_by_dst_csr(edge_index_csr, dst_filter)
            else:
                real_pool = _filter_edge_index_by_dst(edge_index_global.cpu(), dst_filter)
            if real_pool is not None:
                real_pool = real_pool.to(dtype=torch.long, device="cpu")

        rw_pool = None
        if _use_rw_edges(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg):
            wl = _resolve_walk_length(
                args,
                walk_length_override=walk_length_override,
            )
            # Use the shared cache so probe inherits training's just-built
            # full pool (CRN-coupling at the realization level). Key
            # alignment: edge_seed here must match training's epoch seed.
            rw_pool = _build_or_retrieve_full_rw_pool(
                args,
                edge_index_global=edge_index_global,
                num_nodes=num_nodes,
                rw_device=rw_device,
                walk_length=wl,
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
                num_heads=args.num_heads,
                edge_seed=edge_seed,
            )
            if rw_pool is not None:
                rw_pool = _filter_edge_index_by_dst(rw_pool, dst_filter)
                rw_pool = rw_pool.to(dtype=torch.long, device="cpu")
            # Probe-only dedup: remove RW edges already covered by real edges so
            # the probe's marginal-gain estimate of the RW budget isn't inflated
            # by overlap. Training-time edge construction is unaffected.
            if (
                rw_pool is not None
                and rw_pool.numel() > 0
                and real_pool is not None
                and real_pool.numel() > 0
            ):
                if precomputed_real_pool_sorted_keys is not None:
                    rw_pool = _subtract_edge_index_sorted(
                        rw_pool, precomputed_real_pool_sorted_keys, num_nodes
                    )
                else:
                    rw_pool = _subtract_edge_index(rw_pool, real_pool, num_nodes)

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
    if args.model in _GRAPHORMER_VIRTUAL_NODE_MODELS and args.num_global_node > 0:
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
    edge_dst_filter=None,
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
                edge_dst_filter=edge_dst_filter,
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

    return _broadcast_edges(merged_cpu, device, sp_group, sp_src_rank, sp_rank)



class _AdaptiveEdgeBudgetController:
    def __init__(self, config: _AdaptiveEdgeBudgetConfig) -> None:
        self.enabled = config.enabled
        self.block_size = int(config.block_size)
        self.max_total = int(config.max_total_edges_per_query)
        self.warmup_epochs = None if config.warmup_epochs is None else int(config.warmup_epochs)
        self.patience = int(config.patience)
        self.bad_rounds = 0
        self.seen_positive_gain = False

        self.auto_hold_released = False
        self.auto_hold_neg_inf_streak = 0
        self.auto_hold_neg_inf_freeze_after = 5
        self.baseline_acc = -1.0
        self.bootstrap_baseline_loss = float("inf")
        self.bootstrap_baseline_acc = -1.0
        
        self.frozen = not self.enabled
        if not self.enabled:
            self.real_budget = self.max_total
            self.rw_budget = 0
            self.walk_length = None
            return

        self.real_budget = 2 if self.max_total > 0 else 0
        self.rw_budget = 2 if self.max_total > 0 else 0
        self.walk_length = None
        # Dynamic confirmation: the n-th budget adjustment requires n consecutive
        # wins before committing (capped at _max_confirm_rounds).  Early moves
        # have large marginal gains (clear signal); later moves are smaller and
        # closer to noise level, so they need more evidence.
        self._n_budget_updates: int = 0
        self._max_confirm_rounds: int = 3
        self._pending_kind: str | None = None
        self._pending_state: dict | None = None
        self._pending_wins: int = 0

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
        # Expand total budget if headroom remains.
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
        # At the budget ceiling: offer incremental shift candidates so the
        # controller can redistribute between real and rw even after both sides
        # have grown.  Dynamic confirmation (in update()) prevents noise-driven
        # oscillation — a shift must win consecutively before being committed.
        if self.real_budget + self.rw_budget >= self.max_total:
            if self.rw_budget >= self.block_size:
                cand = {
                    "real_edges_per_query": self.real_budget + self.block_size,
                    "rw_edges_per_query":   self.rw_budget - self.block_size,
                }
                if self.walk_length is not None:
                    cand["walk_length"] = int(self.walk_length)
                out["shift_to_real"] = cand
            if self.real_budget >= self.block_size:
                cand = {
                    "real_edges_per_query": self.real_budget - self.block_size,
                    "rw_edges_per_query":   self.rw_budget + self.block_size,
                }
                if self.walk_length is not None:
                    cand["walk_length"] = int(self.walk_length)
                out["shift_to_rw"] = cand
        return out

    def set_state(self, state) -> None:
        if state is None:
            return
        self.real_budget = int(state.get("real_edges_per_query", self.real_budget))
        self.rw_budget = int(state.get("rw_edges_per_query", self.rw_budget))
        if "walk_length" in state:
            self.walk_length = int(state["walk_length"])

    def required_confirm_rounds(self) -> int:
        return min(self._n_budget_updates + 1, self._max_confirm_rounds)

    def update(self, choice, best_gain, next_state=None):
        if best_gain > 0.0:
            self.seen_positive_gain = True
        if next_state is not None and choice is not None:
            required = self.required_confirm_rounds()
            if choice == self._pending_kind:
                self._pending_wins += 1
            else:
                # Direction changed: restart confirmation for new direction.
                self._pending_kind = choice
                self._pending_state = next_state
                self._pending_wins = 1
            if self._pending_wins >= required:
                self.real_budget = int(self._pending_state["real_edges_per_query"])
                self.rw_budget = int(self._pending_state["rw_edges_per_query"])
                if "walk_length" in self._pending_state:
                    self.walk_length = int(self._pending_state["walk_length"])
                self._n_budget_updates += 1
                self._pending_kind = None
                self._pending_state = None
                self._pending_wins = 0
                self.bad_rounds = 0
            return
        # No winning candidate this epoch: reset expand confirmation.
        self._pending_kind = None
        self._pending_state = None
        self._pending_wins = 0
        if not self.seen_positive_gain:
            return
        if best_gain <= 0.0:
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


def _expand_probe_idx_for_reception(probe_idx_global, edge_index_global, n_layers, edge_index_csr=None):
    """Expand probe nodes to include their (n_layers-1)-hop in-neighbors.

    For a k-layer model, the layer-k representation of a probe node depends on
    layer-(k-1) representations of its in-neighbors, which in turn depend on
    layer-(k-2) representations of *their* in-neighbors, and so on.  Including
    only edges whose dst is in the original probe set therefore produces biased
    (stunted) intermediate representations for non-probe neighbor nodes.

    This function returns the union of the probe set and all nodes that feed into
    probe representations through up to (n_layers-1) hops of message passing.
    Use the returned set as the dst filter for edge selection; keep the original
    probe_idx_global for loss computation.
    """
    if n_layers <= 1 or probe_idx_global is None or probe_idx_global.numel() == 0:
        return probe_idx_global

    ei = edge_index_global.cpu().to(torch.long)
    if ei.numel() == 0:
        return probe_idx_global
    num_nodes = int(ei.max().item()) + 1

    probe_cpu = probe_idx_global.cpu().to(torch.long).view(-1)
    valid = (probe_cpu >= 0) & (probe_cpu < num_nodes)
    probe_cpu = probe_cpu[valid]

    in_expanded = torch.zeros(num_nodes, dtype=torch.bool)
    in_expanded[probe_cpu] = True
    frontier = probe_cpu

    for _ in range(n_layers - 1):
        if frontier.numel() == 0:
            break
        if edge_index_csr is not None:
            edges = _filter_by_dst_csr(edge_index_csr, frontier)
        else:
            mask = torch.isin(ei[1], frontier)
            edges = ei[:, mask]
        if edges.numel() == 0:
            break
        srcs = edges[0].unique()
        srcs = srcs[(srcs >= 0) & (srcs < num_nodes)]
        new_nodes = srcs[~in_expanded[srcs]]
        if new_nodes.numel() == 0:
            break
        in_expanded[new_nodes] = True
        frontier = new_nodes

    return in_expanded.nonzero(as_tuple=False).view(-1).to(torch.long)


def _edge_count(attn_edges) -> int:
    if isinstance(attn_edges, dict):
        return int(attn_edges["src"].numel())
    return int(attn_edges.size(1))


def _probe_loss_sp(args, model, x_local, local_y, local_probe_idx, edge_index_probe,
                   device, amp_dtype, sp_group, sp_world_size):
    was_training = model.training
    model.eval()

    loss_sum = torch.zeros(1, device=device, dtype=torch.float32)
    acc_sum = torch.zeros(1, device=device, dtype=torch.float32)
    count = torch.zeros(1, device=device, dtype=torch.long)
    try:
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
    finally:
        if was_training:
            model.train()

    if sp_world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=sp_group)

    mean_loss = float(loss_sum.item() / max(int(count.item()), 1))
    mean_acc = float(acc_sum.item() / max(int(count.item()), 1))
    del out_local
    return mean_loss, mean_acc


def _build_optimizer_bundle(args, model, device, amp_dtype, n_train=None):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )

    warmup_epochs = args.warmup_updates
    tot_epochs = args.epochs
    tot_updates = int(getattr(args, "tot_updates", 0))
    if tot_updates > 0 and n_train is not None and n_train > 0:
        ref_batch = max(1, int(getattr(args, "lr_ref_batch_size", 1000)))
        batches_per_epoch = max(1, (n_train + ref_batch - 1) // ref_batch)
        warmup_epochs = max(1, round(args.warmup_updates / batches_per_epoch))
        tot_epochs = max(warmup_epochs + 1, round(tot_updates / batches_per_epoch))

    lr_schedule = getattr(args, "lr_schedule", "polynomial")
    if lr_schedule == "cosine_with_warmup":
        lr_scheduler = CosineWithWarmupLR(
            optimizer,
            warmup=warmup_epochs,
            tot=tot_epochs,
            lr=args.peak_lr,
        )
    else:
        lr_scheduler = PolynomialDecayLR(
            optimizer,
            warmup=warmup_epochs,
            tot=tot_epochs,
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
    with _budget_phase_cpu_edge_baseline(args):
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
    edge_index_csr=None,
):
    # Bootstrap search phase removed; initial budget defaults to adaptive controller start state.
    return None


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
    training_epoch_edge_seed=None,
):
    if (
        (not controller.enabled)
        or controller.frozen
        or epoch <= int(getattr(args, "warmup_updates", 5))
        or (controller.warmup_epochs is not None and epoch > controller.warmup_epochs)
    ):
        return

    if local_probe_idx is None or probe_idx_global is None or probe_idx_global.numel() == 0:
        controller.frozen = True
        return

    # Use the SAME edge_seed the training step used in this epoch, so probe
    # observes the same RW realization AND the per-candidate edge sampling
    # uses the same minhash seeds as training. With identical seeds, base
    # vs real_up vs rw_up candidate edge sets are *nested* (top-K vs
    # top-K+block from the same ranking) — the candidate gain measures
    # exactly the marginal effect of adding `block_size` edges, with zero
    # noise from independent random subsetting. Falls back to the old
    # independent probe seed when training_epoch_edge_seed is unknown
    # (e.g., when called without the new kwarg by older callers).
    if training_epoch_edge_seed is not None:
        probe_seed = int(training_epoch_edge_seed)
    else:
        probe_seed = int(getattr(args, "seed", 0)) + 100000 + int(epoch)
    base_state = controller.current_state()

    n_layers = max(1, int(getattr(args, "n_layers", 1)))

    # Build a dst-indexed CSR over edge_index_global once and cache it on the
    # controller. The CSR is graph-topology only (no seed / no budget) and lets
    # _expand_probe_idx_for_reception and _filter_by_dst_csr each replace an
    # O(|E|) torch.isin scan with an O(|frontier| × avg_deg) gather. On products
    # this swaps tens-of-million-edge scans for few-hundred-thousand-edge
    # gathers per BFS layer. CPU-only, no GPU footprint.
    edge_index_csr = getattr(controller, "_edge_csr", None)
    if edge_index_csr is None:
        edge_index_csr = _build_dst_csr(edge_index_global, num_nodes)
        controller._edge_csr = edge_index_csr

    # The BFS reception cone is purely a function of
    # (probe_idx_global, edge_index_global, n_layers) — all of which are
    # constant across the whole training run. Cache it on the controller
    # so subsequent probe epochs skip the ~8s expansion. Same idea for the
    # cone-filtered real_pool below.
    _deterministic_key = (
        id(edge_index_global),
        id(probe_idx_global),
        int(num_nodes),
        int(n_layers),
    )
    _cached_cone = getattr(controller, "_cached_cone", None)
    if _cached_cone is not None and _cached_cone.get("key") == _deterministic_key:
        edge_dst_filter = _cached_cone["filter"]
    else:
        edge_dst_filter = _expand_probe_idx_for_reception(
            probe_idx_global, edge_index_global, n_layers,
            edge_index_csr=edge_index_csr,
        )
        controller._cached_cone = {"key": _deterministic_key, "filter": edge_dst_filter}

    # Pre-build the cone-filtered real_pool too — same reasoning: it only
    # depends on (edge_index_csr, edge_dst_filter), both fixed across the
    # run. We also cache the *sorted edge keys* of the real_pool so
    # ``_subtract_edge_index`` can skip its dominant cost (re-materialising
    # 264M int64 keys + internal sort) on every probe.
    # Cache layout: store a bool MASK (size E, ~264 MB on amazon) instead of
    # the full filtered real_pool tensor (~4.2 GiB). The mask selects which
    # columns of edge_index_global have dst in the cone. On every probe we
    # materialize the actual real_pool from the mask on demand — peak
    # memory during the probe is unchanged, but sustained CPU usage
    # between probes drops by ~4 GiB. Correctness is bit-identical: the
    # same edges are produced, just possibly in a different column order
    # (which the per-dst sampler and sorted-keys subtract don't care about).
    _cached_real_pool = getattr(controller, "_cached_real_pool", None)
    if _cached_real_pool is not None and _cached_real_pool.get("key") == _deterministic_key:
        _real_mask = _cached_real_pool.get("mask")
        _precomputed_real_pool_sorted_keys = _cached_real_pool.get("sorted_keys")
        if _real_mask is not None:
            # Materialize transient real_pool — held only for this probe.
            _precomputed_real_pool = edge_index_global[:, _real_mask].to(
                dtype=torch.long, device="cpu"
            ).contiguous()
        else:
            _precomputed_real_pool = None
    else:
        _precomputed_real_pool = None
        _precomputed_real_pool_sorted_keys = None

    probe_rw_device = _budget_phase_probe_rw_device(rw_device)
    # Build the dst-filtered probe pools once on src_rank and reuse them across
    # base + every candidate within this probe epoch.  This is the old fast
    # path: probe edges are restricted to the validation reception cone, so
    # each candidate forward processes a tiny fraction of the full graph and
    # all candidates see the same underlying RW realization (CRN coupling).
    probe_edge_pools = None
    if sp_rank == sp_src_rank:
        probe_edge_pools = _build_probe_edge_pools(
            args,
            edge_index_global,
            num_nodes,
            probe_rw_device,
            probe_idx_global,
            edge_seed=probe_seed,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            walk_length_override=base_state.get("walk_length"),
            edge_dst_filter=edge_dst_filter,
            edge_index_csr=edge_index_csr,
            precomputed_real_pool=_precomputed_real_pool,
            precomputed_real_pool_sorted_keys=_precomputed_real_pool_sorted_keys,
        )
        if probe_edge_pools is not None:
            _real = probe_edge_pools.get("real")
            # First-time real_pool computed: derive a bool MASK over
            # edge_index_global (264 MB on amazon vs the 4.2 GiB pool
            # tensor), and cache mask + sorted_keys. Future probes will
            # materialize the transient real_pool from the mask via
            # `edge_index_global[:, mask]` — same edges, different column
            # order, identical sampling/subtract semantics.
            if _precomputed_real_pool is None and _real is not None:
                _sorted_keys = _compute_sorted_edge_keys(_real, num_nodes)
                # torch.isin on amazon-scale (264M dst, 1.17M cone): ~5-10s
                # one-time. Faster than rerunning _filter_by_dst_csr.
                _real_mask = torch.isin(
                    edge_index_global[1].to(dtype=torch.long, device="cpu"),
                    edge_dst_filter.to(dtype=torch.long, device="cpu"),
                )
                controller._cached_real_pool = {
                    "key": _deterministic_key,
                    "mask": _real_mask,
                    "sorted_keys": _sorted_keys,
                }
            # CSR was only needed to *build* cone + real_pool. Once both
            # downstream caches are populated, CSR is never read again on
            # this controller — drop it to free ~4 GiB CPU memory (amazon
            # scale). If the downstream caches ever invalidate, CSR is
            # cheap to rebuild (~5-10s) and will be re-cached next probe.
            if (
                getattr(controller, "_edge_csr", None) is not None
                and getattr(controller, "_cached_cone", None) is not None
                and getattr(controller, "_cached_real_pool", None) is not None
            ):
                controller._edge_csr = None

    def _probe_edges(budget_state):
        return _build_probe_attention_edges(
            args,
            edge_index_global,
            num_nodes,
            device,
            probe_rw_device,
            sp_group,
            sp_src_rank,
            sp_rank,
            local_nodes,
            probe_idx_global,
            edge_seed=probe_seed,
            edge_budget_state=budget_state,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            edge_pools=probe_edge_pools,
            walk_length_override=budget_state.get("walk_length"),
            edge_dst_filter=edge_dst_filter,
        )

    base_edges = _probe_edges(base_state)
    base_loss, base_acc = _probe_loss_sp(
        args, model, x_local, local_y, local_probe_idx, base_edges,
        device, amp_dtype, sp_group, sp_world_size,
    )
    base_count = _edge_count(base_edges)
    del base_edges

    best_kind = None
    best_gain = float("-inf")
    best_state = None
    best_loss = base_loss
    best_count = 0
    probe_rows = [
        {
            "kind": "current",
            "state": base_state,
            "loss": base_loss,
            "acc": base_acc,
            "gain": 0.0,
            "edges": base_count,
        }
    ]

    candidates = controller.candidate_states()
    is_lump_epoch = any(k.startswith("lump_") for k in candidates)

    for kind, cand_state in candidates.items():
        cand_edges = _probe_edges(cand_state)
        cand_loss, cand_acc = _probe_loss_sp(
            args, model, x_local, local_y, local_probe_idx, cand_edges,
            device, amp_dtype, sp_group, sp_world_size,
        )
        cand_count = _edge_count(cand_edges)
        del cand_edges

        gain = (base_loss - cand_loss) / max(base_loss, 1e-6)
        probe_rows.append(
            {
                "kind": kind,
                "state": cand_state,
                "loss": cand_loss,
                "acc": cand_acc,
                "gain": gain,
                "edges": cand_count,
            }
        )

        if gain > best_gain:
            best_kind = kind
            best_gain = gain
            best_state = cand_state
            best_loss = cand_loss
            best_count = cand_count

    def _budget_pair(state):
        return (
            int(state.get("real_edges_per_query", 0)),
            int(state.get("rw_edges_per_query", 0)),
        )

    winner_by_loss = min(probe_rows, key=lambda row: row["loss"])
    # Loss-only decision: commit iff a non-base candidate strictly beats base
    # on probe loss. No threshold (the progressive-confirmation gate in
    # controller.update absorbs noise filtering).
    commit_ok = best_gain > 0.0

    if args.rank == 0:
        cur_real, cur_rw = _budget_pair(base_state)
        loss_real, loss_rw = _budget_pair(winner_by_loss["state"])
        print(
            f"  ↳ BudgetCtrl probe epoch={epoch} current=({cur_real},{cur_rw}) "
            f"winner_by_loss={winner_by_loss['kind']}({loss_real},{loss_rw})"
        )
        for row in probe_rows:
            real_budget, rw_budget = _budget_pair(row["state"])
            gain_text = "base" if row["kind"] == "current" else f"{row['gain']:.8e}"
            print(
                f"     - {row['kind']:<13} budget=({real_budget},{rw_budget}) "
                f"loss={row['loss']:.6f} acc={row['acc']:.6f} "
                f"gain={gain_text} probe_edges={row['edges']}"
            )

    # Check Condition for Release or Continued Update
    if not controller.auto_hold_released:
        if best_gain == float("-inf"):
            controller.auto_hold_neg_inf_streak += 1
        else:
            controller.auto_hold_neg_inf_streak = 0
        if controller.auto_hold_neg_inf_streak >= controller.auto_hold_neg_inf_freeze_after:
            controller.frozen = True
            if args.rank == 0:
                print(
                    "  ↳ BudgetCtrl [AutoHoldFrozen] "
                    f"epoch={epoch} best_rel_gain_found=-inf repeated "
                    f"{controller.auto_hold_neg_inf_streak} epochs; "
                    "freezing budget."
                )
            return
        if commit_ok:
            controller.auto_hold_released = True
            controller.auto_hold_neg_inf_streak = 0
            if args.rank == 0:
                print(
                    "  ↳ BudgetCtrl [AutoHoldReleased] via positive rel_gain: "
                    f"{best_gain:.8e}"
                )
        else:
            if args.rank == 0:
                print(
                    f"  ↳ BudgetCtrl [AutoHold] epoch={epoch} searching... "
                    f"best_rel_gain_found={best_gain:.8e} "
                    f"neg_inf_streak={controller.auto_hold_neg_inf_streak}/"
                    f"{controller.auto_hold_neg_inf_freeze_after}"
                )
            return

    controller.update(best_kind if commit_ok else None,
                      best_gain, best_state if commit_ok else None)

    if args.rank == 0:
        if commit_ok:
            actual_move = best_kind
        else:
            actual_move = "STAY"
        cur_budget = controller.current_state()
        lump_tag = " [lump-check]" if is_lump_epoch else ""
        pending_tag = ""
        if controller._pending_wins > 0:
            pending_tag = (
                f" [confirm {controller._pending_wins}/"
                f"{controller.required_confirm_rounds()}]"
            )
        print(
            f"  ↳ BudgetCtrl update epoch={epoch}{lump_tag} move={actual_move}{pending_tag} rel_gain={best_gain:.8e} "
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


def _build_merged_edges(args, edge_index_global, num_nodes, final_device, rw_device, nodes_per_rank,
                        edge_seed=None, edge_budget_state=None, adaptive_edge_budget_cfg=None,
                        walk_length_override=None):
    parts = []
    final_torch_device = torch.device(final_device)
    real_edge_sampling_device = _resolve_real_edge_sampling_device(final_torch_device, rw_device)
    real_edges = _resolve_real_edges_for_state(
        args,
        edge_index_global,
        real_edge_sampling_device,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
    )
    if real_edges is not None and real_edges.device != final_torch_device:
        real_edges = real_edges.to(final_torch_device)
    record_scalar("edge_real_edges", int(real_edges.size(1)) if real_edges is not None else 0, reduce="max")
    rw_heads = None
    if _use_rw_edges(args, edge_budget_state, adaptive_edge_budget_cfg):
        wl = _resolve_walk_length(
            args,
            edge_budget_state=edge_budget_state,
            walk_length_override=walk_length_override,
        )
        # Training-side RW build also goes through the shared cache: the
        # built pool is stashed on args and probe will retrieve it
        # later in this epoch without rebuilding.
        rw_heads = _build_or_retrieve_full_rw_pool(
            args,
            edge_index_global=edge_index_global,
            num_nodes=num_nodes,
            rw_device=rw_device,
            walk_length=wl,
            walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            num_heads=args.num_heads,
            edge_seed=edge_seed,
        )
    rw_heads = _resolve_rw_edges_for_state(
        args,
        rw_heads,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
    )
    if rw_heads is not None and rw_heads.device != final_torch_device:
        rw_heads = rw_heads.to(final_torch_device)
    record_scalar("edge_rw_edges", int(rw_heads.size(1)) if rw_heads is not None else 0, reduce="max")
    if real_edges is not None:
        parts.append(real_edges)
    if rw_heads is not None:
        parts.append(rw_heads)
    def _assemble():
        merged_local = _merge_edge_index_list(parts)
        if merged_local is None:
            merged_local = edge_index_global.new_zeros((2, 0), dtype=torch.long).to(final_torch_device)
        if args.model in _GRAPHORMER_VIRTUAL_NODE_MODELS and args.num_global_node > 0:
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


def _broadcast_edges(merged_cpu, device, sp_group, sp_src_rank, sp_rank):
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


def _make_merged_edge_cache_key(
    args,
    edge_index_global: torch.Tensor,
    num_nodes: int,
    rw_build_device,
    resolved_policy: str,
    nodes_per_rank: int,
    edge_seed: int,
    edge_budget_state,
    adaptive_edge_budget_cfg,
    walk_length_override,
) -> tuple:
    """Build a hashable cache key that fully determines the merged edge output."""
    real_budget = _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)
    rw_budget = _get_rw_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)
    walk_length = _resolve_walk_length(args, edge_budget_state, walk_length_override)
    return (
        int(edge_seed),
        int(real_budget),
        int(rw_budget),
        int(walk_length),
        int(getattr(args, "head_hop_walks_per_node", 2)),
        str(rw_build_device),
        str(resolved_policy),
        int(nodes_per_rank),
        int(num_nodes),
        id(edge_index_global),
        int(getattr(args, "num_heads", 1)),
        str(getattr(args, "model", "")),
        int(getattr(args, "num_global_node", 0)),
    )


def _store_merged_edge_cache(key: tuple, merged: torch.Tensor) -> None:
    """Insert *merged* (stored as CPU tensor) into the bounded FIFO cache.

    Capacity rationale (2 slots): the merged-edge cache is consulted on the
    rank-local path of ``_build_and_broadcast_edges`` and only hits when
    foreground reads the slot a prefetch thread just wrote. In normal
    per-epoch-seed training, each epoch produces a fresh key and previous
    epochs' entries never see another hit. We keep slot 0 = "current epoch
    just-built" and slot 1 = "next epoch prefetched", evicting older
    entries. This matches the effective reuse window without holding
    ~1 GiB of unused 200 MiB-each tensors per amazon-scale entry.
    """
    cpu_tensor = merged if merged.device.type == "cpu" else merged.detach().cpu()
    if len(_MERGED_EDGE_CACHE) >= 2:
        _MERGED_EDGE_CACHE.pop(next(iter(_MERGED_EDGE_CACHE)))
    _MERGED_EDGE_CACHE[key] = cpu_tensor


def clear_merged_edge_cache() -> None:
    """Evict all entries; call when edge_index_global is replaced on GPU."""
    _MERGED_EDGE_CACHE.clear()


def _build_and_broadcast_edges(args, edge_index_global, num_nodes, device, rw_device, sp_group,
                               sp_src_rank, sp_rank, nodes_per_rank, edge_seed=None,
                               edge_budget_state=None, adaptive_edge_budget_cfg=None,
                               walk_length_override=None, edge_policy=None,
                               force_broadcast=None):
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
    resolved_policy = _resolve_edge_policy(args, edge_index_global=edge_index_global, edge_policy=edge_policy)
    final_build_device, rw_build_device = _edge_policy_build_devices(resolved_policy, device, rw_device)
    resolved_force_broadcast = _edge_policy_force_broadcast(args, resolved_policy, force_broadcast)

    if edge_seed is not None and not resolved_force_broadcast:
        # Rank-local path: all ranks independently build the same edge_index.
        # fixed_random_seed handles both CPU and CUDA RNG (covers gpu rw_device).
        #
        # Cache: when the same (seed, budget, walk, graph, policy) key recurs
        # (e.g. during static-seed epochs or bootstrap), skip the expensive
        # walk+merge and return the stored CPU tensor moved to the target device.
        cache_key = _make_merged_edge_cache_key(
            args, edge_index_global, num_nodes, rw_build_device, resolved_policy,
            nodes_per_rank, edge_seed, edge_budget_state, adaptive_edge_budget_cfg,
            walk_length_override,
        )
        if cache_key in _MERGED_EDGE_CACHE:
            profile_call("edge_build_local", lambda: None, device=device, payload_bytes=0)
            profile_call("edge_broadcast", lambda: None, device=device, payload_bytes=0)
            cached = _MERGED_EDGE_CACHE[cache_key]
            target = torch.device(final_build_device)
            if cached.device == target:
                return cached
            return cached.to(device=target, dtype=torch.long,
                             non_blocking=(cached.device.type == "cpu"))

        seed_ctx = (
            fixed_random_seed_cpu(edge_seed)
            if torch.device(rw_build_device).type == "cpu"
            else fixed_random_seed(edge_seed)
        )
        with seed_ctx:
            merged = profile_call(
                "edge_build_local",
                lambda: _build_merged_edges(
                    args,
                    edge_index_global,
                    num_nodes,
                    final_build_device,
                    rw_build_device,
                    nodes_per_rank,
                    edge_seed=edge_seed,
                    edge_budget_state=edge_budget_state,
                    adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                    walk_length_override=walk_length_override,
                ),
                device=device,
            )
        _store_merged_edge_cache(cache_key, merged)
        # Record a zero-byte entry so the comm profiler stats remain consistent.
        profile_call("edge_broadcast", lambda: None, device=device, payload_bytes=0)
        return merged

    # ---------------------------------------------------------------------------
    # Fallback: original broadcast path when edge_seed is None or force_broadcast.
    # Cache is applied on src_rank only (non-src ranks never build).
    # ---------------------------------------------------------------------------
    if sp_rank == sp_src_rank:
        _bcast_cache_key = (
            _make_merged_edge_cache_key(
                args, edge_index_global, num_nodes, rw_build_device, resolved_policy,
                nodes_per_rank, edge_seed, edge_budget_state, adaptive_edge_budget_cfg,
                walk_length_override,
            ) if edge_seed is not None else None
        )
        if _bcast_cache_key is not None and _bcast_cache_key in _MERGED_EDGE_CACHE:
            profile_call("edge_build_local", lambda: None, device=device, payload_bytes=0)
            merged = _MERGED_EDGE_CACHE[_bcast_cache_key].to(
                device=device, dtype=torch.long, non_blocking=True
            )
        else:
            merged = profile_call(
                "edge_build_local",
                lambda: _build_merged_edges(
                    args,
                    edge_index_global,
                    num_nodes,
                    final_build_device,
                    rw_build_device,
                    nodes_per_rank,
                    edge_seed=edge_seed,
                    edge_budget_state=edge_budget_state,
                    adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
                    walk_length_override=walk_length_override,
                ),
                device=device,
            )
            if merged.device.type != torch.device(device).type or merged.device != torch.device(device):
                merged = merged.to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=(merged.device.type == "cpu"),
                )
            if _bcast_cache_key is not None:
                _store_merged_edge_cache(_bcast_cache_key, merged)
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
                           walk_length_override=None, edge_policy=None,
                           force_broadcast=None):
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
        edge_policy=edge_policy,
        force_broadcast=force_broadcast,
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
    use_rocauc = str(getattr(args, "dataset", "")).lower() == "genius"

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
    model.eval()
    try:
        with torch.no_grad():
            with _autocast_context(device, amp_dtype):
                out_local = model(x_local, None, edge_index_eval, attn_type=args.attn_type)
            if use_rocauc:
                # Gather class-1 probability for binary ROC AUC
                score_local = out_local.softmax(dim=1)[:, 1].float().contiguous()
            else:
                score_local = out_local.argmax(dim=1)
    finally:
        if was_training:
            model.train()

    if sp_world_size > 1:
        local_len = torch.tensor([score_local.size(0)], dtype=torch.long, device=device)
        len_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(sp_world_size)]
        dist.all_gather(len_list, local_len, group=sp_group)
        pred_lens = [int(t.item()) for t in len_list]
        max_pred_len = max(pred_lens) if pred_lens else 0

        if score_local.size(0) < max_pred_len:
            padded = torch.zeros(max_pred_len, dtype=score_local.dtype, device=device)
            padded[:score_local.size(0)] = score_local
            score_local = padded

        gather_list = [torch.zeros(max_pred_len, dtype=score_local.dtype, device=device) for _ in range(sp_world_size)]
        dist.all_gather(gather_list, score_local, group=sp_group)
        score_chunks = [gather_list[i][:pred_lens[i]] for i in range(sp_world_size)]
        score_global = torch.cat(score_chunks, dim=0)[:num_nodes].cpu()
    else:
        score_global = score_local.cpu()

    result = None
    if args.rank == 0:
        if y is None or split_idx is None:
            raise RuntimeError("Rank 0 requires full labels and split_idx for evaluation.")
        valid_n = min(int(score_global.size(0)), int(num_nodes))
        score_global = score_global[:valid_n]
        y_cpu = y[:valid_n].cpu().view(-1)
        metrics = {}
        if use_rocauc:
            from sklearn.metrics import roc_auc_score
            for split_name, idx in split_idx.items():
                idx_valid = idx[idx < valid_n]
                metrics[split_name] = float(roc_auc_score(
                    y_cpu[idx_valid].numpy(), score_global[idx_valid].numpy()
                ))
        else:
            for split_name, idx in split_idx.items():
                idx_valid = idx[idx < valid_n]
                correct = (score_global[idx_valid].long() == y_cpu[idx_valid]).sum().item()
                metrics[split_name] = float(correct) / max(1, len(idx_valid))
        result = metrics

    del score_local, score_global, out_local, x_local
    if cached_edge_index is None:
        del edge_index_eval
    _cuda_empty_cache(args)
    return result
