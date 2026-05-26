import torch
import torch.distributed as dist
import numpy as np
import random
import contextlib
import os
import sys
import time
import dgl
from typing import List, Optional
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce, remove_self_loops, subgraph, add_self_loops
from gt_sp.initialize import (
    sequence_parallel_is_initialized,
    get_sequence_parallel_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sequence_parallel_src_rank,
    get_sequence_length_per_rank,
    set_global_token_indices,
    get_global_token_indices,
    get_global_token_num,
    set_last_batch_global_token_indices,
    last_batch_flag,
    get_last_batch_flag,
)

_RANDOM_WALK_GRAPH_CACHE = {}
# DGL graph objects built from the coalesced CSR, keyed by (rowptr.data_ptr, col.data_ptr).
# Cached separately so _run_random_walk can look them up without changing the 3-tuple
# returned by _get_random_walk_graph.
_DGL_GRAPH_CACHE: dict = {}


def clear_random_walk_graph_cache() -> None:
    """Evict cached random-walk graph structures after graph storage changes."""
    _RANDOM_WALK_GRAPH_CACHE.clear()
    _DGL_GRAPH_CACHE.clear()


def _compute_local_spd_bias(edge_index: Tensor, n_nodes: int, max_dist: int) -> Tensor:
    """Compute one-hot shortest-path-distance attention bias for a local subgraph (CPU).

    Args:
        edge_index: [2, E] tensor with local 0-indexed node ids (before fix_edge_index).
        n_nodes:    number of nodes in the subgraph.
        max_dist:   max hop distance; distances > max_dist are clamped to the last bin.
                    Output one-hot dimension = max_dist + 1.

    Returns:
        Tensor of shape [n_nodes, n_nodes, max_dist+1], float32, on CPU.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    d = max_dist + 1

    if n_nodes == 0:
        return torch.zeros(0, 0, d, dtype=torch.float32)

    if edge_index.numel() == 0:
        bias = torch.zeros(n_nodes, n_nodes, d, dtype=torch.float32)
        bias[:, :, -1] = 1.0
        for i in range(n_nodes):
            bias[i, i, -1] = 0.0
            bias[i, i, 0] = 1.0
        return bias

    ei = edge_index.cpu().numpy()
    rows = np.concatenate([ei[0], ei[1]])
    cols = np.concatenate([ei[1], ei[0]])
    data_arr = np.ones(len(rows), dtype=np.float32)
    # Guard against out-of-range indices (e.g. if edge_index already has virtual nodes)
    mask = (rows >= 0) & (rows < n_nodes) & (cols >= 0) & (cols < n_nodes)
    rows, cols, data_arr = rows[mask], cols[mask], data_arr[mask]

    adj = csr_matrix((data_arr, (rows, cols)), shape=(n_nodes, n_nodes))
    dist_mat = shortest_path(adj, method='D', directed=False, unweighted=True)  # [n, n], inf → unreachable

    # inf → max_dist (last bin), then clamp to [0, max_dist]
    dist_mat = np.where(np.isinf(dist_mat), float(max_dist), dist_mat)
    dist_mat = np.clip(dist_mat, 0, max_dist)
    dist_int = torch.from_numpy(dist_mat.astype(np.int64))  # [n, n]

    bias = torch.zeros(n_nodes, n_nodes, d, dtype=torch.float32)
    bias.scatter_(2, dist_int.unsqueeze(2), 1.0)
    return bias


def fix_edge_index(x, num_node):
    # Add new edges of virtual nodes
    virt_edges = []

    num_virtual_tokens = 1
    arange_nodes = torch.arange(num_node, device=x.device, dtype=x.dtype)
    for idx in range(num_virtual_tokens):
        virt_edge_index = torch.cat([(arange_nodes + (1 + idx)).view(1, -1), # virtual node index = 0
                                        (x.new_zeros([num_node]) + idx).view(1, -1)], dim=0)
        virt_edges.append(virt_edge_index)

        virt_edge_index = torch.cat([(x.new_zeros([num_node]) + idx).view(1, -1), 
                                    (arange_nodes + (1 + idx)).view(1, -1)], dim=0)
        virt_edges.append(virt_edge_index)

    extra_virt_edges = torch.cat(virt_edges, dim=1)
    x = torch.cat([(x + 1), extra_virt_edges], dim=1) # virtual node index = 0, other nodes start from 1
    return x


def resolve_edge_build_device(args, default_device) -> torch.device:
    """Resolve the device used for edge construction and sampling."""
    spec = getattr(args, "edge_build_device", "same")
    if spec in (None, "", "same"):
        return torch.device(default_device)

    try:
        build_device = torch.device(spec)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            f"Invalid --edge_build_device={spec!r}; expected same/cpu/cuda/cuda:N."
        ) from exc

    if build_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--edge_build_device requests CUDA, but CUDA is not available.")
        if build_device.index is None:
            default_dev = torch.device(default_device)
            if default_dev.type == "cuda" and default_dev.index is not None:
                return torch.device(f"cuda:{default_dev.index}")
            return torch.device(f"cuda:{torch.cuda.current_device()}")

    return build_device


def _estimate_rw_working_set_bytes(
    num_nodes: int,
    num_edges: int,
    walks_per_node: int,
    walk_length: int,
    num_heads: int = 1,
) -> int:
    """First-principles estimate of the GPU working set for one call to
    compute_hop_buckets_random_walk() on a graph of (N, E).

    Each line item below corresponds to a specific allocation in either
    torch_cluster/rw.py (newer COO API) or this module's bucketing loop.
    No multiplicative padding — every byte is traceable to a documented
    operation. The 1-byte boolean tensors are counted explicitly rather
    than approximated as int64.

    Sources:
      - torch_cluster.random_walk:  torch_cluster/rw.py: argsort + reorder
                                    + scatter_add + cumsum to build CSR.
      - thrust::sort auxiliary:     CUB's DeviceRadixSort::SortKeys requires
                                    sizeof(KeyT) × num_items of temporary
                                    storage (https://nvlabs.github.io/cub/
                                    structcub_1_1_device_radix_sort.html).
      - compute_hop_buckets loop:   per-d mask + index_select + stack.

    The PyTorch caching allocator's alignment overhead (≤ 512 B per
    allocation) is dwarfed by the per-tensor sizes here and is omitted.
    """
    E = max(0, int(num_edges))
    N = max(1, int(num_nodes))
    W = max(1, int(walks_per_node))
    L = max(1, int(walk_length))

    # ---------- torch_cluster.random_walk(row, col, ...) body ----------
    # All int64 (8 bytes/element) unless noted.
    #
    # (1) row * num_nodes                          : intermediate, shape (E,)
    sort_step1 = E * 8
    # (2) (row * num_nodes) + col                  : intermediate, shape (E,)
    sort_step2 = E * 8
    # (3) torch.argsort(...) permutation output    : shape (E,)
    argsort_perm = E * 8
    # (4) CUB radix-sort auxiliary storage         : sizeof(int64) × E
    radix_aux = E * 8
    # (5) row[perm], col[perm] (final reordered)   : 2 × shape (E,)
    perm_outputs = 2 * E * 8
    # (6) deg = scatter_add_ on zeros(N)           : shape (N,)
    deg = N * 8
    # (7) rowptr = zeros(N+1) + cumsum             : shape (N+1,)
    rowptr_bytes = (N + 1) * 8
    # (8) walks tensor (random walk kernel output) : shape (N*W, L+1)
    walks_bytes = N * W * (L + 1) * 8

    # ---------- compute_hop_buckets_random_walk per-d loop ----------
    # Live transients during one iteration of d ∈ [1..L] (sequential, so
    # the live peak is for a single d, not the sum):
    #   - valid bool mask    : N*W bytes (1 byte per element)
    #   - kv_nodes[valid]    : ≤ N*W * 8 bytes (worst case all valid)
    #   - query_nodes[valid] : ≤ N*W * 8 bytes
    #   - torch.stack(...)   : 2 × N*W * 8 bytes (the edges tensor)
    valid_mask = N * W * 1
    kv_select = N * W * 8
    query_select = N * W * 8
    stack_edges = 2 * N * W * 8
    per_d_peak = valid_mask + kv_select + query_select + stack_edges

    # ---------- buckets cumulative (across all d) ----------
    # Each hop appends up to N*W edges to the bucket; final bucket holds
    # the sum of all hops' contributions (kept alive until function return).
    bucket_cumulative = 2 * N * W * L * 8

    total = (
        sort_step1 + sort_step2 + argsort_perm + radix_aux
        + perm_outputs + deg + rowptr_bytes + walks_bytes
        + per_d_peak + bucket_cumulative
    )
    return int(total)


# Module-level cache for rw-device decisions to suppress per-rank / per-call
# log spam. Key: (device_type, decision_label); value: previous (estimate_mib,
# free_mib). We only re-print when the decision label flips, not on every
# call with identical inputs.
_RW_DEVICE_DECISION_CACHE: dict = {}

def _clear_rw_device_cache() -> None:
    """Clear the RW-device decision cache so the next call re-evaluates GPU memory."""
    _RW_DEVICE_DECISION_CACHE.clear()


def _expandable_segments_enabled() -> bool:
    """Return True if PYTORCH_CUDA_ALLOC_CONF enables expandable_segments."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return "expandable_segments:True" in conf or "expandable_segments:true" in conf


def _rw_log_is_rank0() -> bool:
    """Rank-0 detection for modules that don't carry args. Falls back to
    True when torch.distributed isn't initialised (single-process runs).
    """
    try:
        import torch.distributed as _dist
        if not _dist.is_available() or not _dist.is_initialized():
            return True
        return _dist.get_rank() == 0
    except Exception:
        return True


def _gpu_concurrent_overhead_bytes(device) -> int:
    """Per-process CUDA library overhead.

    Fixed-ish startup cost for cuBLAS, cuDNN, curand, kernel cache, CUDA
    stream metadata, and optional libs (flash-attention etc.). Independent
    of dataset and only weakly dependent on CUDA / cuDNN version.

    Line-item breakdown (NOT a multiplier):
      - cuBLAS handle + workspace : ~50 MiB / stream × ≤ 4 streams ≤ 200 MiB
        (https://docs.nvidia.com/cuda/cublas/#cublas-context)
      - cuDNN handle + default workspace : up to ~256 MiB
        (https://docs.nvidia.com/deeplearning/cudnn/api/index.html)
      - curand / kernel cache / stream metadata : ~50 MiB combined
      - Headroom for optional libs / future CUDA-version upgrades : ~50-100 MiB
    Bundled conservative ceiling: 512 MiB.

    Note: PyTorch caching allocator fragmentation is *intentionally not*
    subtracted here. ``reserved - allocated`` is memory the allocator
    holds and can hand out to new tensors from its internal pool — it's
    NOT off-limits to RW build. Treating it as overhead would
    double-count and force needless CPU fallbacks at training time
    (when caching allocator naturally holds large reserved blocks across
    step boundaries). If the cached blocks happen to be too fragmented
    to satisfy a large contiguous request, the reactive OOM handler in
    the caller takes over.

    Returns 0 for non-CUDA devices.
    """
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError):
        return 0
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return 0
    return 512 * 1024 * 1024


def _gpu_can_fit_rw(device, need_bytes: int) -> bool:
    """Return True iff *device* currently has room for a need_bytes RW build.

    Decision rule, with no magic multiplier:
      available = free_from_driver + caching_allocator_reserve
      usable    = available - _gpu_concurrent_overhead_bytes(device)
      return need_bytes <= usable

    ``free_from_driver`` is reported by ``cudaMemGetInfo`` and excludes
    everything PyTorch has already cudaMalloc'd. ``caching_allocator_reserve``
    is the slack PyTorch holds from the driver but has not handed to a
    tensor (``reserved - allocated`` in ``memory_stats``). New CUDA
    allocations can be satisfied from EITHER pool, so both are usable.

    Returns True unconditionally for non-CUDA devices or when CUDA's
    ``mem_get_info`` is unavailable, letting the reactive try/except in
    callers catch any residual OOM (e.g., when the caching reserve is
    too fragmented to satisfy a single large contiguous request).
    """
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError):
        return True
    if torch_device.type != "cuda":
        return True
    if not torch.cuda.is_available():
        return True
    try:
        free, _ = torch.cuda.mem_get_info(torch_device)
    except (RuntimeError, AttributeError):
        return True
    # Budget: usable = driver_free [+ reserved_slack] − handle_overhead.
    #
    # Without expandable_segments the caching allocator's reserved-but-not-
    # allocated blocks may be too fragmented to satisfy a single multi-GiB
    # contiguous request, so we conservatively ignore them.
    #
    # With expandable_segments:True the allocator grows existing segments
    # in-place; there are no sub-GiB fragment boundaries for large requests,
    # so the full reserved_slack is reliably usable and we add it back in.
    # This prevents spurious CPU fallbacks after a previous allocation round
    # (e.g. a prior policy profiling run) left a large reserved-but-freed
    # block that the driver hasn't reclaimed yet.
    if _expandable_segments_enabled():
        try:
            reserved_slack = max(
                0,
                torch.cuda.memory_reserved(torch_device) - torch.cuda.memory_allocated(torch_device),
            )
            free = free + reserved_slack
        except Exception:
            pass
    overhead = _gpu_concurrent_overhead_bytes(torch_device)
    usable = max(0, int(free) - overhead)
    return int(need_bytes) <= usable


def _safe_rw_device(rw_dev, args, num_nodes: int, num_edges: int):
    """Decide between GPU and CPU for a single build_head_hop_edges call.

    If the requested rw_dev is CUDA but the estimated RW working set would
    leave too little free memory for the rest of the step, fall back to CPU
    proactively (and log on rank 0). This avoids the partial-allocation-then-
    OOM pattern that fragments the caching allocator before the reactive
    `except RuntimeError` path can take over.
    """
    requested = torch.device(rw_dev)
    if requested.type != "cuda":
        return requested
    estimate = _estimate_rw_working_set_bytes(
        num_nodes=num_nodes,
        num_edges=num_edges,
        walks_per_node=int(getattr(args, "head_hop_walks_per_node", 2)),
        walk_length=int(getattr(args, "head_hop_walk_length", 4)),
        num_heads=int(getattr(args, "num_heads", 1)),
    )
    if _gpu_can_fit_rw(requested, estimate):
        return requested
    rank = int(getattr(args, "rank", 0)) if args is not None else 0
    free_mib = -1.0
    try:
        free, _ = torch.cuda.mem_get_info(requested)
        free_mib = free / (1024 ** 2)
    except Exception:
        pass
    cache_key = ("_safe_rw_device", str(requested), int(num_edges), "cpu")
    prev = _RW_DEVICE_DECISION_CACHE.get(cache_key)
    overhead_mib = _gpu_concurrent_overhead_bytes(requested) / (1024 ** 2)
    usable_mib = max(0.0, free_mib - overhead_mib)
    if prev is None and rank == 0:
        print(
            f"[rw-device] estimated working set {estimate/(1024**2):.0f} MiB "
            f"exceeds usable GPU {usable_mib:.0f} MiB "
            f"(driver_free={free_mib:.0f} − overhead {overhead_mib:.0f} MiB, "
            f"num_edges={num_edges:,}); forcing CPU build to avoid OOM. "
            f"To reduce false CPU fallbacks, "
            f"set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. "
            f"(Further matching calls suppressed.)"
        )
    _RW_DEVICE_DECISION_CACHE[cache_key] = (estimate, free_mib)
    return torch.device("cpu")


def get_seed_batch_size(args) -> int:
    seed_batch_size = getattr(args, "seed_batch_size", None)
    if seed_batch_size is None:
        seed_batch_size = getattr(args, "seq_len")
    return max(1, int(seed_batch_size))


@contextlib.contextmanager
def fixed_random_seed(seed: int):
    """Temporarily set RNG seeds and restore states afterward."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextlib.contextmanager
def fixed_random_seed_cpu(seed: int):
    """Temporarily set CPU-side RNG seeds without touching CUDA RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def adjust_edge_index_nomerge(edge_index, sub_seq_len):
    new_index = edge_index.clone()
    mask = edge_index > 0
    new_index[mask] = edge_index[mask] + ((edge_index[mask].float() - 1) // sub_seq_len).to(torch.int64)

    return new_index


def slice_edge_index(edge_index, start, end):
    if edge_index is None:
        return None
    if edge_index.numel() == 0:
        return edge_index
    mask = (
        (edge_index[0] >= start)
        & (edge_index[0] < end)
        & (edge_index[1] >= start)
        & (edge_index[1] < end)
    )
    if mask.sum().item() == 0:
        return edge_index.new_empty((2, 0), dtype=edge_index.dtype)
    return edge_index[:, mask] - start


def _edge_index_to_ids(edge_index, num_nodes: Optional[int] = None) -> Tensor:
    """Encode edge_index into unique edge ids on CPU for set ops."""
    if edge_index is None:
        return torch.empty((0,), dtype=torch.long)
    if isinstance(edge_index, list):
        tensors = [e for e in edge_index if e is not None and e.numel() > 0]
        if not tensors:
            return torch.empty((0,), dtype=torch.long)
        if num_nodes is None:
            max_node = max(int(e.max().item()) for e in tensors)
            num_nodes = max_node + 1
        edge_index = torch.cat(tensors, dim=1)
    else:
        if edge_index.numel() == 0:
            return torch.empty((0,), dtype=torch.long)
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1

    edge_index = edge_index.to(torch.long).cpu()
    src = edge_index[0]
    dst = edge_index[1]
    ids = src * num_nodes + dst
    return torch.unique(ids)


def edge_index_new_edge_ratio(edge_index_prev, edge_index_next, num_nodes: Optional[int] = None) -> float:
    """
    Ratio of newly added edges from prev -> next.
    Returns 1.0 when prev is empty and next is non-empty, 0.0 when both empty.
    """
    ids_prev = _edge_index_to_ids(edge_index_prev, num_nodes=num_nodes)
    ids_next = _edge_index_to_ids(edge_index_next, num_nodes=num_nodes)
    if ids_prev.numel() == 0:
        return 1.0 if ids_next.numel() > 0 else 0.0
    if ids_next.numel() == 0:
        return 0.0
    new_mask = ~torch.isin(ids_next, ids_prev)
    new_count = int(new_mask.sum().item())
    return float(new_count) / float(ids_prev.numel())


def edge_index_jaccard(edge_index_a, edge_index_b, num_nodes: Optional[int] = None) -> float:
    """Jaccard similarity between two edge sets."""
    ids_a = _edge_index_to_ids(edge_index_a, num_nodes=num_nodes)
    ids_b = _edge_index_to_ids(edge_index_b, num_nodes=num_nodes)
    if ids_a.numel() == 0 and ids_b.numel() == 0:
        return 1.0
    if ids_a.numel() == 0 or ids_b.numel() == 0:
        return 0.0
    inter = torch.isin(ids_a, ids_b).sum().item()
    union = ids_a.numel() + ids_b.numel() - inter
    if union == 0:
        return 1.0
    return float(inter) / float(union)


def pad_y(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full((padlen, ), -100, dtype=x.dtype, device=x.device)
        new_x[:xlen] = x
        x = new_x
    # x = torch.cat([x, torch.full((addlen, ), -100, dtype=x.dtype, device=x.device)], dim=0)
    return x


def pad_2d(x, padlen):
    # xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    # ys = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    # ts = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
    # indexes = torch.stack(indexes, dim=0)

    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x


def pad_attn_bias(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    xlen = x.size(0)
    new_x = x.new_zeros([padlen, padlen, x.size(2)], dtype=x.dtype)
    new_x[:xlen, :xlen, :] = x
    x = new_x
    return x


def pad_2d_bs(x, padlen):
    bs, xlen = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([bs, padlen], dtype=x.dtype)
        new_x[:, :xlen] = x
        x = new_x
    return x


def pad_x_bs(x, padlen):
    bs, xlen2, xlen3 = x.size()
    if xlen2 < padlen:
        new_x = x.new_zeros([bs, padlen, xlen3], dtype=x.dtype)
        new_x[:, :xlen2, :] = x
        x = new_x
    return x


def pad_3d_bs(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    new_x = x.new_zeros([bs, padlen, padlen*seq_parallel_world_size], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3] = x
    x = new_x
    return x


def pad_4d_bs(x, padlen):
    bs, xlen2, xlen3, xlen4 = x.size()
    if xlen2 < padlen:
        new_x = x.new_zeros([bs, padlen, xlen3, xlen4], dtype=x.dtype)
        new_x[:, :xlen2, :, :] = x
        x = new_x
    return x


def pad_5d_bs(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3, xlen4, xlen5 = x.size()
    new_x = x.new_zeros([bs, padlen, padlen*seq_parallel_world_size, xlen4, xlen5], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3, :, :] = x
    x = new_x
    return x


def pad_attn_bias_bs(x, padlen): 
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    new_x = x.new_zeros([bs, padlen, (padlen-1)*seq_parallel_world_size+1], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3] = x
    x = new_x
    return x


def pad_attn_bias_bs_unsplit(x, padlen, graph_node_num):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    if xlen2 < padlen*seq_parallel_world_size+1:
        # new_x = x.new_zeros([bs, padlen*seq_parallel_world_size+1, padlen*seq_parallel_world_size+1], dtype=x.dtype)
        # new_x[:, :xlen2, :xlen3] = x
        # x = new_x

        # Pad "-inf"
        new_x = x.new_zeros([bs, padlen*seq_parallel_world_size+1, padlen*seq_parallel_world_size+1], dtype=x.dtype).fill_(float("-inf"))
        new_x[:, :xlen2, :xlen3] = x
        
        for i in range(graph_node_num.size(0)):
            new_x[i, xlen2:, :graph_node_num[i]] = 0
        x = new_x
        # if get_sequence_parallel_rank() == 0:
        #     print(x[3, :, :])
        # exit(0)
    return x


def pad_spatial_pos_bs_unsplit(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3 = x.size()
    new_x = x.new_zeros([bs, padlen*seq_parallel_world_size, padlen*seq_parallel_world_size], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3] = x
    x = new_x
    return x


def pad_edge_input_bs_unsplit(x, padlen):
    seq_parallel_world_size = get_sequence_parallel_world_size()
    bs, xlen2, xlen3, xlen4, xlen5 = x.size()
    new_x = x.new_zeros([bs, padlen*seq_parallel_world_size, padlen*seq_parallel_world_size, xlen4, xlen5], dtype=x.dtype)
    new_x[:, :xlen2, :xlen3, :, :] = x
    x = new_x
    return x


def random_split_idx(data_y, frac_train, frac_valid, frac_test, seed):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    random.seed(seed)
    all_idx = np.arange(data_y.shape[0])
    random.shuffle(all_idx)
    train_idx = all_idx[:int(frac_train * data_y.shape[0])]
    val_idx = all_idx[int(frac_train * data_y.shape[0]):int((frac_train+frac_valid) * data_y.shape[0])]
    test_idx = all_idx[int((frac_train+frac_valid) * data_y.shape[0]):]
    split_idx = {'train': torch.tensor(train_idx),
                'valid': torch.tensor(val_idx),
                'test': torch.tensor(test_idx)}
    # print(f"Train nodes: {len(train_idx)}, Valid nodes: {len(val_idx)}, Test nodes: {len(test_idx)}")
    return split_idx


def gen_sub_edge_index(edge_index, idx_batch, N):
    """
    Get sub edge_index according to given sequence nodes
        
    Arguments:
        edge_index (Tensor): original edge_index of the whole graph
        idx_batch (Tensor): training node indexes of a batch
        N (Int): number of nodes in the whole graph
    """
    adj = edge_index
    edge_index_i, _ = subgraph(idx_batch, adj, num_nodes=N, relabel_nodes=True)

    return edge_index_i


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def reformat_graph(edge_index, k):
    src, dst = edge_index
    g = dgl.graph((src, dst))
    t0 = time.time()
    with suppress_stdout():
        partition_ids = dgl.metis_partition_assignment(g, k)
    t1 = time.time()
    new_id_mapping = np.empty(g.num_nodes(), dtype=np.int64)
    current_id = 0
    for part_id in range(k):
        nodes_in_part = np.where(partition_ids == part_id)[0]
        new_id_mapping[nodes_in_part] = np.arange(current_id, current_id + len(nodes_in_part))
        current_id += len(nodes_in_part)

    t2 = time.time()

    src, dst = edge_index
    new_edge_index = torch.stack(
        [
            torch.from_numpy(new_id_mapping[src.numpy()]),
            torch.from_numpy(new_id_mapping[dst.numpy()]),
        ],
        dim=0,
    )
    t3 = time.time()
    new_id_mapping_tensor = torch.from_numpy(new_id_mapping)

    sorted_indices = torch.argsort(new_id_mapping_tensor)
    sorted_indices_edge = torch.argsort(new_edge_index[0])
    sorted_edge_index = new_edge_index[:, sorted_indices_edge]
    t4 = time.time()
    # print(f"Time in reorder {t1 - t0} {t2 - t1} {t3 - t2} {t4 - t3}")
    return sorted_edge_index, sorted_indices


def get_batch(args, x, y, idx_batch, adjs, rest_split_sizes, device):
    """Generate a local subsequence in sequence parallel
    """

    # For sequence parallel
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    seq_length = args.seq_len

    assert seq_length % seq_parallel_world_size == 0
    sub_seq_length = seq_length // seq_parallel_world_size
    sub_seq_start = seq_parallel_world_rank * sub_seq_length
    sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

    x_i = x[idx_batch]
    y_i = y[idx_batch]
    attn_bias = torch.cat([torch.tensor(i[idx_batch.cpu(), :][:, idx_batch.cpu()].toarray(), dtype=torch.float32).unsqueeze(0) for i in adjs])
    attn_bias = attn_bias.permute(1, 2, 0) # [s, s, d]

    if idx_batch.shape[0] < seq_length:
        
        assert rest_split_sizes is not None, 'split_sizes should not be None'
        x_i_list = [t for t in torch.split(x_i, rest_split_sizes, dim=0)]
        y_i_list = [t for t in torch.split(y_i, rest_split_sizes, dim=0)]
        attn_bias_list = [t for t in torch.split(attn_bias, rest_split_sizes, dim=0)]

        padlen = max(rest_split_sizes)
        x_i_list_pad = []
        y_i_list_pad = []
        attn_bias_list_pad = []
        for i in range(len(x_i_list)):
            x_i_list_pad.append(pad_2d(x_i_list[i], padlen))
            y_i_list_pad.append(pad_y(y_i_list[i], padlen))
            attn_bias_list_pad.append(pad_attn_bias(attn_bias_list[i], padlen))
        last_batch_flag(True)
        
        return x_i_list_pad[seq_parallel_world_rank].to(device), y_i_list_pad[seq_parallel_world_rank].to(device), attn_bias_list_pad[seq_parallel_world_rank].to(device)
    
    else:
        x_i = x_i[sub_seq_start:sub_seq_end, :].to(device)
        y_i = y_i[sub_seq_start:sub_seq_end].to(device)
        attn_bias = attn_bias[sub_seq_start:sub_seq_end, :, :].to(device) 
        last_batch_flag(False)
        
        return x_i, y_i, attn_bias
    

def _apply_to_edge_index(edge_index, fn):
    if isinstance(edge_index, list):
        return [fn(e) if e is not None else None for e in edge_index]
    if edge_index is None:
        return None
    return fn(edge_index)


def _merge_edge_index_list(edge_list: List[Tensor]) -> Tensor:
    """Merge a list of edge_index tensors into one unique edge_index."""
    tensors = [e for e in edge_list if e is not None and e.numel() > 0]
    if not tensors:
        return edge_list[0] if edge_list else None
    if len(tensors) == 1:
        return tensors[0]
    device = tensors[0].device
    edge_index = torch.cat(tensors, dim=1).to(torch.long)
    num_nodes = int(max(edge_index[0].max().item(), edge_index[1].max().item())) + 1
    return coalesce(edge_index, num_nodes=num_nodes).to(device)


def _get_random_walk_graph(edge_index: Tensor, num_nodes: int, device) -> tuple[Tensor, Tensor, Tensor]:
    """Return (placeholder, rowptr, col) for the given edge_index on *device*.

    Only rowptr and col (CSR) are cached — the COO edge_index_dev is NOT
    stored because col is identical to the coalesced edge_index[1] and row
    can be reconstructed from rowptr on demand (saves ~2×E×8 bytes per cache
    entry, ≈ 4 GiB on amazon).  Callers receive a zero-shape placeholder
    tensor for ``edge_index_dev`` so that ``.new_zeros()`` and device-pinning
    still work; ``_run_random_walk`` uses only rowptr/col for the fast CSR
    path and reconstructs row from rowptr for the rare COO fallback.

    If a CPU CSR is already cached and *device* is CUDA, rowptr and col are
    H2D-transferred instead of rebuilt from scratch (avoids the 6×E×8-byte
    sort/argsort pipeline).
    """
    dev = torch.device(device)
    key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(num_nodes), dev.type, dev.index)
    cached = _RANDOM_WALK_GRAPH_CACHE.get(key)
    if cached is not None:
        return cached

    # GPU target: if CPU CSR already built, H2D-transfer rowptr+col only.
    if dev.type == "cuda":
        cpu_key = (int(edge_index.data_ptr()), int(edge_index.size(1)), int(num_nodes), "cpu", None)
        cpu_cached = _RANDOM_WALK_GRAPH_CACHE.get(cpu_key)
        if cpu_cached is not None:
            _, rowptr_cpu, col_cpu = cpu_cached
            rowptr = rowptr_cpu.to(dev)
            col = col_cpu.to(dev)
            placeholder = torch.empty((2, 0), dtype=edge_index.dtype, device=dev)
            cached = (placeholder, rowptr, col)
            _RANDOM_WALK_GRAPH_CACHE[key] = cached
            return cached

    # Build CSR from scratch on the target device.
    edge_index_dev = edge_index if edge_index.device == dev else edge_index.to(dev)
    adj = SparseTensor(
        row=edge_index_dev[0],
        col=edge_index_dev[1],
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce()
    rowptr, col, _ = adj.csr()
    # Store a placeholder instead of edge_index_dev — see docstring.
    placeholder = torch.empty((2, 0), dtype=edge_index.dtype, device=dev)
    cached = (placeholder, rowptr, col)
    _RANDOM_WALK_GRAPH_CACHE[key] = cached
    return cached


def _map_global_nodes_to_local(nodes_global: Tensor, sampled_nodes: Tensor) -> Tensor:
    sampled_nodes = sampled_nodes.to(torch.long)
    nodes_global = nodes_global.to(torch.long)
    if nodes_global.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=sampled_nodes.device)
    sorted_nodes, order = torch.sort(sampled_nodes)
    sorted_local = torch.arange(sampled_nodes.numel(), dtype=torch.long, device=sampled_nodes.device)[order]
    pos = torch.searchsorted(sorted_nodes, nodes_global)
    pos = torch.clamp(pos, max=max(int(sorted_nodes.numel()) - 1, 0))
    return sorted_local[pos]


def sample_seed_rw_subgraph(
    edge_index: Tensor,
    seed_nodes: Tensor,
    num_nodes: int,
    device,
    walk_length: int,
    walks_per_node: int,
) -> tuple[Tensor, Tensor]:
    """Sample a seed-centric subgraph by random walks on the full graph.

    The returned sampled nodes keep seeds first, followed by visited context nodes.
    Edge semantics follow attention message flow: src provides k/v, dst is the query.
    We keep both walk trajectory edges (curr -> prev) and direct seed-query edges
    (visited -> seed).
    """
    seed_nodes = seed_nodes.to(torch.long).view(-1).cpu()
    if seed_nodes.numel() == 0:
        return seed_nodes, edge_index.new_zeros((2, 0), dtype=torch.long).cpu()

    walk_length = max(1, int(walk_length))
    walks_per_node = max(1, int(walks_per_node))
    rw_device = torch.device(device)
    edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)

    seed_nodes_dev = seed_nodes.to(rw_device)
    starts = seed_nodes_dev.repeat_interleave(walks_per_node)
    walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length)
    starts_walk = walks[:, 0]

    visited_nodes = [seed_nodes_dev]
    edge_parts: List[Tensor] = []
    for d in range(1, walk_length + 1):
        curr = walks[:, d]
        prev = walks[:, d - 1]
        valid = (curr >= 0) & (prev >= 0)
        if not valid.any():
            continue
        curr_valid = curr[valid]
        prev_valid = prev[valid]
        seed_valid = starts_walk[valid]
        edge_parts.append(torch.stack([curr_valid, prev_valid], dim=0))
        edge_parts.append(torch.stack([curr_valid, seed_valid], dim=0))
        visited_nodes.append(curr_valid)

    if len(visited_nodes) > 1:
        context_nodes = torch.unique(torch.cat(visited_nodes[1:], dim=0))
        context_nodes = context_nodes[~torch.isin(context_nodes, seed_nodes_dev)]
        sampled_nodes = torch.cat([seed_nodes_dev, context_nodes], dim=0)
    else:
        sampled_nodes = seed_nodes_dev

    if edge_parts:
        sampled_edges_global = _merge_edge_index_list(edge_parts)
        edge_src_local = _map_global_nodes_to_local(sampled_edges_global[0], sampled_nodes)
        edge_dst_local = _map_global_nodes_to_local(sampled_edges_global[1], sampled_nodes)
        sampled_edge_index = torch.stack([edge_src_local, edge_dst_local], dim=0)
    else:
        sampled_edge_index = edge_index.new_zeros((2, 0), dtype=torch.long, device=rw_device)

    return sampled_nodes.cpu(), sampled_edge_index.cpu()


def compute_hops_random_walk(
    edge_index: Tensor,
    num_nodes: int,
    device: str = "cpu",
    walk_length: int = 6,
    walks_per_node: int = 2,
    num_groups: int = 4,
) -> List[Tensor]:
    """
    使用随机游走为不同 head 组生成不同的子图。
    每个组内的边来自该组的随机游走序列 (seed_node -> d-step node)。
    后处理会尽量避免回退与重复节点。
    """
    walk_length = max(1, int(walk_length))
    num_groups = max(1, int(num_groups))
    if edge_index.numel() == 0:
        return [edge_index for _ in range(num_groups)]

    inferred_nodes = int(edge_index.max().item()) + 1
    num_nodes = max(num_nodes, inferred_nodes)
    rw_device = torch.device(device)
    edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)

    starts = torch.arange(num_nodes, device=rw_device, dtype=torch.long)
    starts = starts.repeat_interleave(walks_per_node * num_groups)

    walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length)
    walks = walks.view(num_groups, -1, walk_length + 1)

    group_edges: List[List[Tensor]] = [[] for _ in range(num_groups)]
    seen = torch.zeros(num_nodes, device=rw_device, dtype=torch.bool)
    for d in range(1, walk_length + 1):
        g = (d - 1) % num_groups
        walks_g = walks[g]
        src = walks_g[:, 0]
        dst = walks_g[:, d]
        valid = dst >= 0
        if d > 1:
            prev = walks_g[:, d - 1]
            valid = valid & (dst != prev)
        if not valid.any():
            continue
        dst_valid = dst[valid]
        src_valid = src[valid]
        unseen = ~seen[dst_valid]
        if not unseen.any():
            continue
        dst_keep = dst_valid[unseen]
        src_keep = src_valid[unseen]
        seen[dst_keep] = True
        group_edges[g].append(torch.stack([src_keep, dst_keep], dim=0))

    buckets: List[Tensor] = []
    for edges in group_edges:
        if not edges:
            buckets.append(edge_index_dev.new_zeros((2, 0), dtype=edge_index_dev.dtype))
        else:
            buckets.append(torch.cat(edges, dim=1))
    return buckets


def compute_hop_buckets_random_walk(
    edge_index: Tensor,
    num_nodes: int,
    num_buckets: int,
    device: str = "cpu",
    walk_length: int = 4,
    walks_per_node: int = 2,
    min_hop: int = 1,
) -> List[Tensor]:
    """
    使用随机游走生成 hop buckets，bucket i 对应 hop i+1，
    最后一个 bucket 合并 >= num_buckets 的 hop.
    设置 min_hop=2 可跳过 1-hop 真实邻居，使 RW 池只包含远距节点。
    """
    num_buckets = max(1, int(num_buckets))
    walk_length = max(1, int(walk_length))
    walks_per_node = max(1, int(walks_per_node))
    min_hop = max(1, int(min_hop))
    if edge_index.numel() == 0:
        return [edge_index.new_zeros((2, 0), dtype=edge_index.dtype) for _ in range(num_buckets)]

    inferred_nodes = int(edge_index.max().item()) + 1
    num_nodes = max(num_nodes, inferred_nodes)
    rw_device = torch.device(device)

    # Proactive GPU-OOM guard. torch_cluster.random_walk internally allocates
    # ~4 × E × 8 bytes for the (row * num_nodes + col, argsort, permute)
    # pipeline, which dominates working set on dense graphs (amazon ≈ 264M
    # edges → ~8 GiB peak). Falling back to CPU here saves all caller paths
    # — multi_tier edge-policy profiling, probe, and training step — from
    # the partial-allocation-then-OOM pattern that fragments the allocator.
    #
    # Safety margin: reserve 10% of currently-free GPU for non-RW transients
    # (model forward, autograd graph, NCCL buffers). Logs are gated on
    # rank 0 and deduplicated against the previous decision so a stable
    # CPU-fallback regime does not flood stdout.
    if rw_device.type == "cuda":
        E_cur = int(edge_index.size(1))
        # The dominant GPU cost is building the CSR (SparseTensor.coalesce +
        # argsort) — 6 × E × 8 bytes on large graphs.  After the first call,
        # _get_random_walk_graph caches the CSR on CPU and can H2D transfer
        # rowptr + col to GPU without any sorting.  In that case, only the
        # walk tensors + transient H2D buffers need to fit, not the full
        # sort pipeline.  Use a reduced estimate when the CPU CSR is cached.
        _cpu_csr_key = (int(edge_index.data_ptr()), E_cur, num_nodes, "cpu", None)
        if _cpu_csr_key in _RANDOM_WALK_GRAPH_CACHE:
            N_cur = max(1, int(num_nodes))
            W = max(1, int(walks_per_node))
            L = max(1, int(walk_length))
            col_h2d = E_cur * 8          # H2D col array (transient, largest piece)
            estimate = int(
                col_h2d
                + N_cur * W * (L + 1) * 8    # walks tensor
                + N_cur * W * 33              # per-d peak (mask+2×select+stack)
                + 2 * N_cur * W * L * 8       # bucket_cumulative
            )
        else:
            estimate = _estimate_rw_working_set_bytes(
                num_nodes=num_nodes,
                num_edges=E_cur,
                walks_per_node=walks_per_node,
                walk_length=walk_length,
            )
        if not _gpu_can_fit_rw(rw_device, estimate):
            try:
                free, _ = torch.cuda.mem_get_info(rw_device)
                free_mib = free / (1024 ** 2)
            except Exception:
                free_mib = -1.0
            cache_key = ("compute_hop_buckets_random_walk", str(rw_device), E_cur, "cpu")
            prev = _RW_DEVICE_DECISION_CACHE.get(cache_key)
            overhead_mib = _gpu_concurrent_overhead_bytes(rw_device) / (1024 ** 2)
            usable_mib = max(0.0, free_mib - overhead_mib)
            if prev is None and _rw_log_is_rank0():
                print(
                    f"[rw-device] compute_hop_buckets_random_walk: estimated "
                    f"working set {estimate/(1024**2):.0f} MiB > usable GPU "
                    f"{usable_mib:.0f} MiB (driver_free={free_mib:.0f} − "
                    f"overhead {overhead_mib:.0f} MiB), "
                    f"num_edges={E_cur:,}; forcing CPU build. NOTE: this overrides "
                    f"only the RW build device; topology policies such as gpu_persist "
                    f"may still cache edge_index_global on GPU. "
                    f"(Further matching calls suppressed.)"
                )
            _RW_DEVICE_DECISION_CACHE[cache_key] = (estimate, free_mib)
            rw_device = torch.device("cpu")

    edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)

    starts = torch.arange(num_nodes, device=rw_device, dtype=torch.long)
    starts = starts.repeat_interleave(walks_per_node)
    try:
        walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as _gpu_walk_exc:
        if rw_device.type != "cuda":
            raise
        # GPU walk failed (OOM or COO fallback unavailable with placeholder
        # edge_index from the CPU→GPU H2D cache path).  Evict the stale GPU
        # CSR entry, clear the device decision cache so the next epoch
        # re-evaluates, and retry the whole walk on CPU.
        _bad_key = (
            int(edge_index.data_ptr()), int(edge_index.size(1)),
            int(num_nodes), rw_device.type, rw_device.index,
        )
        _RANDOM_WALK_GRAPH_CACHE.pop(_bad_key, None)
        _RW_DEVICE_DECISION_CACHE.clear()
        if _rw_log_is_rank0():
            print(
                f"[rw-device] GPU walk failed ({type(_gpu_walk_exc).__name__}); "
                f"evicting GPU CSR cache and retrying on CPU."
            )
        rw_device = torch.device("cpu")
        edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)
        starts = starts.to(rw_device)
        walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length)

    buckets = [edge_index_dev.new_zeros((2, 0), dtype=edge_index_dev.dtype) for _ in range(num_buckets)]
    query_nodes = walks[:, 0]
    for d in range(min_hop, walk_length + 1):
        kv_nodes = walks[:, d]
        valid = kv_nodes >= 0
        if d > 1:
            prev = walks[:, d - 1]
            valid = valid & (kv_nodes != prev)
        if not valid.any():
            continue
        query_nodes_valid = query_nodes[valid]
        kv_nodes_valid = kv_nodes[valid]
        # edge_index[0] provides key/value nodes and edge_index[1] is the updated query node.
        edges = torch.stack([kv_nodes_valid, query_nodes_valid], dim=0)
        bucket_idx = min(d - 1, num_buckets - 1)
        buckets[bucket_idx] = torch.cat([buckets[bucket_idx], edges], dim=1)
    return buckets


def build_head_hop_edges(
    edge_index: Tensor,
    num_nodes: int,
    num_heads: int,
    num_groups: int,
    device: str = "cpu",
    walk_length: int = 4,
    walks_per_node: int = 2,
    min_hop: int = 1,
):
    """
    构建 per-head edge_index 列表。
    当前实现不再按 head 分桶：所有 head 共享同一份随机游走子图。
    返回单个共享 edge_index，避免调用方把同一张图复制多份后再去重。
    min_hop=2 可使 RW 池只包含 2-hop 以上的远距节点。
    """
    _ = num_heads
    _ = num_groups  # kept for backward-compatible call signature
    buckets = compute_hop_buckets_random_walk(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_buckets=1,
        device=device,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
        min_hop=min_hop,
    )
    return buckets[0] if buckets else edge_index.new_zeros((2, 0), dtype=edge_index.dtype)


def _run_random_walk(edge_index: Tensor, rowptr: Tensor, col: Tensor, starts: Tensor, walk_length: int) -> Tensor:
    # DGL path: better CPU parallelism (OpenMP) for CPU-resident graphs.
    # Only used on CPU from the main thread; background prefetch threads fall
    # through to torch_cluster.  DGL's OpenMP workers and its internal CUDA
    # context accesses are not safe when called concurrently with GPU training
    # on the main thread, causing deadlocks.  torch_cluster is thread-safe and
    # respects PyTorch's CPU RNG, so prefetch walks remain deterministic.
    if rowptr.device.type == "cpu":
        import threading as _threading
        if _threading.current_thread() is _threading.main_thread():
            try:
                import dgl as _dgl
                # Draw from PyTorch's CPU RNG instead of using initial_seed().
                # initial_seed() stays constant after manual_seed(), so reseeding
                # DGL with it on every call makes CPU DGL walks repeat across
                # batches/epochs.  Drawing a seed preserves fixed_random_seed*
                # determinism while allowing normal training walks to evolve.
                dgl_seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
                _dgl.seed(dgl_seed)
                key = (int(rowptr.data_ptr()), int(col.data_ptr()))
                g = _DGL_GRAPH_CACHE.get(key)
                if g is None:
                    num_nodes = int(rowptr.numel()) - 1
                    try:
                        # DGL ≥ 1.0: build directly from CSR (rowptr, col).
                        # Avoids the ~E×8-byte repeat_interleave(arange(N)) src
                        # tensor and DGL stores only CSR internally — saves ~2 GiB
                        # persistent memory vs the COO path on ogbn-products scale.
                        g = _dgl.from_csr(rowptr.long(), col.long(), edge_ids=None)
                    except (TypeError, AttributeError):
                        # Older DGL: fall back to COO construction.
                        src = torch.repeat_interleave(
                            torch.arange(num_nodes, dtype=torch.long),
                            rowptr[1:] - rowptr[:-1],
                        )
                        g = _dgl.graph((src, col.long()), num_nodes=num_nodes, device="cpu")
                    _DGL_GRAPH_CACHE[key] = g
                traces, _ = _dgl.sampling.random_walk(g, starts.long(), length=walk_length)
                # DGL fills dead-end steps with -1; torch_cluster repeats the
                # current node instead.  Replace -1 with the previous node so
                # that dead-end handling is consistent between both code paths
                # and the resulting walk distribution matches on directed graphs
                # (e.g. snap-patents has many sink nodes with out-degree 0).
                for _step in range(1, traces.shape[1]):
                    _dead = traces[:, _step] < 0
                    if not _dead.any():
                        break
                    traces[_dead, _step] = traces[_dead, _step - 1]
                return traces
            except Exception:
                pass  # fall through to torch_cluster

    try:
        from torch_cluster import random_walk
    except Exception as exc:
        raise RuntimeError("torch_cluster.random_walk not available") from exc
    # torch_cluster.random_walk API has two historical forms:
    #   - older (≤ 1.5):  (rowptr, col, start, walk_length)  — CSR, faster,
    #                     reuses our pre-built CSR pointers directly.
    #   - newer (≥ 1.6):  (row, col, start, walk_length)     — COO, does
    #                     argsort(row*N + col) to rebuild CSR internally.
    # Try CSR first to preserve the fast path on older installs. On newer
    # torch_cluster the CSR call raises a shape-mismatch error (rowptr has
    # length N+1, col has length E, broadcast fails); fall back to COO.
    # IMPORTANT: catch torch.cuda.OutOfMemoryError separately and re-raise
    # so a real OOM isn't masked by the COO fallback.
    try:
        return random_walk(rowptr, col, starts, walk_length)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError) as exc:
        msg = str(exc)
        if "must match" in msg or "size" in msg or "expected" in msg:
            # Newer torch_cluster (≥ 1.6) requires COO.  edge_index is now
            # always a zero-shape placeholder, so reconstruct row from rowptr.
            n_nodes = int(rowptr.numel()) - 1
            row = torch.repeat_interleave(
                torch.arange(n_nodes, dtype=rowptr.dtype, device=rowptr.device),
                rowptr[1:] - rowptr[:-1],
            )
            return random_walk(row, col, starts, walk_length)
        raise


def compute_group_nodes_random_walk(
    edge_index: Tensor,
    num_nodes: int,
    device: str = "cpu",
    walk_length: int = 6,
    walks_per_node: int = 2,
    num_groups: int = 4,
    max_nodes_per_group: int = 2048,
) -> List[Tensor]:
    """
    使用随机游走生成每个组的节点序列，组内去重，组间可重叠。
    """
    walk_length = max(1, int(walk_length))
    num_groups = max(1, int(num_groups))
    max_nodes_per_group = max(1, int(max_nodes_per_group))
    if edge_index.numel() == 0:
        return [edge_index.new_zeros((0,), dtype=torch.long) for _ in range(num_groups)]

    inferred_nodes = int(edge_index.max().item()) + 1
    num_nodes = max(int(num_nodes), inferred_nodes)
    rw_device = torch.device(device)
    edge_index_dev, rowptr, col = _get_random_walk_graph(edge_index, num_nodes, rw_device)

    group_nodes: List[Tensor] = []
    all_nodes = torch.arange(num_nodes, device=rw_device, dtype=torch.long)
    for g in range(num_groups):
        seeds = all_nodes[g::num_groups]
        if seeds.numel() > max_nodes_per_group:
            seeds = seeds[:max_nodes_per_group]
        if seeds.numel() == 0:
            group_nodes.append(seeds)
            continue
        starts = seeds.repeat_interleave(walks_per_node)
        walks = _run_random_walk(edge_index_dev, rowptr, col, starts, walk_length)
        seen = torch.zeros(num_nodes, device=rw_device, dtype=torch.bool)
        seen[seeds] = True
        nodes_g = [seeds]
        total = int(seeds.numel())
        for d in range(1, walk_length + 1):
            if total >= max_nodes_per_group:
                break
            dst = walks[:, d]
            valid = dst >= 0
            if d > 1:
                prev = walks[:, d - 1]
                valid = valid & (dst != prev)
            if not valid.any():
                continue
            dst_valid = dst[valid]
            unseen = ~seen[dst_valid]
            if not unseen.any():
                continue
            dst_keep = dst_valid[unseen]
            remaining = max_nodes_per_group - total
            if dst_keep.numel() > remaining:
                dst_keep = dst_keep[:remaining]
            seen[dst_keep] = True
            nodes_g.append(dst_keep)
            total += int(dst_keep.numel())
        group_nodes.append(torch.cat(nodes_g) if len(nodes_g) > 1 else seeds)
    return group_nodes


class SubgraphSampler:
    """Base sampler that returns a subgraph given the current batch nodes."""

    def sample(self, edge_index: Tensor, seed_nodes: Tensor, num_nodes: int) -> Tensor:
        raise NotImplementedError


class IdentitySubgraphSampler(SubgraphSampler):
    """Return the incoming edge_index unchanged."""

    def sample(self, edge_index: Tensor, seed_nodes: Tensor, num_nodes: int) -> Tensor:
        return edge_index


class EdgeDropSubgraphSampler(SubgraphSampler):
    """Randomly drop edges with a fixed ratio."""

    def __init__(self, drop_ratio: float) -> None:
        self.drop_ratio = max(0.0, min(1.0, float(drop_ratio)))

    def sample(self, edge_index: Tensor, seed_nodes: Tensor, num_nodes: int) -> Tensor:
        if edge_index.numel() == 0 or self.drop_ratio <= 0:
            return edge_index
        keep_mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_ratio
        # Avoid empty graphs
        if not keep_mask.any():
            return edge_index
        return edge_index[:, keep_mask]


class HeadSubgraphProvider:
    """Produce per-head edge_index views."""

    def build(self, edge_index: Tensor, num_heads: int):
        raise NotImplementedError


class SharedHeadSubgraphProvider(HeadSubgraphProvider):
    """All heads share the same edge_index."""

    def build(self, edge_index, num_heads: int):
        if isinstance(edge_index, list) and len(edge_index) > 0:
            edge_index = edge_index[0]
        return [edge_index for _ in range(num_heads)]


class GroupedHeadSubgraphProvider(HeadSubgraphProvider):
    """
    Split edges into N groups (configurable), then assign each head to a group.
    Heads in the same group share the same subgraph.
    """

    def __init__(self, num_groups: int) -> None:
        self.num_groups = max(1, int(num_groups))

    def build(self, edge_index, num_heads: int):
        if isinstance(edge_index, list) and len(edge_index) > 0:
            edge_index = edge_index[0]
        if edge_index.numel() == 0:
            return [edge_index for _ in range(num_heads)]

        groups = min(self.num_groups, num_heads)
        heads_per_group = (num_heads + groups - 1) // groups

        group_edges = [edge_index.new_zeros((2, 0), dtype=edge_index.dtype) for _ in range(groups)]
        src = edge_index[0].to(torch.long)
        buckets = [[] for _ in range(groups)]
        for idx in range(edge_index.size(1)):
            g = int(src[idx].item() % groups)
            buckets[g].append(idx)
        for g in range(groups):
            if buckets[g]:
                group_edges[g] = edge_index[:, buckets[g]]

        head_edges: List[Tensor] = []
        for h in range(num_heads):
            g = min(h // heads_per_group, groups - 1)
            head_edges.append(group_edges[g])
        return head_edges


class HopMappedHeadSubgraphProvider(HeadSubgraphProvider):
    """
    Map head groups to pre-defined hop buckets. Buckets由外部传入的列表确定。
    """

    def __init__(self, hop_sequence: List[int], num_groups: int) -> None:
        self.hop_sequence = hop_sequence
        self.num_groups = max(1, int(num_groups))

    def build(self, edge_buckets: List[Tensor], num_heads: int):
        if not isinstance(edge_buckets, list) or len(edge_buckets) == 0:
            return [edge_buckets for _ in range(num_heads)]

        groups = min(self.num_groups, num_heads)
        heads_per_group = (num_heads + groups - 1) // groups

        # 准备组子图
        group_edges: List[Tensor] = []
        for g in range(groups):
            hop_idx = self.hop_sequence[g % len(self.hop_sequence)]
            if hop_idx < len(edge_buckets):
                bucket = edge_buckets[hop_idx]
            else:
                bucket = edge_buckets[0]
            if bucket is None:
                bucket = edge_buckets[0] if edge_buckets else None
            group_edges.append(bucket)

        head_edges: List[Tensor] = []
        for h in range(num_heads):
            g = min(h // heads_per_group, groups - 1)
            head_edges.append(group_edges[g])
        return head_edges


def build_subgraph_sampler(args) -> SubgraphSampler:
    if getattr(args, "subgraph_sampler", "identity") == "edge_drop":
        return EdgeDropSubgraphSampler(getattr(args, "edge_drop_ratio", 0.0))
    return IdentitySubgraphSampler()


def build_head_subgraph_provider(args) -> HeadSubgraphProvider:
    provider = getattr(args, "head_subgraph_provider", "hop")
    num_groups = getattr(args, "head_groups", 4)
    if provider == "hop":
        hop_sequence = list(range(max(1, int(num_groups))))
        return HopMappedHeadSubgraphProvider(hop_sequence, num_groups)
    return SharedHeadSubgraphProvider()

def get_batch_blockize(
    args,
    x,
    y,
    idx_batch,
    rest_split_sizes,
    edge_index,
    N,
    device=None,
    seed_label_mask=None,
    force_induced_edges: bool = False,
    apply_graphormer_virtual_edges: bool = True,
):
    """
    Generate a local subsequence in sequence parallel and slice the corresponding edge_index.
    """
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    seq_length = args.seq_len
    batch_mode = getattr(args, "batch_subgraph_mode", "induced")
    model_name = str(getattr(args, "model", "")).lower()
    dynamic_split_sizes = None
    dev = None
    sampled_nodes = None
    induced_edge_index_i_raw = None
    rw_edge_index_i_raw = None
    if device is not None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device

    if force_induced_edges:
        x_i = x[idx_batch]
        y_i = y[idx_batch]
        induced_edge_index_i_raw = gen_sub_edge_index(edge_index, idx_batch, N)
    elif batch_mode == "seed_rw":
        if seq_parallel_world_size > 1:
            src_rank = get_sequence_parallel_src_rank()
            group = get_sequence_parallel_group()
        else:
            src_rank = None
            group = None

        do_sample = (seq_parallel_world_size <= 1) or (dist.get_rank() == src_rank)
        if do_sample:
            sampled_nodes, rw_edge_index_i_raw = sample_seed_rw_subgraph(
                edge_index=edge_index,
                seed_nodes=idx_batch,
                num_nodes=N,
                device=resolve_edge_build_device(args, device if device is not None else edge_index.device),
                walk_length=getattr(args, "head_hop_walk_length", 4),
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            )
        else:
            sampled_nodes = None
            rw_edge_index_i_raw = None

        if seq_parallel_world_size > 1:
            assert dev is not None, "seed_rw broadcasting requires a concrete device in sequence parallel mode."
            broad_device = dev
            if do_sample:
                sizes_broad = torch.tensor(
                    [
                        int(sampled_nodes.size(0)),
                        int(rw_edge_index_i_raw.size(1)),
                    ],
                    device=broad_device,
                    dtype=torch.long,
                )
            else:
                sizes_broad = torch.empty(2, device=broad_device, dtype=torch.long)
            dist.broadcast(sizes_broad, src_rank, group=group)

            sampled_node_count = int(sizes_broad[0].item())
            sampled_edge_count = int(sizes_broad[1].item())
            if do_sample:
                sampled_nodes_broad = sampled_nodes.to(broad_device)
                edge_index_broad = rw_edge_index_i_raw.to(broad_device)
            else:
                sampled_nodes_broad = torch.empty(sampled_node_count, device=broad_device, dtype=torch.long)
                edge_index_broad = torch.empty((2, sampled_edge_count), device=broad_device, dtype=torch.long)
            dist.broadcast(sampled_nodes_broad, src_rank, group=group)
            dist.broadcast(edge_index_broad, src_rank, group=group)
            sampled_nodes = sampled_nodes_broad.cpu()
            rw_edge_index_i_raw = edge_index_broad.cpu()

        x_i = x[sampled_nodes]
        y_i = y.new_full((sampled_nodes.size(0),) + tuple(y.shape[1:]), -100)
        seed_labels = y[idx_batch].clone()
        if seed_label_mask is not None:
            seed_mask_cpu = seed_label_mask.to(torch.bool).cpu()
            seed_labels[~seed_mask_cpu] = -100
        y_i[:idx_batch.shape[0]] = seed_labels
        induced_edge_index_i_raw = gen_sub_edge_index(edge_index, sampled_nodes, N)

        split_sizes = [t.shape[0] for t in torch.tensor_split(torch.zeros(sampled_nodes.size(0)), seq_parallel_world_size, dim=0)]
        dynamic_split_sizes = split_sizes
        sub_real_seq_len = max(split_sizes) + args.num_global_node
        set_last_batch_global_token_indices(
            list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
        )
    else:
        x_i = x[idx_batch]  # [s, x_d]
        y_i = y[idx_batch]  # [s]
        # Keep raw (0-indexed, before fix_edge_index) subgraph for local_spd and RW sampling.
        induced_edge_index_i_raw = gen_sub_edge_index(edge_index, idx_batch, N)

    # --- attn_bias (optional, controlled by args.attn_bias_mode) ---
    _attn_bias_mode = getattr(args, 'attn_bias_mode', 'none')
    full_attn_bias = None  # [n_real, n_real, d] on CPU (before SP split)
    if _attn_bias_mode == 'local_spd':
        _n_real = x_i.shape[0]
        _max_dist = getattr(args, 'attn_bias_max_dist', 5)
        if _n_real > 16384:
            import warnings
            warnings.warn(
                f"[local_spd] seq_len={_n_real} is large; BFS for {_n_real}x{_n_real} may be slow. "
                "Consider using --attn_bias_mode none or reducing --seq_len."
            )
        full_attn_bias = _compute_local_spd_bias(
            induced_edge_index_i_raw.cpu(), n_nodes=_n_real, max_dist=_max_dist,
        )  # [n_real, n_real, d]

    attn_bias = None
    if force_induced_edges:
        edge_index_i_heads = induced_edge_index_i_raw
    elif model_name == "gt":
        edge_parts = [induced_edge_index_i_raw]
        if batch_mode == "seed_rw":
            if rw_edge_index_i_raw is not None and rw_edge_index_i_raw.numel() > 0:
                edge_parts.append(rw_edge_index_i_raw)
            edge_index_i_heads = _merge_edge_index_list(edge_parts)
        else:
            rw_base = induced_edge_index_i_raw
            rw_dev = resolve_edge_build_device(args, dev if dev is not None else rw_base.device)
            # Proactive GPU-OOM guard: if the estimated RW working set won't
            # fit in the current free GPU pool with a 15% safety margin,
            # build on CPU instead of OOM'ing and fragmenting the allocator.
            rw_dev = _safe_rw_device(
                rw_dev, args, num_nodes=x_i.size(0),
                num_edges=int(rw_base.size(1)) if rw_base is not None else 0,
            )
            if rw_base.device != rw_dev:
                rw_base = rw_base.to(rw_dev)
            try:
                rw_edge_index_i_raw = build_head_hop_edges(
                    edge_index=rw_base,
                    num_nodes=x_i.size(0),
                    num_heads=args.num_heads,
                    num_groups=1,
                    device=rw_base.device,
                    walk_length=getattr(args, "head_hop_walk_length", 4),
                    walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
                )
            except torch.cuda.OutOfMemoryError:
                # Pre-check under-estimated; fall back to CPU.
                if int(getattr(args, "rank", 0)) == 0:
                    print("[rw-device] pre-check passed but build still OOM'd; retrying on CPU.")
                rw_base = rw_base.to("cpu")
                torch.cuda.empty_cache()
                rw_edge_index_i_raw = build_head_hop_edges(
                    edge_index=rw_base,
                    num_nodes=x_i.size(0),
                    num_heads=args.num_heads,
                    num_groups=1,
                    device=rw_base.device,
                    walk_length=getattr(args, "head_hop_walk_length", 4),
                    walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
                )
            if rw_edge_index_i_raw is not None and rw_edge_index_i_raw.numel() > 0:
                edge_parts.append(rw_edge_index_i_raw)
            edge_index_i_heads = _merge_edge_index_list(edge_parts)
    elif batch_mode == "seed_rw":
        edge_index_i_heads = rw_edge_index_i_raw
    else:
        rw_base = induced_edge_index_i_raw
        rw_dev = resolve_edge_build_device(args, dev if dev is not None else rw_base.device)
        # Proactive GPU-OOM guard — see comment above.
        rw_dev = _safe_rw_device(
            rw_dev, args, num_nodes=x_i.size(0),
            num_edges=int(rw_base.size(1)) if rw_base is not None else 0,
        )
        if rw_base.device != rw_dev:
            rw_base = rw_base.to(rw_dev)
        try:
            edge_index_i_heads = build_head_hop_edges(
                edge_index=rw_base,
                num_nodes=x_i.size(0),
                num_heads=args.num_heads,
                num_groups=1,
                device=rw_base.device,
                walk_length=getattr(args, "head_hop_walk_length", 4),
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            )
        except torch.cuda.OutOfMemoryError:
            if int(getattr(args, "rank", 0)) == 0:
                print("[rw-device] pre-check passed but build still OOM'd; retrying on CPU.")
            rw_base = rw_base.to("cpu")
            torch.cuda.empty_cache()
            edge_index_i_heads = build_head_hop_edges(
                edge_index=rw_base,
                num_nodes=x_i.size(0),
                num_heads=args.num_heads,
                num_groups=1,
                device=rw_base.device,
                walk_length=getattr(args, "head_hop_walk_length", 4),
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            )

    if args.model == "graphormer" and apply_graphormer_virtual_edges:
        edge_index_i_heads = _apply_to_edge_index(
            edge_index_i_heads,
            lambda e: fix_edge_index(e, x_i.shape[0]),
        )

    # Current head-hop implementation shares the same subgraph across heads.
    # Merge to one tensor early to avoid per-head broadcast/memory overhead.
    if isinstance(edge_index_i_heads, list):
        edge_index_i_heads = _merge_edge_index_list(edge_index_i_heads)

    if dev is not None and edge_index_i_heads.device != dev:
        edge_index_i_heads = edge_index_i_heads.to(dev)
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
        if dist.get_rank() == src_rank:
            size_broad = torch.tensor([edge_index_i_heads.size(1)], device=edge_index_i_heads.device, dtype=torch.long)
        else:
            size_broad = torch.empty(1, device=edge_index_i_heads.device, dtype=torch.long)
        dist.broadcast(size_broad, src_rank, group=group)
        if dist.get_rank() != src_rank:
            edge_index_i_heads = torch.empty((2, int(size_broad.item())), device=edge_index_i_heads.device, dtype=edge_index_i_heads.dtype)
        dist.broadcast(edge_index_i_heads, src_rank, group=group)

    if force_induced_edges:
        split_sizes = rest_split_sizes if idx_batch.shape[0] < seq_length else None
    elif batch_mode == "seed_rw":
        split_sizes = dynamic_split_sizes
    else:
        split_sizes = rest_split_sizes if idx_batch.shape[0] < seq_length else None

    if split_sizes is not None:
        seq_length = max(split_sizes) * seq_parallel_world_size
        x_i = pad_2d(x_i, seq_length)
        y_i = pad_y(y_i, seq_length)

        afterpad_split_sizes = [max(split_sizes)] * seq_parallel_world_size
        x_i_list = [t for t in torch.split(x_i, afterpad_split_sizes, dim=0)]
        y_i_list = [t for t in torch.split(y_i, afterpad_split_sizes, dim=0)]

        if args.model == "graphormer" and apply_graphormer_virtual_edges:
            edge_index_i_heads = _apply_to_edge_index(
                edge_index_i_heads,
                lambda e: adjust_edge_index_nomerge(e, max(split_sizes)),
            )

        x_i = x_i_list[seq_parallel_world_rank]
        y_i = y_i_list[seq_parallel_world_rank]
        sub_seq_start = seq_parallel_world_rank * max(split_sizes)
        sub_seq_end = sub_seq_start + x_i.size(0)

        if full_attn_bias is not None:
            # Pad [n_real, n_real, d] → [seq_length, seq_length, d] (zeros for pad nodes)
            # Use direct padding instead of pad_attn_bias() which requires SP group init
            n_real, _, _d = full_attn_bias.shape
            if n_real < seq_length:
                padded = full_attn_bias.new_zeros([seq_length, seq_length, _d])
                padded[:n_real, :n_real, :] = full_attn_bias
                attn_bias = padded
            else:
                attn_bias = full_attn_bias
            # Slice rows for this rank: [sub_s, full_padded_s, d]
            attn_bias = attn_bias[sub_seq_start:sub_seq_end, :, :]
            if dev is not None:
                attn_bias = attn_bias.to(dev)

        last_batch_flag(True)
    else:
        assert seq_length % seq_parallel_world_size == 0
        sub_seq_length = seq_length // seq_parallel_world_size
        sub_seq_start = seq_parallel_world_rank * sub_seq_length
        sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

        x_i = x_i[sub_seq_start:sub_seq_end, :]
        y_i = y_i[sub_seq_start:sub_seq_end]
        if args.model == "graphormer" and apply_graphormer_virtual_edges:
            edge_index_i_heads = _apply_to_edge_index(
                edge_index_i_heads,
                lambda e: adjust_edge_index_nomerge(e, sub_seq_length),
            )

        if full_attn_bias is not None:
            # Slice rows for this rank: [sub_s, full_s, d]
            attn_bias = full_attn_bias[sub_seq_start:sub_seq_end, :, :]
            if dev is not None:
                attn_bias = attn_bias.to(dev)

        last_batch_flag(False)

    return (x_i, y_i, edge_index_i_heads, attn_bias)


def get_batch_reorder_blockize(args, x, y, idx_batch, rest_split_sizes, device, edge_index, N, k=0):
    """
    Generate a local subsequence in sequence parallel with optional edge reordering.
    """
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
    seq_length = args.seq_len

    x_i = x[idx_batch]
    y_i = y[idx_batch]

    edge_index_i_raw = gen_sub_edge_index(edge_index, idx_batch, N)

    if args.model == "graphormer":
        edge_index_i_raw = fix_edge_index(edge_index_i_raw, idx_batch.shape[0])

    if k > 1:
        if args.rank == 0:
            edge_index_i, sorted_indices = reformat_graph(edge_index_i_raw, k)
            sizes_broad = torch.LongTensor([edge_index_i.shape[1], sorted_indices.shape[0]]).to(device)
        else:
            sizes_broad = torch.empty(2, dtype=torch.int64, device=device)
        if seq_parallel_world_size > 1:
            dist.barrier()
            dist.broadcast(sizes_broad, src_rank, group=group)

        if args.rank == 0:
            edge_index_i_broad = edge_index_i.to(device)
            sorted_indices_broad = sorted_indices.to(device)
        else:
            shape = sizes_broad.tolist()
            edge_index_i_broad = torch.empty((2, shape[0]), device=device, dtype=torch.int64)
            sorted_indices_broad = torch.empty((shape[1]), device=device, dtype=torch.int64)
        if seq_parallel_world_size > 1:
            dist.broadcast(edge_index_i_broad, src_rank, group=group)
            dist.broadcast(sorted_indices_broad, src_rank, group=group)
        edge_index_i = edge_index_i_broad.to("cpu")
        sorted_indices = sorted_indices_broad.to("cpu")
    else:
        edge_index_i = edge_index_i_raw
        sorted_indices = torch.arange(x_i.size(0), device=x_i.device, dtype=torch.long)

    if args.model == "graphormer":
        sorted_indices = sorted_indices[sorted_indices != 0]
        sorted_indices = sorted_indices - 1
        attn_bias = None
    else:
        attn_bias = None
    x_i = torch.index_select(x_i, 0, sorted_indices)
    y_i = torch.index_select(y_i, 0, sorted_indices)
    edge_index_i_heads = edge_index_i

    if idx_batch.shape[0] < seq_length:
        assert rest_split_sizes is not None, "Rest split_sizes should not be None"
        seq_length = max(rest_split_sizes) * seq_parallel_world_size
        x_i = pad_2d(x_i, seq_length)
        y_i = pad_y(y_i, seq_length)

        afterpad_split_sizes = [max(rest_split_sizes)] * seq_parallel_world_size
        x_i_list = [t for t in torch.split(x_i, afterpad_split_sizes, dim=0)]
        y_i_list = [t for t in torch.split(y_i, afterpad_split_sizes, dim=0)]

        if args.model == "graphormer":
            edge_index_i_heads = _apply_to_edge_index(
                edge_index_i_heads,
                lambda e: adjust_edge_index_nomerge(e, max(rest_split_sizes)),
            )

        x_i = x_i_list[seq_parallel_world_rank]
        y_i = y_i_list[seq_parallel_world_rank]
        sub_seq_start = seq_parallel_world_rank * max(rest_split_sizes)
        sub_seq_end = sub_seq_start + x_i.size(0)
        edge_index_i_heads = _apply_to_edge_index(
            edge_index_i_heads,
            lambda e: slice_edge_index(e, sub_seq_start, sub_seq_end),
        )

        if attn_bias is not None:
            attn_bias = pad_attn_bias(attn_bias, seq_length)
            attn_bias_list = [t for t in torch.split(attn_bias, afterpad_split_sizes, dim=0)]
            attn_bias = attn_bias_list[seq_parallel_world_rank]

        last_batch_flag(True)
    else:
        assert seq_length % seq_parallel_world_size == 0
        sub_seq_length = seq_length // seq_parallel_world_size
        sub_seq_start = seq_parallel_world_rank * sub_seq_length
        sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

        x_i = x_i[sub_seq_start:sub_seq_end, :]
        y_i = y_i[sub_seq_start:sub_seq_end]
        edge_index_i_heads = _apply_to_edge_index(
            edge_index_i_heads,
            lambda e: slice_edge_index(e, sub_seq_start, sub_seq_end),
        )

        if attn_bias is not None:
            attn_bias = attn_bias[sub_seq_start:sub_seq_end, :, :]

        if args.model == "graphormer":
            edge_index_i_heads = _apply_to_edge_index(
                edge_index_i_heads,
                lambda e: adjust_edge_index_nomerge(e, sub_seq_length),
            )

        last_batch_flag(False)

    return (x_i, y_i, edge_index_i_heads, attn_bias)


def get_batch_papers100m(args, x, y, idx_batch, attn_bias, rest_split_sizes, device, edge_index, N):
    """
    Dummy bias for faster processing time each iteration
    Generate a local subsequence in sequence parallel
    Get sub edge_index according to sequence length
    """
    # For sequence parallel
    seq_length = args.seq_len
    sub_seq_length = seq_length // args.world_size

    x_i = x[idx_batch] # [s, x_d]
    y_i = y[idx_batch] # [s]

    # Get sub edge_index according to given sequence nodes
    edge_index_i = gen_sub_edge_index(edge_index, idx_batch, N) # NOTE: make sure all rank share the same edge_index_i
    
    if args.model == "graphormer":
        # Fix edge index: add new edges of virtual nodes
        edge_index_i = fix_edge_index(edge_index_i, idx_batch.shape[0])

    if idx_batch.shape[0] < seq_length:
        # 对于剩余的node，feature用0 pad, label用-100 pad, attn_bias用0填充，
        # 先把总的pad到新长度（为了将pad的点都放在最后一个rank），然后划分
        
        assert rest_split_sizes is not None, 'split_sizes should not be None'
        seq_length = max(rest_split_sizes) * args.world_size # 14 * 4 = 56 
        x_i = pad_2d(x_i, seq_length)
        y_i = pad_y(y_i, seq_length)

        if args.model == "graphormer":
            # Adjust edge index values w/o merging global token after all2all 
            edge_index_i = adjust_edge_index_nomerge(edge_index_i, max(rest_split_sizes))   
    else:
        if args.model == "graphormer":
            edge_index_i = adjust_edge_index_nomerge(edge_index_i, sub_seq_length)

    return (x_i, y_i, attn_bias, edge_index_i)


def get_batch_from_loader(args, batch):
    """Generate a batch of local subsequences from dataloader in sequence parallel.
    cut in seq-level: x, attn_edge_type, spatial_pos, in-degree, out-degree, edge_input.
    attn_bias, edge_index copy in each rank.
    Set global token indices for each batch.
    """  
    #### For sequence parallel
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    
    seq_length = batch.x.size(1)

    #### Split input data to each rank: if attn_bias use all2all, need to split attn_bias, spatial_pos, edge_input
    sub_split_seq_lens = batch.sub_split_seq_lens # [9, 9, 9, 8]
    # if args.rank == 0:
    #     print(f'This batch seq len: {seq_length}, sub seq len: {sub_seq_lengths}')

    x_i_list = [t for t in torch.split(batch.x, sub_split_seq_lens, dim=1)]
    in_degree_list = [t for t in torch.split(batch.in_degree, sub_split_seq_lens, dim=1)]
    out_degree_list = [t for t in torch.split(batch.out_degree, sub_split_seq_lens, dim=1)]
    
    # # Attn_bias all2all need
    # spatial_pos_list = [t for t in torch.split(batch.spatial_pos, sub_split_seq_lens, dim=1)]
    # edge_input_list = [t for t in torch.split(batch.edge_input, sub_split_seq_lens, dim=1)]
    # attn_biases = [t for t in torch.split(batch.attn_bias[:, 1:, :], sub_split_seq_lens, dim=1)]
    # attn_bias_list = [torch.cat([torch.index_select(batch.attn_bias, 1, torch.LongTensor([0])), t], dim=1) for t in attn_biases] # add glotal token
    # attn_edge_type_list = [t for t in torch.split(batch.attn_edge_type, sub_seq_lengths, dim=1)]
    
    #### Pad cut data to the same sub seq length. e.g., [9, 9, 9, 8] -> [9, 9, 9, 9]
    padlen = max(sub_split_seq_lens)
    sub_real_seq_len = padlen + args.num_global_node
    
    x_i_list_pad = [pad_x_bs(t, padlen) for t in x_i_list]
    in_degree_list_pad = [pad_2d_bs(t, padlen) for t in in_degree_list]
    out_degree_list_pad = [pad_2d_bs(t, padlen) for t in out_degree_list]
    # attn_edge_type_list_pad = [pad_4d_bs(t, padlen) for t in attn_edge_type_list]
    # spatial_pos_list_pad = [pad_3d_bs(t, padlen) for t in spatial_pos_list]
    # edge_input_list_pad = [pad_5d_bs(t, padlen) for t in edge_input_list]
    # attn_bias_list_pad = [pad_attn_bias_bs(t, sub_real_seq_len) for t in attn_bias_list]

    batch.x = x_i_list_pad[seq_parallel_world_rank] # [bs, padlen, 1]
    batch.in_degree = in_degree_list_pad[seq_parallel_world_rank]
    batch.out_degree = out_degree_list_pad[seq_parallel_world_rank]
    # batch.attn_edge_type = attn_edge_type_list_pad[seq_parallel_world_rank]
    # batch.spatial_pos = spatial_pos_list_pad[seq_parallel_world_rank]
    # batch.edge_input = edge_input_list_pad[seq_parallel_world_rank]
    # batch.attn_bias = attn_bias_list_pad[seq_parallel_world_rank]
    # print(f"{batch.spatial_pos.size()} {batch}")
    if args.dummy_bias:
        batch.attn_bias = None

    #### Unsplit data
    # batch.attn_bias = pad_attn_bias_bs_unsplit(batch.attn_bias, padlen, batch.graph_node_num)
    # batch.spatial_pos = pad_spatial_pos_bs_unsplit(batch.spatial_pos, padlen)
    # batch.edge_input = pad_edge_input_bs_unsplit(batch.edge_input, padlen)

    # Set global token indices for each batch, in graphormer global token idx: 0
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    # set_global_token_indices(global_token_indices)
    batch.global_token_indices = global_token_indices

    del batch.sub_split_seq_lens      
    
    
def get_batch_from_loader_malnet(args, batch):
    """Generate a batch of local subsequences from dataloader in sequence parallel.
    cut in seq-level: x, attn_edge_type, spatial_pos, in-degree, out-degree, edge_input.
    attn_bias, edge_index copy in each rank.
    Set global token indices for each batch.
    """  
    #### For sequence parallel
    seq_parallel_world_size = args.world_size
    seq_length = batch.x.size(1)

    sub_split_seq_lens = batch.sub_split_seq_lens # [9, 9, 9, 8]
    x_i_list = [t for t in torch.split(batch.x, sub_split_seq_lens, dim=1)]
    in_degree_list = [t for t in torch.split(batch.in_degree, sub_split_seq_lens, dim=1)]
    out_degree_list = [t for t in torch.split(batch.out_degree, sub_split_seq_lens, dim=1)]
    
    #### Pad cut data to the same sub seq length. e.g., [9, 9, 9, 8] -> [9, 9, 9, 9]
    padlen = max(sub_split_seq_lens)
    sub_real_seq_len = padlen + args.num_global_node
    
    x_i_list_pad = [pad_x_bs(t, padlen) for t in x_i_list]
    in_degree_list_pad = [pad_2d_bs(t, padlen) for t in in_degree_list]
    out_degree_list_pad = [pad_2d_bs(t, padlen) for t in out_degree_list]
    # attn_edge_type_list_pad = [pad_4d_bs(t, padlen) for t in attn_edge_type_list]
    # spatial_pos_list_pad = [pad_3d_bs(t, padlen) for t in spatial_pos_list]
    # edge_input_list_pad = [pad_5d_bs(t, padlen) for t in edge_input_list]
    # attn_bias_list_pad = [pad_attn_bias_bs(t, sub_real_seq_len) for t in attn_bias_list]

    batch.x = x_i_list_pad # [bs, padlen, 1]
    batch.in_degree = in_degree_list_pad
    batch.out_degree = out_degree_list_pad
    # batch.attn_edge_type = attn_edge_type_list_pad[seq_parallel_world_rank]
    # batch.spatial_pos = spatial_pos_list_pad[seq_parallel_world_rank]
    # batch.edge_input = edge_input_list_pad[seq_parallel_world_rank]
    # batch.attn_bias = attn_bias_list_pad[seq_parallel_world_rank]
    # print(f"{batch.spatial_pos.size()} {batch}")
    if args.dummy_bias:
        batch.attn_bias = None

    #### Unsplit data
    # batch.attn_bias = pad_attn_bias_bs_unsplit(batch.attn_bias, padlen, batch.graph_node_num)
    # batch.spatial_pos = pad_spatial_pos_bs_unsplit(batch.spatial_pos, padlen)
    # batch.edge_input = pad_edge_input_bs_unsplit(batch.edge_input, padlen)

    # Set global token indices for each batch, in graphormer global token idx: 0
    global_token_indices = list(range(0, seq_parallel_world_size * sub_real_seq_len, sub_real_seq_len))
    # set_global_token_indices(global_token_indices)
    batch.global_token_indices = global_token_indices
    
    return (x_i_list_pad, in_degree_list_pad, out_degree_list_pad, global_token_indices)
    

def split_tensor_along_second_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """ Split a tensor along its second dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # global_token_num = get_global_token_num()
    
    # Get the size and dimension.
    split_dim = 1
    split_dim_size = tensor.size()[split_dim] // num_partitions
        
    # Split.
    tensor_list = torch.split(tensor, split_dim_size, dim=split_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def merge_global_token(x_layer: Tensor, merge_dim: int) -> Tensor:
    """Merge each rank's global token embedding into one for k, q, v

    Arguments:
        x_layer (Tensor): input tensor

    Returns:    
        * output (Tensor): merged output tensor
    """
    # TODO consider multiple global tokens case; maybe slow?
    if not get_last_batch_flag():
        global_token_indices = get_global_token_indices(last_batch=False)
    else:
        global_token_indices = get_global_token_indices(last_batch=True)
        
    avg_global_token = torch.mean(torch.index_select(x_layer, merge_dim, 
                                                       torch.LongTensor(global_token_indices).to(x_layer.device)), dim=merge_dim, keepdim=True)
    x_layer = torch.index_select(x_layer, merge_dim, 
                                 torch.LongTensor([i for i in range(x_layer.size(merge_dim)) if i not in global_token_indices]).to(x_layer.device))
    
    x_layer = torch.cat([x_layer, avg_global_token], dim=merge_dim).contiguous()
    return x_layer


def merge_global_token0(x_layer: Tensor, merge_dim: int) -> Tensor:
    """Merge each rank's global token embedding into one for k, q, v

    Arguments:
        x_layer (Tensor): input tensor

    Returns:    
        * output (Tensor): merged output tensor
    """
    # TODO consider multiple global tokens case; maybe slow?
    if not get_last_batch_flag():
        global_token_indices = get_global_token_indices(last_batch=False)
    else:
        global_token_indices = get_global_token_indices(last_batch=True)
        
    # if get_sequence_parallel_rank() == 0:
    #     a = torch.index_select(x_layer, merge_dim, torch.LongTensor(global_token_indices).to(x_layer.device))
    #     print(a.view(4, 4, -1))

    # [bs, 1, np, hn]
    avg_global_token = torch.mean(torch.index_select(x_layer, merge_dim, 
                                                       torch.LongTensor(global_token_indices).to(x_layer.device)), dim=merge_dim, keepdim=True)

    x_layer = torch.index_select(x_layer, merge_dim, 
                                 torch.LongTensor([i for i in range(x_layer.size(merge_dim)) if i not in global_token_indices]).to(x_layer.device))
    x_layer = torch.cat([avg_global_token, x_layer], dim=merge_dim).contiguous()
    return x_layer


def extend_global_token(x_layer: Tensor, extend_dim: int) -> Tensor:
    """Extend global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # TODO consider multiple global tokens case
    global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()
    
    x_layer_list = split_tensor_along_second_dim(x_layer, seq_world_size)
        
    # Split.
    output_list = [torch.cat([x_layer_list[i], x_layer_list[-1]], dim=extend_dim).contiguous() for i in range(seq_world_size)]
    
    return torch.cat(output_list, dim=extend_dim).contiguous()


def extend_global_token0(x_layer: Tensor, extend_dim: int) -> Tensor:
    """Extend global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # TODO consider multiple global tokens case
    # x_layer: [b, s+1, hp] 
    global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()
    
    assert (x_layer.size(extend_dim) - 1) % seq_world_size == 0
    split_sizes = [1] + [(x_layer.size(extend_dim) - 1) // seq_world_size] * seq_world_size

    # Split.
    tensor_list = torch.split(x_layer, split_sizes, dim=extend_dim)    
    output_list = [torch.cat([tensor_list[0], tensor_list[i]], dim=extend_dim).contiguous() for i in range(1, seq_world_size+1)]
    
    return torch.cat(output_list, dim=extend_dim).contiguous()


def copy_global_token0(x_layer: Tensor, extend_dim: int) -> Tensor:
    """copy global token embedding to each rank

    Arguments:
        x_layer (Tensor): input tensor

    Returns:
        * output (Tensor): output tensor
    """
    # x_layer: [b, s+p, hp] 
    # global_token_num = get_global_token_num()
    seq_world_size = get_sequence_parallel_world_size()

    if not get_last_batch_flag():
        global_token_indices = get_global_token_indices(last_batch=False)
    else:
        global_token_indices = get_global_token_indices(last_batch=True)

    assert x_layer.size(extend_dim) % seq_world_size == 0
    x_layer[:, torch.LongTensor(global_token_indices), :] = x_layer[:, 0, :].unsqueeze(1).repeat(1, len(global_token_indices), 1)
    
    return x_layer.contiguous()


def broadcast_data(args, dataset_train, batch, device):
    world_size = get_sequence_parallel_world_size()
    rank = get_sequence_parallel_rank()
    group = get_sequence_parallel_group()
    src_rank = get_sequence_parallel_src_rank()

    # Pack on rank zero
    batch_idx = batch.idx

    if rank == 0:
        flatten_idx = batch_idx.to('cuda')
    else:
        total_numel = batch_idx.numel()
        flatten_idx = torch.empty(total_numel,
                                device=device,
                                dtype=torch.int64)
    # Broadcast
    dist.broadcast(flatten_idx, src_rank, group=group)


    # sampler = SubsetRandomSampler(flatten_idx.tolist())
    # dataloader = DataLoader(
    #             dataset_train,
    #             batch_size=args.batch_size,
    #             num_workers=args.num_workers,
    #             pin_memory=True,
    #             sampler=sampler,
    #             collate_fn=partial(
    #                 collator,
    #                 max_node=get_dataset(args.dataset)["max_node"],
    #                 multi_hop_max_dist=args.multi_hop_max_dist,
    #                 spatial_pos_max=args.spatial_pos_max,
    #             )
    # )
    # # batch = next(iter(dataloader))
    # # print(f"rank {rank} {batch.y}")
    # # for batch in dataloader:
    # #     print(f"rank {rank} {batch.y}")
    # # exit(0)
    # return next(iter(dataloader))


def calc_power_edge_index(edge_index, N, power):
    values = torch.ones(edge_index.size(1), dtype=torch.float32)
    adj_matrix = torch.sparse_coo_tensor(edge_index, values, torch.Size([N, N]))

    m_adj = adj_matrix.clone()
    for _ in range(power - 1):
        m_adj = torch.sparse.mm(m_adj, adj_matrix) 

    sparse_rate = calculate_sparsity(m_adj)

    coalesced_matrix = m_adj.coalesce()
    
    m_edge_index = coalesced_matrix.indices()
    return m_edge_index


def calculate_sparsity(sparse_matrix):
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]

    non_zero_elements = sparse_matrix._nnz()

    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity


def calculate_sparsity_csr(csr_matrix):
    non_zero = csr_matrix.nnz

    total_elements = csr_matrix.shape[0] * csr_matrix.shape[1]

    sparsity = 1 - (non_zero / total_elements)
    return sparsity


def make_strongly_connected(graph):
    if nx.is_weakly_connected(graph):
        print("The graph is already strongly connected.")
        return graph
    
    scc = list(nx.strongly_connected_components(graph))
    
    if len(scc) == 1:
        return graph

    for i in range(len(scc) - 1):
        comp_a = scc[i]
        comp_b = scc[i + 1]
        node_a = next(iter(comp_a))
        node_b = next(iter(comp_b))
        graph.add_edge(node_a, node_b)

    scc = list(nx.strongly_connected_components(graph))
    while len(scc) > 1:
        comp_a = scc[0]
        comp_b = scc[1]
        node_a = next(iter(comp_a))
        node_b = next(iter(comp_b))
        graph.add_edge(node_a, node_b)
        scc = list(nx.strongly_connected_components(graph))

    return graph


def check_conditions(edge_index, num_nodes):
    graph = nx.DiGraph()
    
    nodes = range(num_nodes)
    for node in nodes:
        graph.add_node(node)
        
    edges = list(zip(edge_index[0], edge_index[1]))
    graph.add_edges_from(edges)

    for node in graph.nodes:
        if not graph.has_edge(node, node):
            print(f"Condition 1 failed: Node {node} does not attend to itself.")
            return False

    # graph = make_strongly_connected(graph)
    if not nx.is_weakly_connected(graph):
        print("Condition 3 failed: The graph is not strongly connected.")
        return False

    print("All conditions are satisfied.")
    return True
