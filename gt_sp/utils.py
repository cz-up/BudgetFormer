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
from torch_geometric.utils import remove_self_loops, subgraph, add_self_loops
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
    last_batch_flag,
    get_last_batch_flag,
)


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
    device = tensors[0].device
    edge_index = torch.cat(tensors, dim=1).to(torch.long)
    src = edge_index[0]
    dst = edge_index[1]
    num_nodes = int(max(src.max().item(), dst.max().item())) + 1
    ids = src * num_nodes + dst
    ids_unique = torch.unique(ids)
    src_u = ids_unique // num_nodes
    dst_u = ids_unique % num_nodes
    return torch.stack([src_u, dst_u], dim=0).to(device)


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
    edge_index = edge_index.to(device)

    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)).coalesce()
    rowptr, col, _ = adj.csr()

    starts = torch.arange(num_nodes, device=device, dtype=torch.long)
    starts = starts.repeat_interleave(walks_per_node * num_groups)

    walks = _run_random_walk(edge_index, rowptr, col, starts, walk_length)
    walks = walks.view(num_groups, -1, walk_length + 1)

    group_edges: List[List[Tensor]] = [[] for _ in range(num_groups)]
    seen = torch.zeros(num_nodes, device=device, dtype=torch.bool)
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
            buckets.append(edge_index.new_zeros((2, 0), dtype=edge_index.dtype))
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
) -> List[Tensor]:
    """
    使用随机游走生成 hop buckets，bucket i 对应 hop i+1，
    最后一个 bucket 合并 >= num_buckets 的 hop.
    """
    num_buckets = max(1, int(num_buckets))
    walk_length = max(1, int(walk_length))
    walks_per_node = max(1, int(walks_per_node))
    if edge_index.numel() == 0:
        return [edge_index.new_zeros((2, 0), dtype=edge_index.dtype) for _ in range(num_buckets)]

    inferred_nodes = int(edge_index.max().item()) + 1
    num_nodes = max(num_nodes, inferred_nodes)
    edge_index = edge_index.to(device)

    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes)).coalesce()
    rowptr, col, _ = adj.csr()

    starts = torch.arange(num_nodes, device=device, dtype=torch.long)
    starts = starts.repeat_interleave(walks_per_node)
    walks = _run_random_walk(edge_index, rowptr, col, starts, walk_length)

    buckets = [edge_index.new_zeros((2, 0), dtype=edge_index.dtype) for _ in range(num_buckets)]
    src = walks[:, 0]
    for d in range(1, walk_length + 1):
        dst = walks[:, d]
        valid = dst >= 0
        if d > 1:
            prev = walks[:, d - 1]
            valid = valid & (dst != prev)
        if not valid.any():
            continue
        src_valid = src[valid]
        dst_valid = dst[valid]
        edges = torch.stack([src_valid, dst_valid], dim=0)
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
):
    """
    构建 per-head edge_index 列表。
    当前实现不再按 head 分桶：所有 head 共享同一份随机游走子图。
    """
    _ = num_groups  # kept for backward-compatible call signature
    buckets = compute_hop_buckets_random_walk(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_buckets=1,
        device=device,
        walk_length=walk_length,
        walks_per_node=walks_per_node,
    )
    shared_edge_index = buckets[0] if buckets else edge_index.new_zeros((2, 0), dtype=edge_index.dtype)
    return [shared_edge_index for _ in range(max(1, int(num_heads)))]


def _run_random_walk(edge_index: Tensor, rowptr: Tensor, col: Tensor, starts: Tensor, walk_length: int) -> Tensor:
    try:
        from torch_cluster import random_walk
    except Exception as exc:
        raise RuntimeError("torch_cluster.random_walk not available") from exc
    try:
        return random_walk(rowptr, col, starts, walk_length)
    except RuntimeError as exc:
        msg = str(exc)
        if "must match" in msg or "size" in msg:
            return random_walk(edge_index[0], edge_index[1], starts, walk_length)
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
    edge_index = edge_index.to(device)
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce()
    rowptr, col, _ = adj.csr()

    group_nodes: List[Tensor] = []
    all_nodes = torch.arange(num_nodes, device=device, dtype=torch.long)
    for g in range(num_groups):
        seeds = all_nodes[g::num_groups]
        if seeds.numel() > max_nodes_per_group:
            seeds = seeds[:max_nodes_per_group]
        if seeds.numel() == 0:
            group_nodes.append(seeds)
            continue
        starts = seeds.repeat_interleave(walks_per_node)
        walks = _run_random_walk(edge_index, rowptr, col, starts, walk_length)
        seen = torch.zeros(num_nodes, device=device, dtype=torch.bool)
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


def get_batch_blockize(args, x, y, idx_batch, rest_split_sizes, edge_index, N, device=None):
    """
    Generate a local subsequence in sequence parallel and slice the corresponding edge_index.
    """
    seq_parallel_world_size = get_sequence_parallel_world_size() if sequence_parallel_is_initialized() else 1
    seq_parallel_world_rank = get_sequence_parallel_rank() if sequence_parallel_is_initialized() else 0
    seq_length = args.seq_len

    x_i = x[idx_batch]  # [s, x_d]
    y_i = y[idx_batch]  # [s]

    edge_index_i = gen_sub_edge_index(edge_index, idx_batch, N)

    if args.model == "graphormer":
        edge_index_i = fix_edge_index(edge_index_i, idx_batch.shape[0])

    dev = None
    if device is not None:
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cuda":
            edge_index_i = edge_index_i.to(dev)

    attn_bias = None
    try:
        edge_index_i_heads = build_head_hop_edges(
            edge_index=edge_index_i,
            num_nodes=x_i.size(0),
            num_heads=args.num_heads,
            num_groups=1,
            device=edge_index_i.device,
            walk_length=getattr(args, "head_hop_walk_length", 4),
            walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
        )
    except RuntimeError:
        edge_index_i = edge_index_i.to("cpu")
        edge_index_i_heads = build_head_hop_edges(
            edge_index=edge_index_i,
            num_nodes=x_i.size(0),
            num_heads=args.num_heads,
            num_groups=1,
            device=edge_index_i.device,
            walk_length=getattr(args, "head_hop_walk_length", 4),
            walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
        )
    if dev is not None and dev.type == "cuda":
        if isinstance(edge_index_i_heads, list):
            edge_index_i_heads = [e.to(dev) for e in edge_index_i_heads]
        else:
            edge_index_i_heads = edge_index_i_heads.to(dev)
    if seq_parallel_world_size > 1:
        src_rank = get_sequence_parallel_src_rank()
        group = get_sequence_parallel_group()
        if isinstance(edge_index_i_heads, list):
            device_ref = edge_index_i_heads[0].device if edge_index_i_heads else edge_index_i.device
            dtype_ref = edge_index_i_heads[0].dtype if edge_index_i_heads else edge_index_i.dtype
            if dist.get_rank() == src_rank:
                head_count = torch.tensor([len(edge_index_i_heads)], device=device_ref, dtype=torch.long)
            else:
                head_count = torch.empty(1, device=device_ref, dtype=torch.long)
            dist.broadcast(head_count, src_rank, group=group)
            head_count_val = int(head_count.item())
            if dist.get_rank() == src_rank:
                size_broad = torch.tensor(
                    [e.size(1) for e in edge_index_i_heads],
                    device=device_ref,
                    dtype=torch.long,
                )
            else:
                size_broad = torch.empty(head_count_val, device=device_ref, dtype=torch.long)
            dist.broadcast(size_broad, src_rank, group=group)
            if dist.get_rank() != src_rank:
                edge_index_i_heads = [
                    torch.empty((2, int(size_broad[i].item())), device=device_ref, dtype=dtype_ref)
                    for i in range(head_count_val)
                ]
            for i in range(head_count_val):
                dist.broadcast(edge_index_i_heads[i], src_rank, group=group)
        else:
            if dist.get_rank() == src_rank:
                size_broad = torch.tensor([edge_index_i_heads.size(1)], device=edge_index_i_heads.device, dtype=torch.long)
            else:
                size_broad = torch.empty(1, device=edge_index_i_heads.device, dtype=torch.long)
            dist.broadcast(size_broad, src_rank, group=group)
            if dist.get_rank() != src_rank:
                edge_index_i_heads = torch.empty((2, int(size_broad.item())), device=edge_index_i_heads.device, dtype=edge_index_i_heads.dtype)
            dist.broadcast(edge_index_i_heads, src_rank, group=group)

    if idx_batch.shape[0] < seq_length:
        assert rest_split_sizes is not None, 'Rest split_sizes should not be None'
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
        last_batch_flag(True)
    else:
        assert seq_length % seq_parallel_world_size == 0
        sub_seq_length = seq_length // seq_parallel_world_size
        sub_seq_start = seq_parallel_world_rank * sub_seq_length
        sub_seq_end = (seq_parallel_world_rank + 1) * sub_seq_length

        x_i = x_i[sub_seq_start:sub_seq_end, :]
        y_i = y_i[sub_seq_start:sub_seq_end]
        if args.model == "graphormer":
            edge_index_i_heads = _apply_to_edge_index(
                edge_index_i_heads,
                lambda e: adjust_edge_index_nomerge(e, sub_seq_length),
            )

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
