import contextlib
import gc
import os
import random
import resource
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import coalesce

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
from utils.split_utils import load_default_split


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
    feature = torch.load(data_path + "/x.pt")
    y = torch.load(data_path + "/y.pt")
    edge_index_global = torch.load(data_path + "/edge_index.pt")
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
        if getattr(args, "sparse_query_chunk_size", 0) > 0:
            model.set_sparse_attention_query_chunk_size(args.sparse_query_chunk_size)
        if getattr(args, "activation_checkpoint", False):
            model.set_activation_checkpoint(True)
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
    random_edge_blocks: bool
    probe_size: int
    block_size: int
    warmup_epochs: int
    patience: int
    gain_threshold: float
    max_real_edges_per_query: int
    max_rw_edges_per_query: int


def _pin_cpu_tensor(tensor):
    if tensor.device.type == "cpu" and torch.cuda.is_available():
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
    return tensor


def _random_block_sampling_enabled(args, adaptive_edge_budget_cfg=None) -> bool:
    if adaptive_edge_budget_cfg is not None:
        return bool(adaptive_edge_budget_cfg.random_edge_blocks)
    return bool(getattr(args, "random_edge_blocks", False))


def _adaptive_edge_budget_enabled(args) -> bool:
    return bool(getattr(args, "adaptive_edge_budget", False))


def _get_real_edge_budget(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> int:
    if edge_budget_state is not None and "real_edges_per_query" in edge_budget_state:
        return int(edge_budget_state["real_edges_per_query"])
    if adaptive_edge_budget_cfg is not None:
        return int(adaptive_edge_budget_cfg.max_real_edges_per_query)
    return int(getattr(args, "real_edges_per_query", 0))


def _get_rw_edge_budget(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> int:
    if edge_budget_state is not None and "rw_edges_per_query" in edge_budget_state:
        return int(edge_budget_state["rw_edges_per_query"])
    if adaptive_edge_budget_cfg is not None:
        return int(adaptive_edge_budget_cfg.max_rw_edges_per_query)
    return int(getattr(args, "rw_edges_per_query", 0))


def _use_real_edges(args, adaptive_edge_budget_cfg=None) -> bool:
    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return _get_real_edge_budget(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg) > 0
    return bool(getattr(args, "include_real_edges", 0))


def _use_real_edges_for_state(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> bool:
    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg) > 0
    return bool(getattr(args, "include_real_edges", 0))


def _use_rw_edges(args, edge_budget_state=None, adaptive_edge_budget_cfg=None) -> bool:
    if int(getattr(args, "head_hop_walks_per_node", 0)) <= 0:
        return False
    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        return _get_rw_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg) > 0
    return True


def _resolve_adaptive_edge_budget_config(args, split_idx) -> _AdaptiveEdgeBudgetConfig:
    enabled = _adaptive_edge_budget_enabled(args)
    random_edge_blocks = bool(getattr(args, "random_edge_blocks", False) or enabled)

    probe_size = int(getattr(args, "adaptive_edge_budget_probe_size", 0))
    if enabled and probe_size <= 0:
        valid_idx = split_idx.get("valid")
        valid_n = int(valid_idx.numel()) if valid_idx is not None else 0
        probe_size = min(512, valid_n) if valid_n > 0 else 0
    else:
        probe_size = max(0, probe_size)

    block_size = int(getattr(args, "adaptive_edge_budget_block_size", 0))
    if block_size <= 0:
        block_size = 1

    warmup_epochs = int(getattr(args, "adaptive_edge_budget_warmup_epochs", 0))
    if warmup_epochs <= 0:
        warmup_epochs = max(1_000_000, int(args.epochs))

    patience = int(getattr(args, "adaptive_edge_budget_patience", 0))
    if patience <= 0:
        patience = 3
    patience = max(3, patience)

    return _AdaptiveEdgeBudgetConfig(
        enabled=enabled,
        random_edge_blocks=random_edge_blocks,
        probe_size=probe_size,
        block_size=max(1, block_size),
        warmup_epochs=max(0, warmup_epochs),
        patience=patience,
        gain_threshold=float(getattr(args, "adaptive_edge_budget_gain_threshold", 0.0)),
        max_real_edges_per_query=max(0, int(getattr(args, "real_edges_per_query", 0))),
        max_rw_edges_per_query=max(0, int(getattr(args, "rw_edges_per_query", 0))),
    )


def _edge_block_seed(args, edge_seed, offset: int) -> int:
    base = getattr(args, "seed", 0) if edge_seed is None else int(edge_seed)
    return int(base) + int(offset)


def _sample_edges_per_query_random(edge_index, max_edges_per_query: int, seed: int):
    if edge_index is None or edge_index.numel() == 0 or int(max_edges_per_query) <= 0:
        return edge_index

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

    real_budget = _get_real_edge_budget(args, edge_budget_state, adaptive_edge_budget_cfg)
    if real_budget <= 0:
        return None

    real_edges = edge_index_global.to(device)
    if _random_block_sampling_enabled(args, adaptive_edge_budget_cfg):
        real_edges = _sample_edges_per_query_random(
            real_edges,
            real_budget,
            _edge_block_seed(args, edge_seed, 17),
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
    return _sample_edges_per_query_random(
        rw_edges,
        rw_budget,
        _edge_block_seed(args, edge_seed, 37),
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
):
    probe_idx_global = probe_idx_global.to(dtype=torch.long, device="cpu").view(-1)
    seed_ctx = (
        _fixed_torch_cpu_seed(edge_seed)
        if edge_seed is not None and torch.device(rw_device).type == "cpu"
        else contextlib.nullcontext()
    )
    with seed_ctx:
        self_loop_pool = None
        if getattr(args, "include_self_loops", 0):
            self_loop = torch.stack([probe_idx_global, probe_idx_global], dim=0)
            self_loop_pool = self_loop.to(dtype=torch.long, device="cpu")

        real_pool = None
        if _use_real_edges(args, adaptive_edge_budget_cfg):
            real_pool = _filter_edge_index_by_dst(edge_index_global.cpu(), probe_idx_global)
            if real_pool is not None:
                real_pool = real_pool.to(dtype=torch.long, device="cpu")

        rw_pool = None
        if _use_rw_edges(args, adaptive_edge_budget_cfg=adaptive_edge_budget_cfg):
            rw_pool = build_head_hop_edges(
                edge_index=edge_index_global,
                num_nodes=num_nodes,
                num_heads=args.num_heads,
                num_groups=1,
                device=rw_device,
                walk_length=getattr(args, "head_hop_walk_length", 4),
                walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
            )
            if isinstance(rw_pool, list):
                rw_pool = _merge_edge_index_list(rw_pool)
            if rw_pool is not None:
                rw_pool = _filter_edge_index_by_dst(rw_pool, probe_idx_global)
                rw_pool = rw_pool.to(dtype=torch.long, device="cpu")

    return {
        "self_loop": self_loop_pool,
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
    self_loop = edge_pools.get("self_loop")
    if self_loop is not None and self_loop.numel() > 0:
        parts.append(self_loop)

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
            template = edge_pools.get("self_loop")
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
):
    if sp_rank == sp_src_rank:
        pools = edge_pools
        if pools is None:
            pools = _build_probe_edge_pools(
                args,
                edge_index_global,
                num_nodes,
                rw_device,
                probe_idx_global,
                edge_seed=edge_seed,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
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

    merged = _broadcast_prefetched_edges(merged_cpu, device, sp_group, sp_src_rank, sp_rank)
    return _finalize_attention_edges(args, merged)


class _AdaptiveEdgeBudgetController:
    def __init__(self, config: _AdaptiveEdgeBudgetConfig) -> None:
        self.enabled = config.enabled
        self.block_size = int(config.block_size)
        self.max_real = int(config.max_real_edges_per_query)
        self.max_rw = int(config.max_rw_edges_per_query)
        self.warmup_epochs = int(config.warmup_epochs)
        self.gain_threshold = float(config.gain_threshold)
        self.patience = int(config.patience)
        self.bad_rounds = 0
        self.seen_positive_gain = False
        self.frozen = not self.enabled
        if not self.enabled:
            self.real_budget = self.max_real
            self.rw_budget = self.max_rw
            return

        self.real_budget = 2 if self.max_real > 0 else 0
        self.rw_budget = 2 if self.max_rw > 0 else 0
        if self.warmup_epochs <= 0:
            self.frozen = True

    def current_state(self):
        return {
            "real_edges_per_query": int(self.real_budget),
            "rw_edges_per_query": int(self.rw_budget),
        }

    def candidate_states(self):
        out = {}
        if self.real_budget < self.max_real:
            out["real"] = {
                "real_edges_per_query": min(self.max_real, self.real_budget + self.block_size),
                "rw_edges_per_query": self.rw_budget,
            }
        if self.rw_budget < self.max_rw:
            out["rw"] = {
                "real_edges_per_query": self.real_budget,
                "rw_edges_per_query": min(self.max_rw, self.rw_budget + self.block_size),
            }
        return out

    def update(self, choice, best_gain, next_state=None):
        if best_gain > 0.0:
            self.seen_positive_gain = True
        if next_state is not None and choice is not None:
            self.real_budget = int(next_state["real_edges_per_query"])
            self.rw_budget = int(next_state["rw_edges_per_query"])
            self.bad_rounds = 0
            return
        if not self.seen_positive_gain:
            return
        if best_gain <= self.gain_threshold:
            self.bad_rounds += 1
            if self.bad_rounds >= self.patience:
                self.frozen = True


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
                loss_sum = F.nll_loss(
                    out_local.index_select(0, probe_idx_eff),
                    local_y_eff.index_select(0, probe_idx_eff).long(),
                    reduction="sum",
                ).to(torch.float32).view(1)
                count = torch.tensor([probe_idx_eff.numel()], device=device, dtype=torch.long)

    if sp_world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=sp_group)

    _restore_dropout(drop_states)
    if not was_training:
        model.eval()

    mean_loss = float(loss_sum.item() / max(int(count.item()), 1))
    del out_local
    _cuda_empty_cache(args)
    return mean_loss


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
    if (not controller.enabled) or controller.frozen or epoch > controller.warmup_epochs:
        return
    if local_probe_idx is None or probe_idx_global is None or probe_idx_global.numel() == 0:
        controller.frozen = True
        return

    probe_seed = int(getattr(args, "seed", 0)) + 100000 + int(epoch)
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
    base_state = controller.current_state()
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
    base_loss = _probe_loss_sp(
        args, model, x_local, local_y, local_probe_idx, base_edges,
        device, amp_dtype, sp_group, sp_world_size,
    )
    base_count = _edge_count(base_edges)
    del base_edges

    best_kind = None
    best_gain = float("-inf")
    best_state = None
    best_loss = None
    best_count = None
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
        cand_loss = _probe_loss_sp(
            args, model, x_local, local_y, local_probe_idx, cand_edges,
            device, amp_dtype, sp_group, sp_world_size,
        )
        cand_count = _edge_count(cand_edges)
        delta_edges = max(cand_count - base_count, 1)
        gain = (base_loss - cand_loss) / float(delta_edges)
        if gain > best_gain:
            best_kind = kind
            best_gain = gain
            best_state = cand_state
            best_loss = cand_loss
            best_count = cand_count
        del cand_edges

    controller.update(best_kind if best_gain > controller.gain_threshold else None,
                      best_gain, best_state if best_gain > controller.gain_threshold else None)

    if args.rank == 0:
        print(
            f"  ↳ BudgetCtrl epoch={epoch} probe_loss={base_loss:.4f} "
            f"edges={base_count} choice={best_kind} gain={best_gain:.6e} "
            f"next_real={controller.real_budget} next_rw={controller.rw_budget}"
            + (
                f" cand_loss={best_loss:.4f} cand_edges={best_count}"
                if best_loss is not None and best_count is not None
                else ""
            )
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
                        edge_seed=None, edge_budget_state=None, adaptive_edge_budget_cfg=None):
    parts = []
    final_torch_device = torch.device(final_device)
    if getattr(args, "include_self_loops", 0):
        self_loop = torch.arange(num_nodes, device=final_torch_device, dtype=torch.long)
        self_loop = torch.stack([self_loop, self_loop], dim=0)
        parts.append(self_loop)
    real_edges = _resolve_real_edges_for_state(
        args,
        edge_index_global,
        final_torch_device,
        edge_seed=edge_seed,
        edge_budget_state=edge_budget_state,
        adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
    )
    rw_heads = None
    if _use_rw_edges(args, edge_budget_state, adaptive_edge_budget_cfg):
        rw_heads = build_head_hop_edges(
            edge_index=edge_index_global,
            num_nodes=num_nodes,
            num_heads=args.num_heads,
            num_groups=1,
            device=rw_device,
            walk_length=getattr(args, "head_hop_walk_length", 4),
            walks_per_node=getattr(args, "head_hop_walks_per_node", 2),
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
    if real_edges is not None:
        parts.append(real_edges)
    if rw_heads is not None:
        parts.append(rw_heads)
    merged = _merge_edge_index_list(parts)
    if merged is None:
        merged = edge_index_global.new_zeros((2, 0), dtype=torch.long).to(final_torch_device)
    if args.model == "graphormer" and args.num_global_node > 0:
        merged = fix_edge_index(merged, num_nodes)
        merged = adjust_edge_index_nomerge(merged, nodes_per_rank)
    return merged


def _finalize_attention_edges(args, merged):
    if args.attn_type == "sparse" and getattr(args, "sparse_query_chunk_size", 0) > 0:
        return _pack_chunked_edges(merged, args.sparse_query_chunk_size, mode="gpu_chunk")
    return merged


def _build_prefetched_cpu_edges(args, edge_index_global, num_nodes, rw_device, nodes_per_rank, edge_seed,
                                edge_budget_state=None, adaptive_edge_budget_cfg=None):
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
        dist.broadcast(size_t, sp_src_rank, group=sp_group)
        if sp_rank != sp_src_rank:
            merged = torch.empty((2, int(size_t.item())), device=device, dtype=torch.long)
        dist.broadcast(merged, sp_src_rank, group=sp_group)
    return merged


def _pack_chunked_edges(merged, chunk_size, mode):
    chunk = max(int(chunk_size), 1)
    edge_store = merged.detach()

    if mode == "cpu_stream":
        edge_store = edge_store.cpu()
    elif mode != "gpu_chunk":
        raise ValueError(f"Unsupported chunked edge payload mode: {mode}")

    if edge_store.numel() == 0:
        empty = torch.empty((0,), dtype=torch.long, device=edge_store.device)
        offsets = torch.zeros(1, dtype=torch.long, device=edge_store.device)
        if mode == "cpu_stream":
            empty = _pin_cpu_tensor(empty)
            offsets = _pin_cpu_tensor(offsets)
        return {
            "mode": mode,
            "src": empty,
            "dst": empty.clone(),
            "offsets": offsets,
            "chunk_size": chunk,
        }

    edge_index = edge_store[:2].to(torch.long).contiguous()
    edge_hops = edge_store[2].to(torch.long).contiguous() if edge_store.size(0) == 3 else None

    order = torch.argsort(edge_index[1])
    src_sorted = edge_index[0].index_select(0, order).contiguous()
    dst_sorted = edge_index[1].index_select(0, order).contiguous()
    hop_sorted = edge_hops.index_select(0, order).contiguous() if edge_hops is not None else None

    max_dst = int(dst_sorted.max().item()) if dst_sorted.numel() > 0 else -1
    num_chunks = max((max_dst + chunk) // chunk, 0)
    offsets = torch.zeros(num_chunks + 1, dtype=torch.long, device=dst_sorted.device)
    if dst_sorted.numel() > 0 and num_chunks > 0:
        dst_bins = torch.div(dst_sorted, chunk, rounding_mode="floor")
        counts = torch.bincount(dst_bins, minlength=num_chunks)
        offsets[1:] = counts.cumsum(dim=0)

    if mode == "cpu_stream":
        src_sorted = _pin_cpu_tensor(src_sorted)
        dst_sorted = _pin_cpu_tensor(dst_sorted)
        offsets = _pin_cpu_tensor(offsets)
        if hop_sorted is not None:
            hop_sorted = _pin_cpu_tensor(hop_sorted)

    payload = {
        "mode": mode,
        "src": src_sorted,
        "dst": dst_sorted,
        "offsets": offsets,
        "chunk_size": chunk,
    }
    if hop_sorted is not None:
        payload["hops"] = hop_sorted
    return payload


def _build_streaming_edges(args, edge_index_global, num_nodes, rw_device, nodes_per_rank, edge_seed=None,
                           edge_budget_state=None, adaptive_edge_budget_cfg=None):
    seed_ctx = fixed_random_seed(edge_seed) if edge_seed is not None else contextlib.nullcontext()
    with seed_ctx:
        merged = _build_merged_edges(
            args, edge_index_global, num_nodes, "cpu", rw_device, nodes_per_rank,
            edge_seed=edge_seed,
            edge_budget_state=edge_budget_state,
            adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
        )
    return _pack_chunked_edges(merged, args.sparse_query_chunk_size, mode="cpu_stream")


def _build_and_broadcast_edges(args, edge_index_global, num_nodes, device, rw_device, sp_group,
                               sp_src_rank, sp_rank, nodes_per_rank, edge_seed=None,
                               edge_budget_state=None, adaptive_edge_budget_cfg=None):
    if sp_rank == sp_src_rank:
        seed_ctx = (
            _fixed_torch_cpu_seed(edge_seed)
            if edge_seed is not None and torch.device(rw_device).type == "cpu"
            else contextlib.nullcontext()
        )
        with seed_ctx:
            merged = _build_merged_edges(
                args,
                edge_index_global,
                num_nodes,
                device,
                rw_device,
                nodes_per_rank,
                edge_seed=edge_seed,
                edge_budget_state=edge_budget_state,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            )
        size_t = torch.tensor([merged.shape[1]], device=device, dtype=torch.long)
    else:
        size_t = torch.empty(1, device=device, dtype=torch.long)
        merged = None

    if dist.is_initialized():
        dist.broadcast(size_t, sp_src_rank, group=sp_group)
        if sp_rank != sp_src_rank:
            merged = torch.empty((2, int(size_t.item())), device=device, dtype=torch.long)
        dist.broadcast(merged, sp_src_rank, group=sp_group)

    return merged


def _build_attention_edges(args, edge_index_global, num_nodes, device, rw_device, sp_group,
                           sp_src_rank, sp_rank, nodes_per_rank, edge_seed=None,
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
    merged = _build_and_broadcast_edges(
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
    )
    return _finalize_attention_edges(args, merged)


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


def _eval_sp(args, model, feature, y, split_idx, edge_index_global, num_nodes,
             device, rw_device, sp_group, sp_src_rank, sp_rank, sp_world_size,
             rank_start, rank_end, local_nodes, amp_dtype=None, cached_edge_index=None,
             edge_budget_state=None, adaptive_edge_budget_cfg=None):
    if cached_edge_index is None:
        with fixed_random_seed(args.seed):
            edge_index_eval = _build_attention_edges(
                args, edge_index_global, num_nodes, device, rw_device,
                sp_group, sp_src_rank, sp_rank, local_nodes,
                edge_seed=args.seed if getattr(args, "stream_edges_from_cpu", False) else None,
                edge_budget_state=edge_budget_state,
                adaptive_edge_budget_cfg=adaptive_edge_budget_cfg,
            )
    else:
        edge_index_eval = cached_edge_index

    was_training = model.training
    model.train()
    drop_states = _set_dropout_eval(model)
    with torch.no_grad():
        x_local = feature[rank_start:rank_end].float().to(device)
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
