import torch
import torch.distributed as dist

from gt_sp.initialize import (
    get_sequence_parallel_group,
    get_sequence_parallel_src_rank,
)


def sync_params_and_buffers(model):
    if not dist.is_initialized():
        return
    for _, param in model.state_dict().items():
        torch.distributed.broadcast(
            param.data,
            src=get_sequence_parallel_src_rank(),
            group=get_sequence_parallel_group(),
        )


class _GradBucket:
    __slots__ = (
        "params",
        "ranges",
        "flat",
    )

    def __init__(self, params, ranges, flat):
        self.params = params
        self.ranges = ranges
        self.flat = flat


class GradientReducer:
    """Flattened gradient all-reduce (synchronous).

    NOTE: Async/overlap mode was removed because in this codebase all ranks
    share a single SP process group.  Launching gradient all-reduce
    asynchronously on a dedicated CUDA stream caused NCCL communicator
    contention with SeqAllToAll, making bwd_time ~29% slower.  The simpler
    synchronous path avoids that contention entirely.
    """

    def __init__(
        self,
        model,
        process_group=None,
        bucket_cap_mb: float = 25.0,
        world_size: int | None = None,
    ) -> None:
        self.group = process_group
        self.world_size = int(world_size) if world_size is not None else 1
        self.enabled = (
            self.group is not None
            and dist.is_initialized()
            and self.world_size > 1
        )
        self.bucket_cap_bytes = max(1, int(float(bucket_cap_mb) * 1024 * 1024))
        self._buckets = []
        self._prepared = False

        if not self.enabled:
            return

        self._build_buckets(model)

    def _build_buckets(self, model) -> None:
        params = [param for param in model.parameters() if param.requires_grad]
        params.reverse()

        cur_params = []
        cur_ranges = []
        cur_numel = 0
        cur_bytes = 0
        cur_key = None

        def flush_bucket():
            nonlocal cur_params, cur_ranges, cur_numel, cur_bytes, cur_key
            if not cur_params:
                return
            flat = torch.zeros(cur_numel, device=cur_key[0], dtype=cur_key[1])
            bucket = _GradBucket(list(cur_params), list(cur_ranges), flat)
            self._buckets.append(bucket)
            cur_params = []
            cur_ranges = []
            cur_numel = 0
            cur_bytes = 0
            cur_key = None

        for param in params:
            key = (param.device, param.dtype)
            param_bytes = param.numel() * param.element_size()
            if cur_params and (key != cur_key or (cur_bytes + param_bytes) > self.bucket_cap_bytes):
                flush_bucket()
            start = cur_numel
            end = start + param.numel()
            cur_params.append(param)
            cur_ranges.append((start, end))
            cur_numel = end
            cur_bytes += param_bytes
            cur_key = key

        flush_bucket()

    def prepare_backward(self) -> None:
        if not self.enabled:
            return
        self._prepared = True

    def finalize_backward(self) -> None:
        if not self.enabled:
            return
        for bucket in self._buckets:
            self._reduce_bucket_sync(bucket)
        for bucket in self._buckets:
            self._scatter_bucket(bucket)
        self._prepared = False

    def _reduce_bucket_sync(self, bucket: _GradBucket) -> None:
        bucket.flat.zero_()
        for param, (start, end) in zip(bucket.params, bucket.ranges):
            grad = param.grad
            if grad is None:
                continue
            bucket.flat[start:end].copy_(grad.reshape(-1))
        bucket.flat.div_(self.world_size)
        dist.all_reduce(bucket.flat, op=dist.ReduceOp.SUM, group=self.group)

    def _scatter_bucket(self, bucket: _GradBucket) -> None:
        for param, (start, end) in zip(bucket.params, bucket.ranges):
            grad_view = bucket.flat[start:end].view_as(param)
            if param.grad is None:
                param.grad = grad_view.clone()
            else:
                param.grad.copy_(grad_view)

    def extra_repr(self) -> str:
        if not self.enabled:
            return "disabled"
        total_buckets = len(self._buckets)
        return (
            f"bucket_mb={self.bucket_cap_bytes / (1024 ** 2):.1f}, "
            f"buckets={total_buckets}"
        )


def build_gradient_reducer(
    model,
    process_group=None,
    bucket_cap_mb: float = 25.0,
    world_size: int | None = None,
):
    return GradientReducer(
        model=model,
        process_group=process_group,
        bucket_cap_mb=bucket_cap_mb,
        world_size=world_size,
    )


Reducer = GradientReducer
