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
        "ready_count",
        "launched",
        "work",
        "ready_event",
    )

    def __init__(self, params, ranges, flat):
        self.params = params
        self.ranges = ranges
        self.flat = flat
        self.ready_count = 0
        self.launched = False
        self.work = None
        self.ready_event = None


class GradientReducer:
    """Flattened gradient all-reduce with optional backward overlap."""

    def __init__(
        self,
        model,
        process_group=None,
        bucket_cap_mb: float = 25.0,
        overlap: bool = False,
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
        self._overlap_requested = bool(overlap)
        self._hooks = []
        self._param_to_bucket = {}
        self._buckets = []
        self._prepared = False
        self._comm_stream = None

        if not self.enabled:
            self.overlap_enabled = False
            return

        self._build_buckets(model)
        self.overlap_enabled = (
            self._overlap_requested
            and torch.cuda.is_available()
            and any(bucket.flat.is_cuda for bucket in self._buckets)
        )
        if self.overlap_enabled:
            self._comm_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            self._register_hooks()

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
            bucket_idx = len(self._buckets)
            bucket = _GradBucket(list(cur_params), list(cur_ranges), flat)
            self._buckets.append(bucket)
            for param, (start, end) in zip(bucket.params, bucket.ranges):
                self._param_to_bucket[id(param)] = (bucket_idx, start, end)
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

    def _register_hooks(self) -> None:
        for bucket_idx, bucket in enumerate(self._buckets):
            for _, param in enumerate(bucket.params):
                hook = param.register_hook(self._make_hook(bucket_idx, param))
                self._hooks.append(hook)

    def _make_hook(self, bucket_idx, param):
        def _hook(grad):
            if not self._prepared:
                return grad
            bucket = self._buckets[bucket_idx]
            _, start, end = self._param_to_bucket[id(param)]
            bucket.flat[start:end].copy_(grad.reshape(-1))
            bucket.ready_count += 1
            if bucket.ready_count == len(bucket.params):
                self._launch_bucket_async(bucket)
            return grad

        return _hook

    def _launch_bucket_async(self, bucket: _GradBucket) -> None:
        if bucket.launched:
            return
        bucket.launched = True
        bucket.ready_event = torch.cuda.Event()
        bucket.ready_event.record(torch.cuda.current_stream())
        with torch.cuda.stream(self._comm_stream):
            self._comm_stream.wait_event(bucket.ready_event)
            bucket.flat.div_(self.world_size)
            bucket.work = dist.all_reduce(
                bucket.flat,
                op=dist.ReduceOp.SUM,
                group=self.group,
                async_op=True,
            )

    def prepare_backward(self) -> None:
        if not self.enabled:
            return
        self._prepared = True
        for bucket in self._buckets:
            bucket.flat.zero_()
            bucket.ready_count = 0
            bucket.launched = False
            bucket.work = None
            bucket.ready_event = None

    def finalize_backward(self) -> None:
        if not self.enabled:
            return

        if not self.overlap_enabled:
            for bucket in self._buckets:
                self._reduce_bucket_sync(bucket)
            for bucket in self._buckets:
                self._scatter_bucket(bucket)
            self._prepared = False
            return

        for bucket in self._buckets:
            if not bucket.launched:
                self._reduce_bucket_sync(bucket)

        for bucket in self._buckets:
            if bucket.work is not None:
                bucket.work.wait()

        torch.cuda.current_stream().wait_stream(self._comm_stream)
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
            f"buckets={total_buckets}, overlap={int(self.overlap_enabled)}"
        )


def build_gradient_reducer(
    model,
    process_group=None,
    bucket_cap_mb: float = 25.0,
    overlap: bool = False,
    world_size: int | None = None,
):
    return GradientReducer(
        model=model,
        process_group=process_group,
        bucket_cap_mb=bucket_cap_mb,
        overlap=overlap,
        world_size=world_size,
    )


Reducer = GradientReducer
