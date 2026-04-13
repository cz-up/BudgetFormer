import time
import threading
from collections import defaultdict

import torch


_PROFILE_ENABLED = False
_PROFILE_STATS = defaultdict(lambda: {"count": 0, "cpu_ms": 0.0, "cuda_pairs": [], "bytes": 0})
_PROFILE_SCALARS = {}
_PROFILE_LOCK = threading.Lock()


def enable_comm_profiler(enabled: bool = True) -> None:
    global _PROFILE_ENABLED
    _PROFILE_ENABLED = bool(enabled)


def comm_profiler_enabled() -> bool:
    return _PROFILE_ENABLED


def reset_comm_profiler() -> None:
    with _PROFILE_LOCK:
        _PROFILE_STATS.clear()
        _PROFILE_SCALARS.clear()


def profile_call(name: str, fn, device=None, payload_bytes: int = 0):
    if not _PROFILE_ENABLED:
        return fn()

    with _PROFILE_LOCK:
        stat = _PROFILE_STATS[name]
        stat["count"] += 1
        stat["bytes"] += max(0, int(payload_bytes))

    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda" and torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device=dev))
        result = fn()
        end.record(torch.cuda.current_stream(device=dev))
        with _PROFILE_LOCK:
            _PROFILE_STATS[name]["cuda_pairs"].append((start, end))
        return result

    t0 = time.perf_counter()
    result = fn()
    with _PROFILE_LOCK:
        _PROFILE_STATS[name]["cpu_ms"] += (time.perf_counter() - t0) * 1000.0
    return result


def record_scalar(name: str, value, reduce: str = "last") -> None:
    if not _PROFILE_ENABLED:
        return

    reduce = str(reduce)
    with _PROFILE_LOCK:
        cur = _PROFILE_SCALARS.get(name)
        if cur is None:
            _PROFILE_SCALARS[name] = {"reduce": reduce, "value": value}
            return
        if cur["reduce"] != reduce:
            raise ValueError(f"Profiler scalar {name!r} reduce mismatch: {cur['reduce']} vs {reduce}")
        if reduce == "last":
            cur["value"] = value
        elif reduce == "sum":
            cur["value"] += value
        elif reduce == "max":
            cur["value"] = value if value > cur["value"] else cur["value"]
        else:
            raise ValueError(f"Unsupported profiler scalar reduction: {reduce}")


def get_comm_profile_summary(reset: bool = False):
    if not _PROFILE_ENABLED:
        return {}

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    summary = {}
    with _PROFILE_LOCK:
        stats_items = list(_PROFILE_STATS.items())
        scalar_items = list(_PROFILE_SCALARS.items())
    for name, stat in stats_items:
        total_ms = float(stat["cpu_ms"])
        for start, end in stat["cuda_pairs"]:
            total_ms += float(start.elapsed_time(end))
        count = int(stat["count"])
        summary[name] = {
            "kind": "timing",
            "count": count,
            "total_ms": total_ms,
            "avg_ms": total_ms / max(count, 1),
            "total_bytes": int(stat["bytes"]),
        }
    for name, stat in scalar_items:
        summary[name] = {
            "kind": "scalar",
            "reduce": stat["reduce"],
            "value": stat["value"],
        }

    if reset:
        reset_comm_profiler()
    return summary
