"""Model-agnostic multi-tier activation checkpointing for SP graph transformers.

This module owns two public objects:

  _MultiTierResourceManager
      Profile-driven planner that jointly optimises topology cache placement and
      per-layer backward-state tier (recompute / keep_mha / retain).  Fully
      decoupled from any specific model architecture.

  apply_tier(layer, mode, x, **kwargs)
      Generic per-layer dispatch that invokes the correct execution path for the
      tier assigned by _MultiTierResourceManager.mode(i).  Requires each model's
      EncoderLayer to implement forward_attn_only() and forward_ffn_checkpointed().
"""

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from gt_sp.initialize import (
    get_sequence_parallel_group,
    sequence_parallel_is_initialized,
)


# ---------------------------------------------------------------------------
# Generic tier dispatch
# ---------------------------------------------------------------------------

def apply_tier(layer, mode: str, x, **kwargs):
    """Execute *layer* on *x* using the activation-checkpointing tier *mode*.

    Tier semantics
    --------------
    retain    : no checkpointing — full GPU activation retention.
    keep_mha  : MHA runs without checkpointing; FFN block is checkpointed.
    ffn_only  : alias for keep_mha (same execution path).
    recompute : whole-layer gradient checkpoint (default / fallback).

    Each EncoderLayer must implement:
      forward_attn_only(x, **kwargs) -> x_after_attn
      forward_ffn_checkpointed(x)    -> x_after_ffn
    """
    if mode == "retain":
        return layer(x, **kwargs)
    if mode in ("keep_mha", "ffn_only"):
        x = layer.forward_attn_only(x, **kwargs)
        return layer.forward_ffn_checkpointed(x)
    # "recompute" / "layer" / any unrecognised string → full-layer recompute.
    def _full(h):
        return layer(h, **kwargs)
    return checkpoint(_full, x, use_reentrant=False)


# ---------------------------------------------------------------------------
# _MultiTierResourceManager
# ---------------------------------------------------------------------------

class _MultiTierResourceManager:
    """Profile-driven multi-tier activation scheduler for SP graph transformers.

    Implements a Profile → Plan → Execute framework that jointly optimises:
      - topology cache placement (edge_index on GPU vs CPU)
      - per-layer backward-state tier (recompute / keep_mha / retain)

    State machine
    -------------
    DEFERRED              : waiting for edge budget to stabilise.
    WARMUP_RECOMPUTE      : all layers RECOMPUTE. Measures peak_HBM^R, T_fwd^R,
                            T_bwd^R, fwd_live_warmup.
    CALIBRATE_MHA         : last layer keep_mha, rest recompute. Measures
                            peak_mha_last, G_layer^mha_last, T_fwd^mha, T_bwd^mha.
    CALIBRATE_MHA_FRONT   : first layer keep_mha, rest recompute. Measures
                            peak_mha_first, G_layer^mha_first.  The planner uses
                            max(last, first) to guard against heterogeneous layers
                            (e.g. denser attention in early layers).
    CALIBRATE_RETAIN      : last layer retain, rest recompute.
    CALIBRATE_RETAIN_FRONT: first layer retain, rest recompute.
    ACTIVE                : structured planner has run; _modes + _cache_edge set.

    Note: T2_OFFLOAD (CPU offload) is excluded from the planner search space.
    save_on_cpu does not reduce forward-pass peak GPU memory vs T0_RECOMPUTE
    (activations are computed on GPU before the CPU transfer), and the
    synchronous non-pinned PCIe transfer adds pure overhead in the backward
    pass.  The three remaining tiers (T0/T1/T4) cover the memory-speed
    tradeoff space without the CALIBRATE_OFFLOAD step.

    Training loop responsibilities
    ------------------------------
    - Call plan(device) at the start of each forward pass.
    - Call record_post_forward_memory(device) after the encoder loop.
    - Call notify_step_end(device, t_bwd, t_fwd) after backward + optimizer.step().
    - Once is_active(): read cache_edge and call override_cache_decision()
      after cross-rank sync (same pattern as _CommAwareCheckpointer).
    """

    # Tier identifiers — must match apply_tier() branch labels.
    T0_RECOMPUTE = "recompute"
    T1_KEEP_MHA  = "keep_mha"
    T2_OFFLOAD   = "offload"
    T4_RETAIN    = "retain"

    # Edge-placement / build policies.
    EDGE_GPU_PERSIST   = "gpu_persist"
    EDGE_GPU_EPHEMERAL = "gpu_ephemeral"
    EDGE_CPU_BROADCAST = "cpu_broadcast"
    EDGE_CPU_RANK_LOCAL_PREFETCH = "cpu_rank_local_prefetch"
    EDGE_CPU_BROADCAST_PREFETCH = "cpu_broadcast_prefetch"

    # State machine values.
    _DEFERRED               = -1
    _WARMUP_RECOMPUTE       =  0
    _CALIBRATE_MHA          =  1   # last-layer T1 probe
    _CALIBRATE_MHA_FRONT    =  2   # first-layer T1 probe
    _CALIBRATE_RETAIN       =  3   # last-layer T4 probe
    _CALIBRATE_RETAIN_FRONT =  4   # first-layer T4 probe
    _ACTIVE                 =  5

    def __init__(self, n_layers: int, safety_margin: float = 0.15, deferred: bool = False):
        self.n_layers      = int(n_layers)
        self.safety_margin = float(safety_margin)
        self._state        = self._DEFERRED if deferred else self._WARMUP_RECOMPUTE
        self._modes: list  = [self.T0_RECOMPUTE] * self.n_layers
        self._warmup_target: int = 1 if deferred else 3
        self._warmup_peak_samples: list[int] = []
        self._warmup_fwd_live_samples: list[int] = []
        self._warmup_cpu_live_samples: list[int] = []
        self._warmup_t_fwd_samples: list[float] = []
        self._warmup_t_bwd_samples: list[float] = []
        self._deferred_baseline_ready: bool = False
        self._deferred_peak: int = 0
        self._deferred_fwd_live: int = 0
        self._deferred_cpu_live: int = 0
        self._deferred_t_fwd: float = 0.0
        self._deferred_t_bwd: float = 0.0

        # --- Profile measurements ---
        self._fwd_live_warmup:       int   = 0
        self._fwd_live_offload:      int   = 0
        self._fwd_live_mha:          int   = 0
        self._fwd_live_mha_front:    int   = 0
        self._fwd_live_retain:       int   = 0
        self._fwd_live_retain_front: int   = 0
        self._cpu_live_warmup:       int   = 0
        self._cpu_live_offload:      int   = 0
        self._peak_warmup:           int   = 0
        self._peak_mha:              int   = 0   # consolidated max(last, front)
        self._peak_mha_front:        int   = 0
        self._peak_retain:           int   = 0   # consolidated max(last, front)
        self._peak_retain_front:     int   = 0
        self._t_fwd_recompute:    float = 0.0
        self._t_fwd_offload:      float = 0.0
        self._t_fwd_mha:          float = 0.0
        self._t_fwd_retain:       float = 0.0
        self._t_bwd_recompute:    float = 0.0
        self._t_bwd_offload:      float = 0.0
        self._t_bwd_mha:          float = 0.0
        self._t_bwd_retain:       float = 0.0
        self._m_layer_offload:    int   = 0   # GPU bytes delta per offload layer
        self._m_layer_mha:        int   = 0   # GPU bytes per keep_mha layer
        self._m_layer_full:       int   = 0   # GPU bytes per retain layer
        self._cpu_bytes_offload:  int   = 0   # CPU bytes delta per offload layer

        # --- Edge/topology info ---
        self._edge_bytes:  int   = 0
        self._t_h2d_edge:  float = 0.0
        self._edge_profiles: dict = {}
        self._gpu_memory_limit_bytes: int = 0

        # --- Calibration OOM flags ---
        # Set by mark_current_calibration_infeasible() when a CALIBRATE_* step
        # OOMs. The planner skips candidate plans that use the disabled tier.
        self._t1_infeasible: bool = False
        self._t4_infeasible: bool = False

        # --- Final plan ---
        self._cache_edge:          bool = False
        self._edge_policy:         str  = self.EDGE_GPU_EPHEMERAL
        self._best_nonpersistent_edge_policy: str = self.EDGE_GPU_EPHEMERAL
        self._best_nonpersistent_tiers: list = [self.T0_RECOMPUTE] * self.n_layers

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rank0() -> bool:
        try:
            import torch.distributed as _dist
            return (not _dist.is_initialized()) or _dist.get_rank() == 0
        except Exception:
            return True

    @staticmethod
    def _available_cpu_bytes() -> int:
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            return 16 * (1024 ** 3)  # 16 GiB conservative default

    @staticmethod
    def _process_rss_bytes() -> int:
        try:
            import psutil
            return int(psutil.Process().memory_info().rss)
        except Exception:
            return 0

    def set_gpu_memory_limit_bytes(self, limit_bytes: int) -> None:
        """Override the per-rank GPU total used by the multi_tier planner."""
        self._gpu_memory_limit_bytes = max(0, int(limit_bytes))

    def _effective_total_gpu_bytes(self, real_total_gpu: int) -> int:
        if self._gpu_memory_limit_bytes <= 0:
            return int(real_total_gpu)
        return max(1, min(int(real_total_gpu), int(self._gpu_memory_limit_bytes)))

    def _gpu_total_msg(self, real_total_gpu: int, effective_total_gpu: int) -> str:
        real_mib = real_total_gpu / (1024 ** 2)
        effective_mib = effective_total_gpu / (1024 ** 2)
        if int(real_total_gpu) == int(effective_total_gpu):
            return f"GPU total={real_mib:.0f} MiB"
        return f"GPU total={real_mib:.0f} MiB, planner_limit={effective_mib:.0f} MiB"

    @staticmethod
    def _sync_max_scalar(value, device, dtype) -> int | float:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or not sequence_parallel_is_initialized()
        ):
            return value
        group = get_sequence_parallel_group()
        tensor = torch.tensor([value], device=device, dtype=dtype)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, group=group)
        return tensor.item()

    @classmethod
    def _edge_policy_order(cls):
        return (
            cls.EDGE_GPU_PERSIST,
            cls.EDGE_GPU_EPHEMERAL,
            cls.EDGE_CPU_RANK_LOCAL_PREFETCH,
            cls.EDGE_CPU_BROADCAST_PREFETCH,
            cls.EDGE_CPU_BROADCAST,
        )

    def set_edge_policy_profile(
        self,
        policy: str,
        prep_time_s: float,
        gpu_peak_bytes: int,
        cpu_delta_bytes: int,
        live_edge_bytes: int = 0,
        serial_time_s = None,
        overlap_time_s: float = 0.0,
        enabled: bool = True,
    ) -> None:
        policy = str(policy)
        if policy not in self._edge_policy_order():
            raise ValueError(f"Unsupported edge policy profile: {policy!r}")
        if not enabled:
            self._edge_profiles.pop(policy, None)
            return
        self._edge_profiles[policy] = {
            "prep_time_s": max(0.0, float(prep_time_s)),
            "serial_time_s": (
                max(0.0, float(serial_time_s))
                if serial_time_s is not None
                else max(0.0, float(prep_time_s))
            ),
            "overlap_time_s": max(0.0, float(overlap_time_s)),
            "gpu_peak_bytes": max(0, int(gpu_peak_bytes)),
            "cpu_delta_bytes": max(0, int(cpu_delta_bytes)),
            "live_edge_bytes": max(0, int(live_edge_bytes)),
        }

    def _edge_policy_profiles(self):
        if self._edge_profiles:
            return [
                (policy, self._edge_profiles[policy])
                for policy in self._edge_policy_order()
                if policy in self._edge_profiles
            ]
        profiles = [
            (
                self.EDGE_GPU_EPHEMERAL,
                {
                    "prep_time_s": max(0.0, float(self._t_h2d_edge)),
                    "serial_time_s": max(0.0, float(self._t_h2d_edge)),
                    "overlap_time_s": 0.0,
                    "gpu_peak_bytes": 0,
                    "cpu_delta_bytes": 0,
                    "live_edge_bytes": 0,
                },
            ),
        ]
        if self._edge_bytes > 0:
            profiles.append(
                (
                    self.EDGE_GPU_PERSIST,
                    {
                        "prep_time_s": 0.0,
                        "serial_time_s": 0.0,
                        "overlap_time_s": 0.0,
                        "gpu_peak_bytes": 0,
                        "cpu_delta_bytes": 0,
                        "live_edge_bytes": max(0, int(self._edge_bytes)),
                    },
                )
            )
        return profiles

    # ------------------------------------------------------------------
    # Public API for training loop
    # ------------------------------------------------------------------

    def notify_budget_frozen(self, reuse_deferred_baseline: bool = False) -> None:
        """Signal that edge budget is stable; begin profiling."""
        if self._state != self._DEFERRED:
            return
        _already_announced = False
        if reuse_deferred_baseline and self._deferred_baseline_ready:
            # Previously this skipped WARMUP_RECOMPUTE entirely and reused both
            # memory and timing from the deferred (budget-adjustment) phase.
            # That caused incorrect calibration: the runtime edge policy switches
            # at freeze time (cpu_broadcast → post-freeze policy), so backward
            # time can change by 1-2 s per step even with no tier change.
            # Reusing the old T_bwd as the recompute baseline made CALIBRATE_RETAIN
            # appear slower than recompute and caused the planner to always select
            # all-recompute.
            #
            # Fix: always run 1 fresh WARMUP_RECOMPUTE step with the post-freeze
            # edges before proceeding to the calibration stages.  The extra epoch
            # cost is unavoidable for a valid baseline.
            if self._is_rank0():
                print(
                    "[MultiTierManager] Edge budget frozen. "
                    "Running 1 fresh WARMUP_RECOMPUTE step with post-freeze edge "
                    "policy before calibration (deferred timing not reused)."
                )
            _already_announced = True
        self._warmup_target = 1
        self._warmup_peak_samples.clear()
        self._warmup_fwd_live_samples.clear()
        self._warmup_cpu_live_samples.clear()
        self._warmup_t_fwd_samples.clear()
        self._warmup_t_bwd_samples.clear()
        self._state = self._WARMUP_RECOMPUTE
        if self._is_rank0() and not _already_announced:
            print("[MultiTierManager] Edge budget frozen. Starting WARMUP_RECOMPUTE.")

    def global_offload_requested(self) -> bool:
        """multi_tier no longer uses an outer save_on_cpu wrapper."""
        return False

    def plan(self, device) -> None:
        """Set per-layer modes and reset peak stats at the start of each forward."""
        if not torch.cuda.is_available():
            return
        if self._state == self._DEFERRED:
            self._modes = [self.T0_RECOMPUTE] * self.n_layers
            torch.cuda.reset_peak_memory_stats(device)
            return
        if self._state == self._WARMUP_RECOMPUTE:
            self._modes = [self.T0_RECOMPUTE] * self.n_layers
            torch.cuda.reset_peak_memory_stats(device)
            if self._is_rank0():
                current_idx = len(self._warmup_peak_samples) + 1
                print(
                    f"[MultiTierManager] WARMUP_RECOMPUTE: all {self.n_layers} layers → recompute "
                    f"(sample {current_idx}/{self._warmup_target})"
                )
        elif self._state == self._CALIBRATE_MHA:
            modes = [self.T0_RECOMPUTE] * self.n_layers
            modes[-1] = self.T1_KEEP_MHA
            self._modes = modes
            torch.cuda.reset_peak_memory_stats(device)
            if self._is_rank0():
                print("[MultiTierManager] CALIBRATE_MHA: last layer → keep_mha, rest → recompute")
        elif self._state == self._CALIBRATE_MHA_FRONT:
            modes = [self.T0_RECOMPUTE] * self.n_layers
            modes[0] = self.T1_KEEP_MHA
            self._modes = modes
            torch.cuda.reset_peak_memory_stats(device)
            if self._is_rank0():
                print("[MultiTierManager] CALIBRATE_MHA_FRONT: first layer → keep_mha, rest → recompute")
        elif self._state == self._CALIBRATE_RETAIN:
            modes = [self.T0_RECOMPUTE] * self.n_layers
            modes[-1] = self.T4_RETAIN
            self._modes = modes
            torch.cuda.reset_peak_memory_stats(device)
            if self._is_rank0():
                print("[MultiTierManager] CALIBRATE_RETAIN: last layer → retain, rest → recompute")
        elif self._state == self._CALIBRATE_RETAIN_FRONT:
            modes = [self.T0_RECOMPUTE] * self.n_layers
            modes[0] = self.T4_RETAIN
            self._modes = modes
            torch.cuda.reset_peak_memory_stats(device)
            if self._is_rank0():
                print("[MultiTierManager] CALIBRATE_RETAIN_FRONT: first layer → retain, rest → recompute")
        # ACTIVE: modes already set by _apply_plan(); no-op.

    def record_post_forward_memory(self, device) -> None:
        """Snapshot live GPU memory after encoder forward, before backward.

        Must be called by the model's forward() after the encoder layer loop.
        Records the live activation footprint before backward allocations,
        enabling accurate per-tier memory measurement.
        """
        if not torch.cuda.is_available():
            return
        live = torch.cuda.memory_allocated(device)
        live = int(self._sync_max_scalar(int(live), device, torch.long))
        cpu_live = int(self._sync_max_scalar(int(self._process_rss_bytes()), device, torch.long))
        if self._state == self._DEFERRED:
            self._deferred_fwd_live = live
            self._deferred_cpu_live = cpu_live
        elif self._state == self._WARMUP_RECOMPUTE:
            self._fwd_live_warmup = live
            self._cpu_live_warmup = cpu_live
        elif self._state == self._CALIBRATE_MHA:
            self._fwd_live_mha = live
        elif self._state == self._CALIBRATE_MHA_FRONT:
            self._fwd_live_mha_front = live
        elif self._state == self._CALIBRATE_RETAIN:
            self._fwd_live_retain = live
        elif self._state == self._CALIBRATE_RETAIN_FRONT:
            self._fwd_live_retain_front = live

    def notify_step_end(self, device, t_bwd: float = None, t_fwd: float = None) -> None:
        """Advance the state machine after backward + optimizer.step().

        Must be called after optimizer.step() so optimizer state (lazily
        initialised on the first step) is included in peak measurements.
        """
        if not torch.cuda.is_available():
            return
        if self._state == self._ACTIVE:
            return
        is_r0 = self._is_rank0()
        real_total_gpu = torch.cuda.get_device_properties(device).total_memory
        total_gpu = self._effective_total_gpu_bytes(real_total_gpu)
        gpu_total_msg = self._gpu_total_msg(real_total_gpu, total_gpu)

        if self._state == self._DEFERRED:
            self._deferred_peak = int(
                self._sync_max_scalar(
                    int(torch.cuda.max_memory_allocated(device)),
                    device,
                    torch.long,
                )
            )
            if t_fwd is not None:
                self._deferred_t_fwd = float(
                    self._sync_max_scalar(float(t_fwd), device, torch.float64)
                )
            if t_bwd is not None:
                self._deferred_t_bwd = float(
                    self._sync_max_scalar(float(t_bwd), device, torch.float64)
                )
            self._deferred_baseline_ready = True
            return

        if self._state == self._WARMUP_RECOMPUTE:
            peak_sample = int(
                self._sync_max_scalar(
                    int(torch.cuda.max_memory_allocated(device)),
                    device,
                    torch.long,
                )
            )
            t_fwd_sample = (
                float(self._sync_max_scalar(float(t_fwd), device, torch.float64))
                if t_fwd is not None
                else None
            )
            t_bwd_sample = (
                float(self._sync_max_scalar(float(t_bwd), device, torch.float64))
                if t_bwd is not None
                else None
            )
            self._warmup_peak_samples.append(peak_sample)
            self._warmup_fwd_live_samples.append(int(self._fwd_live_warmup))
            self._warmup_cpu_live_samples.append(int(self._cpu_live_warmup))
            if t_fwd_sample is not None:
                self._warmup_t_fwd_samples.append(t_fwd_sample)
            if t_bwd_sample is not None:
                self._warmup_t_bwd_samples.append(t_bwd_sample)
            if len(self._warmup_peak_samples) < self._warmup_target:
                if is_r0:
                    print(
                        f"[MultiTierManager] WARMUP_RECOMPUTE sample "
                        f"{len(self._warmup_peak_samples)}/{self._warmup_target} recorded."
                    )
                return
            keep_start = 1 if self._warmup_target > 1 and len(self._warmup_peak_samples) > 1 else 0
            kept_peaks = self._warmup_peak_samples[keep_start:]
            kept_fwd_live = self._warmup_fwd_live_samples[keep_start:]
            kept_cpu_live = self._warmup_cpu_live_samples[keep_start:]
            kept_t_fwd = self._warmup_t_fwd_samples[keep_start:] if self._warmup_t_fwd_samples else []
            kept_t_bwd = self._warmup_t_bwd_samples[keep_start:] if self._warmup_t_bwd_samples else []
            self._peak_warmup = max(kept_peaks) if kept_peaks else peak_sample
            self._fwd_live_warmup = max(kept_fwd_live) if kept_fwd_live else int(self._fwd_live_warmup)
            self._cpu_live_warmup = max(kept_cpu_live) if kept_cpu_live else int(self._cpu_live_warmup)
            if kept_t_fwd:
                self._t_fwd_recompute = sum(kept_t_fwd) / len(kept_t_fwd)
            if kept_t_bwd:
                self._t_bwd_recompute = sum(kept_t_bwd) / len(kept_t_bwd)
            self._state = self._CALIBRATE_MHA
            if is_r0:
                keep_note = ""
                if self._warmup_target > 1:
                    keep_note = (
                        f"  samples={self._warmup_target} "
                        f"(dropped cold-start sample, averaged last {len(kept_t_fwd) or len(kept_peaks)})"
                    )
                print(
                    f"[MultiTierManager] WARMUP_RECOMPUTE done: "
                    f"peak={self._peak_warmup/(1024**2):.0f} MiB  "
                    f"T_fwd={self._t_fwd_recompute*1000:.0f} ms  "
                    f"T_bwd={self._t_bwd_recompute*1000:.0f} ms  "
                    f"({gpu_total_msg})"
                    f"{keep_note}"
                )

        elif self._state == self._CALIBRATE_MHA:
            _peak_mha_last = int(
                self._sync_max_scalar(
                    int(torch.cuda.max_memory_allocated(device)),
                    device,
                    torch.long,
                )
            )
            if t_fwd is not None:
                self._t_fwd_mha = float(
                    self._sync_max_scalar(float(t_fwd), device, torch.float64)
                )
            if t_bwd is not None:
                self._t_bwd_mha = float(
                    self._sync_max_scalar(float(t_bwd), device, torch.float64)
                )
            self._peak_mha = _peak_mha_last   # may be updated after FRONT probe
            m_mha_last = self._fwd_live_mha - self._fwd_live_warmup
            self._m_layer_mha = max(m_mha_last, 1)
            # Pre-check: FRONT probe requires first-layer MHA to persist through
            # the entire backward pass, coexisting with recomputation of all
            # subsequent layers. Worst-case peak ≈ peak_warmup + m_layer_mha.
            # If that exceeds 90% of physical GPU, skip the FRONT probe entirely
            # and use the last-layer measurement — safer than risking an OOM that
            # dirties the CUDA allocator and cascades into the next training step.
            _peak_est_front = self._peak_warmup + self._m_layer_mha
            _physical_gpu = torch.cuda.get_device_properties(device).total_memory
            if _peak_est_front > int(_physical_gpu * 0.90):
                if is_r0:
                    print(
                        f"[MultiTierManager] Skipping CALIBRATE_MHA_FRONT: "
                        f"est_peak={_peak_est_front/(1024**2):.0f} MiB > "
                        f"90% physical ({int(_physical_gpu*0.90)/(1024**2):.0f} MiB). "
                        f"Using last-layer M_layer^mha={m_mha_last/(1024**2):.1f} MiB."
                    )
                self._state = self._CALIBRATE_RETAIN
            else:
                self._state = self._CALIBRATE_MHA_FRONT
            if is_r0:
                print(
                    f"[MultiTierManager] CALIBRATE_MHA (last layer) done: "
                    f"M_layer^mha={m_mha_last/(1024**2):.1f} MiB  "
                    f"peak={_peak_mha_last/(1024**2):.0f} MiB  "
                    f"net_delta={((_peak_mha_last - self._peak_warmup)/(1024**2)):+.1f} MiB  "
                    f"T_fwd={self._t_fwd_mha*1000:.0f} ms  "
                    f"T_bwd={self._t_bwd_mha*1000:.0f} ms"
                )

        elif self._state == self._CALIBRATE_MHA_FRONT:
            _peak_mha_front = int(
                self._sync_max_scalar(
                    int(torch.cuda.max_memory_allocated(device)),
                    device,
                    torch.long,
                )
            )
            self._peak_mha_front = _peak_mha_front
            m_mha_front = self._fwd_live_mha_front - self._fwd_live_warmup
            # Consolidate: take conservative max of both probes.
            m_mha_last = self._m_layer_mha   # stored from previous state
            m_mha = max(m_mha_last, m_mha_front)
            if m_mha <= 0:
                m_mha = max(int(self._peak_warmup * 0.10), 1)
                if is_r0:
                    print(
                        f"[MultiTierManager] WARNING: M_layer^mha non-positive after both probes; "
                        f"fallback M_layer^mha={m_mha/(1024**2):.1f} MiB"
                    )
            self._m_layer_mha = m_mha
            self._peak_mha = max(self._peak_mha, _peak_mha_front)
            self._state = self._CALIBRATE_RETAIN
            if is_r0:
                print(
                    f"[MultiTierManager] CALIBRATE_MHA_FRONT (first layer) done: "
                    f"M_layer^mha_front={m_mha_front/(1024**2):.1f} MiB  "
                    f"peak_front={_peak_mha_front/(1024**2):.0f} MiB  "
                    f"→ M_layer^mha(max)={m_mha/(1024**2):.1f} MiB  "
                    f"peak(max)={self._peak_mha/(1024**2):.0f} MiB"
                )

        elif self._state == self._CALIBRATE_RETAIN:
            _peak_retain_last = int(
                self._sync_max_scalar(
                    int(torch.cuda.max_memory_allocated(device)),
                    device,
                    torch.long,
                )
            )
            if t_fwd is not None:
                self._t_fwd_retain = float(
                    self._sync_max_scalar(float(t_fwd), device, torch.float64)
                )
            if t_bwd is not None:
                self._t_bwd_retain = float(
                    self._sync_max_scalar(float(t_bwd), device, torch.float64)
                )
            self._peak_retain = _peak_retain_last   # may be updated after FRONT probe
            m_full_last = self._fwd_live_retain - self._fwd_live_warmup
            self._m_layer_full = max(m_full_last, 1)
            if is_r0:
                print(
                    f"[MultiTierManager] CALIBRATE_RETAIN (last layer) done: "
                    f"M_layer^full={m_full_last/(1024**2):.1f} MiB  "
                    f"peak={_peak_retain_last/(1024**2):.0f} MiB  "
                    f"net_delta={((_peak_retain_last - self._peak_warmup)/(1024**2)):+.1f} MiB  "
                    f"T_fwd={self._t_fwd_retain*1000:.0f} ms  "
                    f"T_bwd={self._t_bwd_retain*1000:.0f} ms"
                )
            # Pre-check (mirrors CALIBRATE_MHA → FRONT): retaining the FIRST layer
            # keeps its full activation live through the entire backward pass while
            # every later layer is recomputed, so worst-case peak ≈ peak_warmup +
            # m_layer_full. Retain is strictly heavier than keep_mha
            # (m_full >= m_mha), so if that estimate exceeds 90% of physical GPU the
            # FRONT probe is almost certain to OOM. An OOM here is especially costly:
            # it fragments the CUDA allocator and, as observed in practice, leaves
            # even the all-recompute fallback unable to fit on the retry. Skip the
            # probe entirely and finalise with the last-layer measurement instead.
            _peak_est_front = self._peak_warmup + self._m_layer_full
            _physical_gpu = torch.cuda.get_device_properties(device).total_memory
            if _peak_est_front > int(_physical_gpu * 0.90):
                if is_r0:
                    print(
                        f"[MultiTierManager] Skipping CALIBRATE_RETAIN_FRONT: "
                        f"est_peak={_peak_est_front/(1024**2):.0f} MiB > "
                        f"90% physical ({int(_physical_gpu*0.90)/(1024**2):.0f} MiB). "
                        f"Using last-layer M_layer^full={m_full_last/(1024**2):.1f} MiB."
                    )
                self._run_active_plan(device, total_gpu)
            else:
                self._state = self._CALIBRATE_RETAIN_FRONT

        elif self._state == self._CALIBRATE_RETAIN_FRONT:
            _peak_retain_front = int(
                self._sync_max_scalar(
                    int(torch.cuda.max_memory_allocated(device)),
                    device,
                    torch.long,
                )
            )
            self._peak_retain_front = _peak_retain_front
            m_full_front = self._fwd_live_retain_front - self._fwd_live_warmup
            m_full_last = self._m_layer_full   # stored from previous state
            m_full = max(m_full_last, m_full_front)
            if m_full <= 0:
                m_full = max(int(self._m_layer_mha * 2), 1)
                if is_r0:
                    print(
                        f"[MultiTierManager] WARNING: M_layer^full non-positive after both probes; "
                        f"fallback M_layer^full={m_full/(1024**2):.1f} MiB"
                    )
            self._m_layer_full = m_full
            self._peak_retain = max(self._peak_retain, _peak_retain_front)
            if is_r0:
                print(
                    f"[MultiTierManager] CALIBRATE_RETAIN_FRONT (first layer) done: "
                    f"M_layer^full_front={m_full_front/(1024**2):.1f} MiB  "
                    f"peak_front={_peak_retain_front/(1024**2):.0f} MiB  "
                    f"→ M_layer^full(max)={m_full/(1024**2):.1f} MiB  "
                    f"peak(max)={self._peak_retain/(1024**2):.0f} MiB"
                )
            self._run_active_plan(device, total_gpu)

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    def _run_active_plan(self, device, total_gpu: int) -> None:
        """Search structured R/O/K/N plans under GPU/CPU constraints.

        Feasibility uses two independent memory budgets:

        * **transient_limit** – peak allowed during the edge-build phase.
          With expandable_segments the build tensors are fully released
          before the model forward begins, so the two phases never compete
          for memory simultaneously.  The build phase therefore gets a
          relaxed limit (97 % of total_gpu vs the conservative peak_limit).
          Without expandable_segments we fall back to the same peak_limit
          to avoid fragmentation-driven OOMs.

        * **peak_limit** – peak allowed during the model forward+backward
          phase (includes activations, edge_persist bytes, etc.).  Always
          capped at ``total_gpu * (1 - safety_margin)``.

        The primary optimisation objective is minimum estimated step time
        ``t_est = t_model_est + edge_time``, where ``edge_time`` already
        accounts for CPU-prefetch overlap with the model.
        """
        del device
        is_r0 = self._is_rank0()
        cpu_budget = self._available_cpu_bytes()
        peak_limit = max(int(total_gpu * (1.0 - self.safety_margin)), int(self._peak_warmup))

        # Separate transient budget for the edge-build phase.
        _exp_seg = False
        try:
            from gt_sp.utils import _expandable_segments_enabled as _check_exp_seg
            _exp_seg = _check_exp_seg()
        except Exception:
            pass
        # 97 % leaves a small buffer for measurement noise while still being
        # significantly more permissive than peak_limit (85 % by default).
        transient_limit = int(total_gpu * 0.97) if _exp_seg else peak_limit

        best_key = None
        best_step_time = float("inf")
        best_model_time = float("inf")
        best_fwd_time = 0.0
        best_bwd_time = 0.0
        best_edge_prep_time = 0.0
        best_policy = self.EDGE_GPU_EPHEMERAL
        best_nonpersistent_key = None
        best_nonpersistent_policy = self.EDGE_GPU_EPHEMERAL
        best_nonpersistent_tiers = [self.T0_RECOMPUTE] * self.n_layers
        best_tiers = [self.T0_RECOMPUTE] * self.n_layers
        # Per-policy decision log: policy → (status_str, best_t_ms or None)
        policy_log: dict = {}  # policy -> (status_str, best_t_ms_or_none)

        for edge_policy, edge_profile in self._edge_policy_profiles():
            edge_peak = int(edge_profile.get("live_edge_bytes", 0))
            edge_serial_time = float(
                edge_profile.get(
                    "serial_time_s",
                    edge_profile.get("prep_time_s", 0.0),
                )
            )
            edge_overlap_time = float(edge_profile.get("overlap_time_s", 0.0))
            prep_gpu_peak = int(edge_profile.get("gpu_peak_bytes", 0))
            prep_cpu_delta = int(edge_profile.get("cpu_delta_bytes", 0))

            # --- Transient feasibility (build phase) ---
            if prep_gpu_peak > transient_limit:
                policy_log[edge_policy] = (
                    f"skip:build_peak={prep_gpu_peak/(1024**2):.0f}>"
                    f"transient_limit={transient_limit/(1024**2):.0f} MiB",
                    None,
                )
                continue
            if prep_cpu_delta > cpu_budget:
                policy_log[edge_policy] = (
                    f"skip:cpu_delta={prep_cpu_delta/(1024**2):.0f}>"
                    f"cpu_budget={cpu_budget/(1024**2):.0f} MiB",
                    None,
                )
                continue

            policy_best_t = float("inf")
            for tiers, k_off, cpu_est, peak_est, t_fwd_est, t_bwd_est, t_model_est in self._enumerate_structured_plans(
                peak_limit=peak_limit,
                cpu_budget=cpu_budget,
                edge_peak=edge_peak,
            ):
                edge_time = edge_serial_time + max(0.0, edge_overlap_time - t_model_est)
                t_est = t_model_est + edge_time
                policy_best_t = min(policy_best_t, t_est)
                key = (
                    t_est,
                    k_off,
                    cpu_est,
                    int(edge_peak > 0),
                    prep_cpu_delta,
                    prep_gpu_peak,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_step_time = t_est
                    best_model_time = t_model_est
                    best_fwd_time = t_fwd_est
                    best_bwd_time = t_bwd_est
                    best_edge_prep_time = edge_time
                    best_policy = edge_policy
                    best_tiers = tiers
                if edge_peak == 0 and (best_nonpersistent_key is None or key < best_nonpersistent_key):
                    best_nonpersistent_key = key
                    best_nonpersistent_policy = edge_policy
                    best_nonpersistent_tiers = tiers

            if policy_best_t == float("inf"):
                policy_log[edge_policy] = (
                    f"skip:no_feasible_tier(model_peak>peak_limit={peak_limit/(1024**2):.0f} MiB)",
                    None,
                )
            else:
                policy_log[edge_policy] = ("ok", policy_best_t * 1000.0)

        if best_key is None:
            best_policy = best_nonpersistent_policy
            best_tiers = [self.T0_RECOMPUTE] * self.n_layers
            best_nonpersistent_tiers = best_tiers
            best_fwd_time = self._t_fwd_recompute
            best_bwd_time = self._t_bwd_recompute
            best_model_time = best_fwd_time + best_bwd_time
            best_step_time = best_model_time

        self._edge_policy = best_policy
        self._best_nonpersistent_edge_policy = best_nonpersistent_policy
        self._best_nonpersistent_tiers = best_nonpersistent_tiers
        self._cache_edge = best_policy == self.EDGE_GPU_PERSIST
        self._modes = best_tiers
        self._state = self._ACTIVE

        if is_r0:
            tier_counts = {}
            for t in best_tiers:
                tier_counts[t] = tier_counts.get(t, 0) + 1
            # Build per-policy summary line: show estimated step time or skip reason.
            policy_parts = []
            for p, (status, best_t_ms) in policy_log.items():
                if status == "ok":
                    marker = " [SELECTED]" if p == best_policy else ""
                    policy_parts.append(f"{p}:{best_t_ms:.0f}ms{marker}")
                else:
                    policy_parts.append(f"{p}:{status}")
            print(
                f"[MultiTierManager] ACTIVE plan: "
                f"edge_policy={best_policy}  "
                f"tiers={tier_counts}  "
                f"peak_limit={peak_limit/(1024**2):.0f} MiB  "
                f"transient_limit={transient_limit/(1024**2):.0f} MiB"
                f"{'(relaxed)' if _exp_seg else ''}  "
                f"net_delta(mha/retain)={(self._peak_mha - self._peak_warmup)/(1024**2):+.1f}"
                f"/{(self._peak_retain - self._peak_warmup)/(1024**2):+.1f} MiB  "
                f"est_T_step={best_step_time*1000:.0f} ms "
                f"(model={best_model_time*1000:.0f} ms "
                f"fwd={best_fwd_time*1000:.0f} ms "
                f"bwd={best_bwd_time*1000:.0f} ms "
                f"edge_prep={best_edge_prep_time*1000:.0f} ms)\n"
                f"[MultiTierManager]   policy candidates: {' | '.join(policy_parts)}"
            )

    def _enumerate_structured_plans(self, peak_limit: int, cpu_budget: int, edge_peak: int):
        """Yield feasible [R][K][N] structured plans.

        Recompute fills the prefix, keep_mha forms a suffix, and retain
        occupies the deepest suffix.  T2_OFFLOAD is excluded: save_on_cpu
        does not reduce forward-pass peak GPU memory vs T0_RECOMPUTE (all
        activations are computed on GPU before the CPU transfer), and the
        synchronous non-pinned PCIe transfer in the backward pass is pure
        overhead compared to recomputation.
        """
        L = self.n_layers
        if L == 0 or self._t_bwd_recompute <= 0:
            return []

        t_f_r = self._t_fwd_recompute / L if self._t_fwd_recompute > 0 else 0.0
        t_f_k = max(t_f_r + (self._t_fwd_mha - self._t_fwd_recompute), 0.0)
        t_f_n = max(t_f_r + (self._t_fwd_retain - self._t_fwd_recompute), 0.0)
        t_r = self._t_bwd_recompute / L
        t_k = max(t_r + (self._t_bwd_mha - self._t_bwd_recompute), 0.0)
        t_n = max(t_r + (self._t_bwd_retain - self._t_bwd_recompute), 0.0)

        m_full = int(self._m_layer_full)
        m_mha = int(self._m_layer_mha)
        mha_delta = self._peak_mha - self._peak_warmup
        retain_delta = self._peak_retain - self._peak_warmup
        # 5% per-additional-layer fragmentation factor: the memory allocator
        # cannot always pack multiple live activation tensors contiguously, so
        # the measured single-layer delta under-estimates the true peak for
        # k > 1 layers of the same tier.
        _FRAG_FACTOR = 0.05

        candidates = []
        for k_keep in range(L + 1):
            if k_keep > 0 and self._t1_infeasible:
                continue
            for k_ret in range(L - k_keep + 1):
                if k_ret > 0 and self._t4_infeasible:
                    continue
                k_rec = L - k_keep - k_ret
                extra_k = max(0, k_keep - 1)
                extra_r = max(0, k_ret - 1)
                m_mha_scaled = int(m_mha * (1.0 + _FRAG_FACTOR * extra_k))
                m_full_scaled = int(m_full * (1.0 + _FRAG_FACTOR * extra_r))
                mha_extra = (mha_delta + extra_k * m_mha_scaled) if k_keep > 0 else 0
                retain_extra = (retain_delta + extra_r * m_full_scaled) if k_ret > 0 else 0
                net_fwd_extra = max(0, mha_extra + retain_extra)
                peak_est = self._peak_warmup + edge_peak + net_fwd_extra
                if peak_est > peak_limit:
                    continue
                tiers = (
                    [self.T0_RECOMPUTE] * k_rec
                    + [self.T1_KEEP_MHA] * k_keep
                    + [self.T4_RETAIN] * k_ret
                )
                t_fwd_est = (k_rec * t_f_r) + (k_keep * t_f_k) + (k_ret * t_f_n)
                t_bwd_est = (k_rec * t_r) + (k_keep * t_k) + (k_ret * t_n)
                t_model_est = t_fwd_est + t_bwd_est
                candidates.append((tiers, 0, 0, peak_est, t_fwd_est, t_bwd_est, t_model_est))
        return candidates

    def mode(self, layer_idx: int) -> str:
        return self._modes[layer_idx]

    def is_active(self) -> bool:
        return self._state == self._ACTIVE

    @property
    def state_name(self) -> str:
        names = {
            self._DEFERRED: "DEFERRED",
            self._WARMUP_RECOMPUTE: "WARMUP_RECOMPUTE",
            self._CALIBRATE_MHA: "CALIBRATE_MHA",
            self._CALIBRATE_MHA_FRONT: "CALIBRATE_MHA_FRONT",
            self._CALIBRATE_RETAIN: "CALIBRATE_RETAIN",
            self._CALIBRATE_RETAIN_FRONT: "CALIBRATE_RETAIN_FRONT",
            self._ACTIVE: "ACTIVE",
        }
        return names.get(self._state, f"UNKNOWN({self._state})")

    def needs_precise_timing(self) -> bool:
        """Return True while multi_tier is still calibrating its cost model."""
        return self._state not in (self._DEFERRED, self._ACTIVE)

    @property
    def cache_edge(self) -> bool:
        return self._cache_edge

    @property
    def edge_policy(self) -> str:
        return self._edge_policy

    @property
    def best_nonpersistent_edge_policy(self) -> str:
        return self._best_nonpersistent_edge_policy

    @property
    def best_nonpersistent_tiers(self) -> list:
        return list(self._best_nonpersistent_tiers)

    def override_active_plan(self, edge_policy: str, modes: list[str]) -> None:
        """Apply a cross-rank-synchronised ACTIVE plan."""
        if self._state != self._ACTIVE:
            return
        edge_policy = str(edge_policy)
        if edge_policy not in (
            self.EDGE_GPU_PERSIST,
            self.EDGE_GPU_EPHEMERAL,
            self.EDGE_CPU_RANK_LOCAL_PREFETCH,
            self.EDGE_CPU_BROADCAST_PREFETCH,
            self.EDGE_CPU_BROADCAST,
        ):
            raise ValueError(f"Unsupported multi_tier edge policy: {edge_policy!r}")
        modes = list(modes)
        if len(modes) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} multi_tier modes, got {len(modes)}"
            )
        valid_modes = {
            self.T0_RECOMPUTE,
            self.T1_KEEP_MHA,
            self.T2_OFFLOAD,
            self.T4_RETAIN,
        }
        if any(m not in valid_modes for m in modes):
            raise ValueError(f"Unsupported multi_tier mode list: {modes!r}")
        self._edge_policy = edge_policy
        self._cache_edge = edge_policy == self.EDGE_GPU_PERSIST
        self._modes = modes
        if self._is_rank0():
            tier_counts = {}
            for t in modes:
                tier_counts[t] = tier_counts.get(t, 0) + 1
            print(
                f"[MultiTierManager] Synced ACTIVE plan: "
                f"edge_policy={self._edge_policy} tiers={tier_counts}"
            )

    def reconsider_with_actual_timing(
        self,
        actual_t_model_s: float,
        actual_prefetch_wait_s: float,
        switch_threshold_s: float = 0.5,
    ) -> bool:
        """Re-evaluate the ACTIVE plan using observed runtime timing.

        Called after several ACTIVE epochs when a CPU-prefetch policy is
        selected.  The profiling-time overlap estimate can be wrong because:
          (a) actual model time differs from calibration (different tier mix,
              GPU-only pressure without competing CPU build);
          (b) actual CPU build time is slower than profiling (system load,
              memory pressure from enlarged RSS).

        Using the *observed* ``actual_t_model_s`` + ``actual_prefetch_wait_s``
        gives a fair, apples-to-apples comparison with GPU policies.

        Returns True if the plan was updated (caller should re-apply the edge
        placement decision).  Returns False when no switch is warranted or
        this manager is not in ACTIVE state.
        """
        if self._state != self._ACTIVE:
            return False
        current_policy = self._edge_policy
        _CPU_PREFETCH = (self.EDGE_CPU_BROADCAST_PREFETCH, self.EDGE_CPU_RANK_LOCAL_PREFETCH)
        if current_policy not in _CPU_PREFETCH:
            return False
        if not self._edge_profiles:
            return False

        # Actual effective step time under the current CPU-prefetch plan:
        #   t_step = t_model + prefetch_wait + stage_time
        current_profile = self._edge_profiles.get(current_policy, {})
        stage_time = float(current_profile.get("serial_time_s", 0.0))
        actual_cpu_t_step = float(actual_t_model_s) + float(actual_prefetch_wait_s) + stage_time

        # Best GPU alternative: t_step = t_model + gpu_build_time (serial, no overlap)
        best_gpu_policy = None
        best_gpu_t_step = float("inf")
        for gpu_pol in (self.EDGE_GPU_EPHEMERAL, self.EDGE_GPU_PERSIST):
            prof = self._edge_profiles.get(gpu_pol)
            if prof is None:
                continue
            gpu_serial = float(prof.get("serial_time_s", prof.get("prep_time_s", 0.0)))
            gpu_t_step = float(actual_t_model_s) + gpu_serial
            if gpu_t_step < best_gpu_t_step:
                best_gpu_t_step = gpu_t_step
                best_gpu_policy = gpu_pol

        if best_gpu_policy is None:
            return False

        gain_s = actual_cpu_t_step - best_gpu_t_step
        is_r0 = self._is_rank0()
        if gain_s <= switch_threshold_s:
            if is_r0:
                print(
                    f"[MultiTierManager] adaptive-timing check: "
                    f"current={current_policy} actual_t={actual_cpu_t_step*1000:.0f} ms "
                    f"(model={actual_t_model_s*1000:.0f} ms + wait={actual_prefetch_wait_s*1000:.0f} ms + stage={stage_time*1000:.0f} ms) "
                    f"vs best_gpu={best_gpu_policy} est_t={best_gpu_t_step*1000:.0f} ms — "
                    f"gain={gain_s*1000:.0f} ms < threshold={switch_threshold_s*1000:.0f} ms, keeping current plan."
                )
            return False

        # GPU is significantly better → switch, keeping same tier configuration.
        if is_r0:
            tier_counts = {}
            for t in self._modes:
                tier_counts[t] = tier_counts.get(t, 0) + 1
            print(
                f"[MultiTierManager] adaptive-timing switch: "
                f"{current_policy} actual_t={actual_cpu_t_step*1000:.0f} ms "
                f"(model={actual_t_model_s*1000:.0f} ms + wait={actual_prefetch_wait_s*1000:.0f} ms + stage={stage_time*1000:.0f} ms) "
                f"→ {best_gpu_policy} est_t={best_gpu_t_step*1000:.0f} ms "
                f"(saving ~{gain_s*1000:.0f} ms/epoch). "
                f"Tiers unchanged: {tier_counts}."
            )
        self._edge_policy = best_gpu_policy
        self._cache_edge = best_gpu_policy == self.EDGE_GPU_PERSIST
        return True

    def override_cache_decision(self, cache: bool) -> None:
        """Apply the cross-rank-synced topology cache decision."""
        if self._state != self._ACTIVE:
            return
        self._cache_edge = bool(cache)
        self._edge_policy = (
            self.EDGE_GPU_PERSIST if self._cache_edge else self._best_nonpersistent_edge_policy
        )
        if self._is_rank0():
            print(
                f"[MultiTierManager] Topology cache (post-sync): {cache} "
                f"(edge_policy={self._edge_policy})"
            )

    def set_edge_info(self, edge_bytes: int, t_h2d: float) -> None:
        """Register edge_index size and H2D time for joint topology decision."""
        self._edge_bytes  = max(0, int(edge_bytes))
        self._t_h2d_edge  = max(0.0, float(t_h2d))

    def mark_current_calibration_infeasible(self, device) -> bool:
        """Disable the tier currently being calibrated and finalise the plan.

        Called when a CALIBRATE_* step OOM'd before notify_step_end could
        fire. Per-layer GPU memory ordering is T0 < T1 < T4, so an OOM at
        any T1 probe guarantees T4 at the same / harder position would also
        OOM — we skip the remaining heavier probes and jump straight to
        finalising the ACTIVE plan with the surviving tiers (always T0,
        possibly T1 if it had already passed both MHA probes).

        Returns True if the state was advanced (caller should retry the
        training step under the new plan). Returns False when the OOM
        happened in a state that has no further fallback — DEFERRED,
        WARMUP_RECOMPUTE (all-recompute itself does not fit), or ACTIVE
        (use apply_oom_fallback instead).
        """
        if not torch.cuda.is_available():
            return False
        if self._state in (self._DEFERRED, self._WARMUP_RECOMPUTE, self._ACTIVE):
            return False

        real_total_gpu = torch.cuda.get_device_properties(device).total_memory
        total_gpu = self._effective_total_gpu_bytes(real_total_gpu)
        is_r0 = self._is_rank0()

        if self._state in (self._CALIBRATE_MHA, self._CALIBRATE_MHA_FRONT):
            # Cheapest non-T0 tier failed → heavier T4 cannot fit either.
            # Disable both and finalise with all-T0.
            self._t1_infeasible = True
            self._t4_infeasible = True
            if is_r0:
                print(
                    f"[MultiTierManager] OOM in {self.state_name}: "
                    f"T1 failed → heavier T4 also unfit. Disabling both. "
                    f"Finalising ACTIVE plan (all-recompute)."
                )
            # Flush the caching allocator before finalising so the immediately
            # following retry step starts from a clean memory state, not the
            # fragmented state left behind by the OOM.
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            self._run_active_plan(device, total_gpu)
            return True

        if self._state in (self._CALIBRATE_RETAIN, self._CALIBRATE_RETAIN_FRONT):
            # T1 probes had already passed, so T1 remains in the search space.
            self._t4_infeasible = True
            if is_r0:
                print(
                    f"[MultiTierManager] OOM in {self.state_name}: "
                    f"T4_RETAIN disabled. Finalising ACTIVE plan with T0/T1."
                )
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            self._run_active_plan(device, total_gpu)
            return True

        return False

    def apply_oom_fallback(self, device, sp_group=None, sp_world_size: int = 1) -> bool:
        """Downgrade the most expensive tier by one step after an OOM event.

        Traverses layers from last to first and demotes the first T4_RETAIN
        found to T1_KEEP_MHA, or the first T1_KEEP_MHA found to T0_RECOMPUTE.
        Returns True if a demotion was made, False if all layers are already
        T0_RECOMPUTE (no further fallback possible).

        Syncs the updated mode vector across SP ranks so all ranks apply the
        same plan (MIN reduction → most conservative rank wins).
        """
        if self._state != self._ACTIVE:
            return False
        _tier_order = {self.T4_RETAIN: 2, self.T1_KEEP_MHA: 1, self.T0_RECOMPUTE: 0}
        _demote = {self.T4_RETAIN: self.T1_KEEP_MHA, self.T1_KEEP_MHA: self.T0_RECOMPUTE}
        new_modes = list(self._modes)
        demoted = False
        for i in range(len(new_modes) - 1, -1, -1):
            if new_modes[i] in _demote:
                new_modes[i] = _demote[new_modes[i]]
                demoted = True
                break

        # Sync across SP ranks: use MIN so the most conservative rank wins.
        if sp_world_size > 1 and dist.is_initialized() and sp_group is not None:
            _to_id = {self.T0_RECOMPUTE: 0, self.T1_KEEP_MHA: 1, self.T4_RETAIN: 2}
            _to_mode = {v: k for k, v in _to_id.items()}
            ids = torch.tensor(
                [_to_id.get(m, 0) for m in new_modes],
                device=device, dtype=torch.long,
            )
            dist.all_reduce(ids, op=dist.ReduceOp.MIN, group=sp_group)
            new_modes = [_to_mode.get(int(v), self.T0_RECOMPUTE) for v in ids.tolist()]
            demoted_sync = torch.tensor([int(demoted)], device=device, dtype=torch.long)
            dist.all_reduce(demoted_sync, op=dist.ReduceOp.MAX, group=sp_group)
            demoted = bool(demoted_sync.item())

        self._modes = new_modes
        if self._is_rank0():
            tier_counts: dict = {}
            for t in new_modes:
                tier_counts[t] = tier_counts.get(t, 0) + 1
            print(f"[MultiTierManager] OOM fallback applied: tiers={tier_counts}")
        return demoted
