"""Host CPU + arithmetic NPU/GPU utilization helpers.

These three numbers populate ``ProfilingReport.utilization`` so callers
no longer see ``not_measured:requires_profiler_capture`` for the common
case. They are not a substitute for Snapdragon Profiler / QHAS — see
``host_power.py`` for the same caveat about coarse vs precise tooling.
"""
from __future__ import annotations

try:
    import psutil
except ImportError:  # pragma: no cover - in [real] extras
    psutil = None


# Hexagon clock frequency assumptions (Hz). Roughly correct for the
# Snapdragon X Elite Hexagon V73; different SoCs would require a small
# clock-table lookup. Used only when wall-clock timing is available.
_HEXAGON_BASE_HZ_BY_ARCH = {
    "V66": 1_000_000_000,
    "V68": 1_300_000_000,
    "V69": 1_500_000_000,
    "V73": 1_500_000_000,   # X Elite
    "V75": 1_800_000_000,   # 8-Gen-3 / future
    "V79": 2_000_000_000,
    "V81": 2_200_000_000,
}


def cpu_percent_blocking(interval_s: float = 0.1) -> float:
    """psutil.cpu_percent across all cores, averaged over ``interval_s``.

    Returns 0.0 if psutil is unavailable.
    """
    if psutil is None:
        return 0.0
    try:
        return float(psutil.cpu_percent(interval=interval_s))
    except Exception:
        return 0.0


def npu_utilization_from_cycles(
    *,
    total_cycles: int,
    wall_time_us: float,
    hexagon_arch: str = "V73",
) -> float:
    """Estimate NPU utilization from linting cycle counts.

    util_pct = total_cycles / (wall_time_s * hex_clock_hz) * 100

    Returns a value in 0–100. Returns 0.0 when inputs are non-positive
    or the architecture isn't in the clock table.
    """
    if total_cycles <= 0 or wall_time_us <= 0:
        return 0.0
    clock = _HEXAGON_BASE_HZ_BY_ARCH.get(hexagon_arch.upper())
    if not clock:
        return 0.0
    wall_s = wall_time_us / 1_000_000.0
    pct = (total_cycles / (wall_s * clock)) * 100.0
    # Clamp to a sane range — values > 100 % indicate clock mismatch.
    return float(max(0.0, min(pct, 100.0)))


def gpu_utilization_from_chrometrace(trace_path: str) -> float:
    """Best-effort GPU utilisation from a QHAS chrometrace JSON.

    Sums the ``dur`` fields of events tagged ``cat=="GPU"`` and divides
    by the trace span. Returns 0.0 if the file is missing or malformed.
    """
    import json
    from pathlib import Path

    path = Path(trace_path)
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    events = data.get("traceEvents") or data
    if not isinstance(events, list):
        return 0.0
    gpu_total = 0.0
    span_min = float("inf")
    span_max = 0.0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if (ev.get("cat") or "").upper() != "GPU":
            continue
        try:
            dur = float(ev.get("dur", 0))
            ts = float(ev.get("ts", 0))
        except (TypeError, ValueError):
            continue
        gpu_total += dur
        span_min = min(span_min, ts)
        span_max = max(span_max, ts + dur)
    if span_max <= span_min:
        return 0.0
    span = span_max - span_min
    return float(min(100.0, (gpu_total / span) * 100.0))


__all__ = [
    "cpu_percent_blocking",
    "npu_utilization_from_cycles",
    "gpu_utilization_from_chrometrace",
]
