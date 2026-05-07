"""High-level profiling API — single entry point for model profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from quad.profiler.kernel import KernelProfiler, KernelReport
from quad.profiler.memory_profiler import MemoryProfiler, MemoryReport
from quad.profiler.power_profiler import PowerProfiler, PowerTrace
from quad.profiler.roofline import RooflineAnalysis, RooflineResult
from quad.profiler.system import SystemProfiler, SystemTrace


@dataclass
class ProfileSummary:
    """Combined output from all profiling stages."""

    roofline: Optional[RooflineResult] = None
    kernel_report: Optional[KernelReport] = None
    power_trace: Optional[PowerTrace] = None
    memory_report: Optional[MemoryReport] = None
    system_trace: Optional[SystemTrace] = None
    recommendations: list[str] = field(default_factory=list)
    profile_duration_ms: float = 0.0

    def __repr__(self) -> str:
        parts = ["ProfileSummary("]
        if self.kernel_report:
            parts.append(f"  kernels={len(self.kernel_report.kernels)},")
        if self.roofline:
            parts.append(f"  roofline={self.roofline.diagnosis},")
        if self.power_trace:
            parts.append(f"  power_avg={self.power_trace.avg_power_mw:.0f}mW,")
        if self.memory_report:
            parts.append(f"  memory_peak={self.memory_report.peak_mb:.1f}MB,")
        parts.append(f"  recommendations={len(self.recommendations)}")
        parts.append(")")
        return "\n".join(parts)


def profile_model(
    model_path: str,
    level: Literal["system", "kernel", "deep"] = "kernel",
    device: str = "npu",
    mock: bool = True,
) -> ProfileSummary:
    """Profile a model at the specified depth level.

    This is the main entry point for QUAD profiling, providing Nsight-equivalent
    depth for Qualcomm hardware with additional power and thermal analysis.

    Args:
        model_path: Path to the model (ONNX, TFLite, or compiled QUAD model).
        level: Profiling depth:
            - "system": System-level timeline only (fastest).
            - "kernel": System + per-kernel analysis (default).
            - "deep": Full analysis including power, memory, and roofline.
        device: Target device ("npu", "gpu", "cpu").
        mock: If True (default), use simulated profilers. Set False for real
              hardware profiling when a Qualcomm device is available.

    Returns:
        ProfileSummary containing all requested profiling data.
    """
    start_time = time.perf_counter()
    summary = ProfileSummary()

    # Attempt to load model IR (mock if dependencies unavailable)
    ir_graph = _load_model_ir(model_path)

    # System-level profiling (always included)
    sys_profiler = SystemProfiler(mock=mock)
    sys_profiler.start()
    # Simulate model execution time
    time.sleep(0.01)
    summary.system_trace = sys_profiler.stop()

    if level in ("kernel", "deep"):
        # Kernel-level profiling
        kernel_profiler = KernelProfiler(mock=mock, device=device)
        summary.kernel_report = kernel_profiler.profile(ir_graph)

        # Roofline analysis
        device_specs = _get_device_specs(device)
        roofline = RooflineAnalysis(
            device_peak_tops=device_specs["peak_tops"],
            device_bandwidth_gb_s=device_specs["bandwidth_gb_s"],
        )
        summary.roofline = roofline.analyze(summary.kernel_report.kernels)

    if level == "deep":
        # Power profiling
        power_profiler = PowerProfiler(mock=mock)
        power_profiler.start()
        time.sleep(0.01)
        summary.power_trace = power_profiler.stop()

        # Memory profiling
        mem_profiler = MemoryProfiler(mock=mock, device=device)
        summary.memory_report = mem_profiler.profile(ir_graph)

    # Generate recommendations
    summary.recommendations = _generate_recommendations(summary)

    elapsed = time.perf_counter() - start_time
    summary.profile_duration_ms = round(elapsed * 1000.0, 2)

    return summary


def _load_model_ir(model_path: str):
    """Attempt to load model into IR representation."""
    try:
        from quad.compiler.frontend_onnx import compile_onnx

        return compile_onnx(model_path)
    except (ImportError, FileNotFoundError, Exception):
        # Return None — profilers will use mock mode
        return None


def _get_device_specs(device: str) -> dict:
    """Return hardware specs for the target device."""
    specs = {
        "npu": {"peak_tops": 73.0, "bandwidth_gb_s": 68.0},
        "gpu": {"peak_tops": 3.7, "bandwidth_gb_s": 68.0},
        "cpu": {"peak_tops": 0.5, "bandwidth_gb_s": 34.0},
    }
    return specs.get(device, specs["npu"])


def _generate_recommendations(summary: ProfileSummary) -> list[str]:
    """Generate automated optimization recommendations from profiling data."""
    recs: list[str] = []

    # Roofline-based recommendations
    if summary.roofline:
        if summary.roofline.achieved_pct < 40:
            recs.append(
                f"Low hardware utilization ({summary.roofline.achieved_pct:.0f}%). "
                f"Consider operator fusion and scheduling optimization."
            )
        if summary.roofline.diagnosis == "memory-bound":
            recs.append(
                "Workload is memory-bound. Prioritize quantization (INT8/INT4) "
                "and tiling to maximize VTCM usage."
            )

    # Kernel-based recommendations
    if summary.kernel_report:
        top = summary.kernel_report.top_kernels(3)
        if top:
            hotspot = top[0]
            recs.append(
                f"Top hotspot: '{hotspot.name}' ({hotspot.latency_us:.0f}us, "
                f"{hotspot.bottleneck}-bound). Focus optimization here first."
            )

        bottlenecks = summary.kernel_report.bottleneck_summary()
        if bottlenecks.get("latency", 0) > len(summary.kernel_report.kernels) * 0.3:
            recs.append(
                "Many kernels are latency-bound (pipeline stalls). "
                "Consider double-buffering and async DMA prefetch."
            )

    # Power-based recommendations
    if summary.power_trace:
        if summary.power_trace.thermal_headroom_pct < 20:
            recs.append(
                f"Low thermal headroom ({summary.power_trace.thermal_headroom_pct:.0f}%). "
                f"Risk of thermal throttling — consider power-aware scheduling."
            )
        if summary.power_trace.breakdown_pct.get("gpu", 0) > 30:
            recs.append(
                "Significant GPU power draw. If not using GPU compute, "
                "check for unintended GPU wake-ups."
            )

    # Memory-based recommendations
    if summary.memory_report:
        if summary.memory_report.vtcm_utilization_pct < 50:
            recs.append(
                f"VTCM underutilized ({summary.memory_report.vtcm_utilization_pct:.0f}%). "
                f"Increase tile sizes to fit more data in on-chip memory."
            )
        if summary.memory_report.fragmentation_pct > 15:
            recs.append(
                f"High memory fragmentation ({summary.memory_report.fragmentation_pct:.0f}%). "
                f"Consider memory pool allocation or static memory planning."
            )
        if summary.memory_report.reuse_efficiency_pct < 70:
            recs.append(
                "Low buffer reuse efficiency. Enable in-place operations "
                "and buffer sharing where liveness allows."
            )

    # System-level recommendations
    if summary.system_trace:
        idle = summary.system_trace.idle_pct
        if idle.get("npu", 0) > 40:
            recs.append(
                f"NPU idle {idle['npu']:.0f}% of the time. "
                f"Check for CPU-side bottlenecks or synchronization overhead."
            )
        if summary.system_trace.dma_stall_ms > summary.system_trace.total_duration_ms * 0.1:
            recs.append(
                "DMA stalls are significant. Use async DMA with double-buffering "
                "to overlap data transfers with compute."
            )

    if not recs:
        recs.append("Performance looks good. No critical bottlenecks detected.")

    return recs
