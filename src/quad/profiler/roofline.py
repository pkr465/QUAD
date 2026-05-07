"""Roofline model analysis for Qualcomm hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quad.profiler.kernel import KernelMetrics


@dataclass
class RooflineResult:
    """Result of a roofline analysis."""

    peak_tops: float
    peak_bandwidth_gb_s: float
    ridge_point: float  # arithmetic intensity where compute=memory bound
    achieved_tops: float
    achieved_pct: float  # percentage of peak achieved
    diagnosis: Literal["compute-bound", "memory-bound"]
    recommendation: str

    def __repr__(self) -> str:
        return (
            f"RooflineResult(achieved={self.achieved_pct:.1f}% of peak, "
            f"diagnosis={self.diagnosis!r})"
        )


class RooflineAnalysis:
    """Roofline model for analyzing compute vs. memory boundedness.

    The roofline model characterizes kernel performance relative to the
    theoretical hardware limits (peak compute and peak bandwidth), helping
    identify whether workloads are compute-bound or memory-bound.

    Args:
        device_peak_tops: Peak compute throughput in TOPS.
        device_bandwidth_gb_s: Peak memory bandwidth in GB/s.
    """

    def __init__(self, device_peak_tops: float = 73.0, device_bandwidth_gb_s: float = 68.0):
        self.device_peak_tops = device_peak_tops
        self.device_bandwidth_gb_s = device_bandwidth_gb_s
        # Ridge point: ops/byte where compute ceiling meets bandwidth ceiling
        # peak_tops (in Giga-ops/s) / bandwidth (GB/s) = ops/byte
        self._ridge_point = (device_peak_tops * 1000.0) / device_bandwidth_gb_s

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity at the ridge point (ops/byte)."""
        return self._ridge_point

    def analyze(self, kernels: list[KernelMetrics]) -> RooflineResult:
        """Perform roofline analysis on a set of kernel metrics.

        Args:
            kernels: List of KernelMetrics from the kernel profiler.

        Returns:
            RooflineResult with diagnosis and recommendations.
        """
        if not kernels:
            return RooflineResult(
                peak_tops=self.device_peak_tops,
                peak_bandwidth_gb_s=self.device_bandwidth_gb_s,
                ridge_point=self._ridge_point,
                achieved_tops=0.0,
                achieved_pct=0.0,
                diagnosis="memory-bound",
                recommendation="No kernels to analyze.",
            )

        # Compute achieved throughput from kernel metrics
        # Use weighted average by latency
        total_latency_us = sum(k.latency_us for k in kernels)
        if total_latency_us == 0:
            total_latency_us = 1.0

        weighted_util = sum(
            k.compute_utilization_pct * k.latency_us for k in kernels
        ) / total_latency_us

        achieved_tops = self.device_peak_tops * (weighted_util / 100.0)
        achieved_pct = round(weighted_util, 1)

        # Weighted average arithmetic intensity
        weighted_ai = sum(
            k.arithmetic_intensity * k.latency_us for k in kernels
        ) / total_latency_us

        # Diagnosis based on where we fall relative to ridge point
        if weighted_ai < self._ridge_point:
            diagnosis: Literal["compute-bound", "memory-bound"] = "memory-bound"
        else:
            diagnosis = "compute-bound"

        # Generate actionable recommendation
        recommendation = self._generate_recommendation(
            diagnosis, achieved_pct, weighted_ai, kernels
        )

        return RooflineResult(
            peak_tops=self.device_peak_tops,
            peak_bandwidth_gb_s=self.device_bandwidth_gb_s,
            ridge_point=round(self._ridge_point, 2),
            achieved_tops=round(achieved_tops, 2),
            achieved_pct=achieved_pct,
            diagnosis=diagnosis,
            recommendation=recommendation,
        )

    def _generate_recommendation(
        self,
        diagnosis: str,
        achieved_pct: float,
        avg_arithmetic_intensity: float,
        kernels: list[KernelMetrics],
    ) -> str:
        """Generate actionable optimization recommendations."""
        recommendations: list[str] = []

        if diagnosis == "memory-bound":
            recommendations.append(
                "Workload is memory-bound. Consider: operator fusion to reduce "
                "memory traffic, increasing tile sizes for better VTCM utilization, "
                "or quantizing to INT8/INT4 to reduce bandwidth pressure."
            )
            # Check for specific memory-bound ops
            mem_bound_ops = [k for k in kernels if k.bottleneck == "memory"]
            if mem_bound_ops:
                worst = max(mem_bound_ops, key=lambda k: k.latency_us)
                recommendations.append(
                    f"Top memory-bound kernel: '{worst.name}' "
                    f"(AI={worst.arithmetic_intensity:.1f} ops/byte). "
                    f"Target this op for fusion or tiling optimization."
                )
        else:
            recommendations.append(
                "Workload is compute-bound. Consider: leveraging INT8/INT4 quantization "
                "for higher throughput, parallelizing across HVX+HTP cores, or "
                "decomposing large ops into HTP-friendly tile shapes."
            )

        if achieved_pct < 50.0:
            recommendations.append(
                f"Only achieving {achieved_pct:.0f}% of peak. Significant room for "
                f"optimization — check for pipeline stalls and suboptimal scheduling."
            )

        return " ".join(recommendations)
