"""Memory profiler for tracking allocations, bandwidth, and fragmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

try:
    from quad.compiler.ir import IRGraph, IRNode
except ImportError:
    IRGraph = None
    IRNode = None


@dataclass
class AllocationEvent:
    """A single memory allocation event."""

    timestamp_ms: float
    size_mb: float
    device: str  # "vtcm", "ddr", "system"
    name: str


@dataclass
class MemoryReport:
    """Complete memory profiling report."""

    peak_mb: float = 0.0
    avg_mb: float = 0.0
    vtcm_utilization_pct: float = 0.0
    ddr_bandwidth_gb_s: float = 0.0
    allocations: list[AllocationEvent] = field(default_factory=list)
    fragmentation_pct: float = 0.0
    reuse_efficiency_pct: float = 0.0

    @property
    def allocation_count(self) -> int:
        return len(self.allocations)

    def vtcm_allocations(self) -> list[AllocationEvent]:
        """Return only VTCM allocations."""
        return [a for a in self.allocations if a.device == "vtcm"]

    def ddr_allocations(self) -> list[AllocationEvent]:
        """Return only DDR allocations."""
        return [a for a in self.allocations if a.device == "ddr"]


class MemoryProfiler:
    """Profiles memory usage patterns for Qualcomm hardware.

    Tracks VTCM (on-chip tightly coupled memory), DDR, and system memory
    allocations. Provides fragmentation analysis and reuse efficiency
    metrics critical for mobile deployment.
    """

    def __init__(self, mock: bool = True, device: str = "npu"):
        self._mock = mock
        self._device = device
        # Qualcomm Hexagon VTCM: typically 8MB
        self._vtcm_capacity_mb = 8.0
        # DDR available for accelerator: varies, assume 4GB
        self._ddr_capacity_mb = 4096.0

    def profile(self, ir_graph=None) -> MemoryReport:
        """Profile memory usage for the given IR graph.

        Args:
            ir_graph: An IRGraph instance. If None or in mock mode,
                      generates realistic simulated data.

        Returns:
            MemoryReport with allocation details and metrics.
        """
        if self._mock or ir_graph is None:
            return self._generate_mock_report(ir_graph)

        # Real profiling path — placeholder
        return MemoryReport()

    def _generate_mock_report(self, ir_graph=None) -> MemoryReport:
        """Generate realistic mock memory profiling data."""
        # Typical model memory patterns
        num_allocations = random.randint(25, 60)

        allocations: list[AllocationEvent] = []
        running_total_mb = 0.0
        peak_mb = 0.0
        total_mb_sum = 0.0

        vtcm_used_mb = 0.0
        total_duration_ms = random.uniform(15.0, 50.0)

        tensor_names = [
            "input_tensor", "conv1_weight", "conv1_output", "bn1_running_mean",
            "conv2_weight", "conv2_output", "attention_qkv", "attention_scores",
            "attention_output", "ffn_weight1", "ffn_intermediate", "ffn_weight2",
            "ffn_output", "residual_buffer", "layer_norm_gamma", "layer_norm_beta",
            "pooled_output", "logits", "softmax_output", "final_output",
        ]

        for i in range(num_allocations):
            timestamp_ms = (i / num_allocations) * total_duration_ms

            # Mix of VTCM and DDR allocations
            if random.random() < 0.4:
                device = "vtcm"
                size_mb = round(random.uniform(0.01, 2.0), 3)
                vtcm_used_mb = min(vtcm_used_mb + size_mb, self._vtcm_capacity_mb)
            else:
                device = "ddr"
                size_mb = round(random.uniform(0.1, 32.0), 3)

            name = random.choice(tensor_names) + f"_{i}"

            # Simulate some frees (negative doesn't appear in events, just in running total)
            if i > 5 and random.random() < 0.3:
                running_total_mb = max(0, running_total_mb - random.uniform(0.5, 10.0))

            running_total_mb += size_mb
            peak_mb = max(peak_mb, running_total_mb)
            total_mb_sum += running_total_mb

            allocations.append(AllocationEvent(
                timestamp_ms=round(timestamp_ms, 2),
                size_mb=size_mb,
                device=device,
                name=name,
            ))

        avg_mb = total_mb_sum / num_allocations if num_allocations > 0 else 0.0

        # VTCM utilization: how much of VTCM capacity is being used
        vtcm_utilization_pct = min((vtcm_used_mb / self._vtcm_capacity_mb) * 100.0, 100.0)

        # DDR bandwidth — realistic for Snapdragon
        ddr_bandwidth_gb_s = round(random.uniform(20.0, 55.0), 1)

        # Fragmentation: how much memory is wasted due to fragmentation
        fragmentation_pct = round(random.uniform(3.0, 18.0), 1)

        # Reuse efficiency: how well are buffers being reused
        reuse_efficiency_pct = round(random.uniform(60.0, 92.0), 1)

        return MemoryReport(
            peak_mb=round(peak_mb, 2),
            avg_mb=round(avg_mb, 2),
            vtcm_utilization_pct=round(vtcm_utilization_pct, 1),
            ddr_bandwidth_gb_s=ddr_bandwidth_gb_s,
            allocations=allocations,
            fragmentation_pct=fragmentation_pct,
            reuse_efficiency_pct=reuse_efficiency_pct,
        )
