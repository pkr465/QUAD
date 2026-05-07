"""System-level profiler capturing unified CPU+GPU+NPU+DMA timeline."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TraceEvent:
    """A single trace event on the system timeline."""

    timestamp_us: float
    duration_us: float
    device: str  # "cpu", "gpu", "npu", "dma"
    name: str
    category: str  # "compute", "transfer", "sync", "idle"


@dataclass
class SystemTrace:
    """Unified system trace result."""

    events: list[TraceEvent] = field(default_factory=list)
    total_duration_ms: float = 0.0
    idle_pct: dict[str, float] = field(default_factory=dict)
    dma_stall_ms: float = 0.0

    @property
    def event_count(self) -> int:
        return len(self.events)

    def device_events(self, device: str) -> list[TraceEvent]:
        """Filter events by device."""
        return [e for e in self.events if e.device == device]

    def timeline_summary(self) -> dict[str, float]:
        """Return total time per device in ms."""
        summary: dict[str, float] = {}
        for event in self.events:
            summary[event.device] = summary.get(event.device, 0.0) + event.duration_us / 1000.0
        return summary


class SystemProfiler:
    """Captures a unified timeline of CPU+GPU+NPU+DMA activity.

    Supports mock mode for generating realistic simulated data without
    requiring actual Qualcomm hardware.
    """

    def __init__(self, mock: bool = True):
        self._mock = mock
        self._running = False
        self._start_time: Optional[float] = None
        self._trace: Optional[SystemTrace] = None

    def start(self) -> None:
        """Begin system-level profiling."""
        self._running = True
        self._start_time = time.perf_counter()
        self._trace = None

    def stop(self) -> SystemTrace:
        """Stop profiling and return the system trace."""
        if not self._running:
            raise RuntimeError("Profiler is not running. Call start() first.")

        elapsed_s = time.perf_counter() - self._start_time
        self._running = False

        if self._mock:
            self._trace = self._generate_mock_trace(elapsed_s)
        else:
            # Real hardware path — placeholder
            self._trace = SystemTrace(total_duration_ms=elapsed_s * 1000.0)

        return self._trace

    def __enter__(self) -> "SystemProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._running:
            self.stop()

    @property
    def trace(self) -> Optional[SystemTrace]:
        """Access the last captured trace."""
        return self._trace

    # ------------------------------------------------------------------
    # Mock data generation
    # ------------------------------------------------------------------

    def _generate_mock_trace(self, elapsed_s: float) -> SystemTrace:
        """Generate a realistic mock system trace."""
        total_duration_ms = max(elapsed_s * 1000.0, 25.0)
        total_duration_us = total_duration_ms * 1000.0

        devices = ["cpu", "gpu", "npu", "dma"]
        categories_map = {
            "cpu": ["compute", "sync", "idle"],
            "gpu": ["compute", "transfer", "idle"],
            "npu": ["compute", "transfer", "idle"],
            "dma": ["transfer", "sync", "idle"],
        }

        op_names = {
            "cpu": ["preprocess", "postprocess", "quantize", "dequantize", "schedule"],
            "gpu": ["conv2d", "relu", "batch_norm", "upsample", "attention"],
            "npu": ["conv2d_hexNN", "matmul_hvx", "depthwise_hexNN", "softmax_hvx", "layernorm_hvx"],
            "dma": ["ddr_to_vtcm", "vtcm_to_ddr", "host_to_device", "device_to_host"],
        }

        events: list[TraceEvent] = []
        device_busy_us: dict[str, float] = {d: 0.0 for d in devices}

        # Generate events across the timeline
        num_events = random.randint(40, 80)
        for _ in range(num_events):
            device = random.choice(devices)
            category = random.choices(
                categories_map[device],
                weights=[0.6, 0.25, 0.15] if len(categories_map[device]) == 3 else [0.7, 0.3],
            )[0]
            name = random.choice(op_names[device])

            timestamp_us = random.uniform(0, total_duration_us * 0.9)

            if category == "idle":
                duration_us = random.uniform(50, 500)
            elif device == "npu":
                duration_us = random.uniform(100, 5000)
            elif device == "dma":
                duration_us = random.uniform(20, 800)
            else:
                duration_us = random.uniform(50, 2000)

            events.append(TraceEvent(
                timestamp_us=round(timestamp_us, 2),
                duration_us=round(duration_us, 2),
                device=device,
                name=name,
                category=category,
            ))

            if category != "idle":
                device_busy_us[device] += duration_us

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp_us)

        # Compute idle percentages
        idle_pct: dict[str, float] = {}
        for device in devices:
            busy_ratio = min(device_busy_us[device] / total_duration_us, 0.95)
            idle_pct[device] = round((1.0 - busy_ratio) * 100.0, 1)

        # DMA stall — realistic fraction
        dma_stall_ms = round(random.uniform(0.5, total_duration_ms * 0.08), 3)

        return SystemTrace(
            events=events,
            total_duration_ms=round(total_duration_ms, 3),
            idle_pct=idle_pct,
            dma_stall_ms=dma_stall_ms,
        )
