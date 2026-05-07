"""Power profiler — a key QUAD differentiator for mobile/edge deployment."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PowerSample:
    """A single power measurement sample."""

    timestamp_ms: float
    npu_mw: float
    gpu_mw: float
    cpu_mw: float
    total_mw: float


@dataclass
class BatteryImpact:
    """Battery life impact estimation."""

    hours_at_workload: float
    drain_pct_per_hour: float
    battery_mah: float
    voltage: float

    def __repr__(self) -> str:
        return (
            f"BatteryImpact(hours={self.hours_at_workload:.1f}h, "
            f"drain={self.drain_pct_per_hour:.1f}%/hr)"
        )


@dataclass
class PowerTrace:
    """Complete power trace from a profiling session."""

    samples: list[PowerSample] = field(default_factory=list)
    avg_power_mw: float = 0.0
    peak_power_mw: float = 0.0
    energy_mj: float = 0.0
    breakdown_pct: dict[str, float] = field(default_factory=dict)
    thermal_headroom_pct: float = 0.0

    def battery_impact(self, battery_mah: float = 5000.0, voltage: float = 3.85) -> BatteryImpact:
        """Estimate battery life impact at the measured power draw.

        Args:
            battery_mah: Battery capacity in mAh.
            voltage: Nominal battery voltage.

        Returns:
            BatteryImpact with estimated runtime.
        """
        battery_mwh = battery_mah * voltage
        if self.avg_power_mw <= 0:
            return BatteryImpact(
                hours_at_workload=float("inf"),
                drain_pct_per_hour=0.0,
                battery_mah=battery_mah,
                voltage=voltage,
            )

        hours = battery_mwh / self.avg_power_mw
        drain_pct_per_hour = 100.0 / hours if hours > 0 else 100.0

        return BatteryImpact(
            hours_at_workload=round(hours, 2),
            drain_pct_per_hour=round(drain_pct_per_hour, 2),
            battery_mah=battery_mah,
            voltage=voltage,
        )


class PowerProfiler:
    """Power profiler for measuring per-subsystem power consumption.

    This is a key QUAD differentiator — mobile/edge deployments need
    power-aware optimization that desktop-focused tools like Nsight lack.

    Supports context manager usage for scoped power capture.
    """

    def __init__(self, mock: bool = True, sample_interval_ms: float = 1.0):
        self._mock = mock
        self._sample_interval_ms = sample_interval_ms
        self._running = False
        self._start_time: Optional[float] = None
        self._trace: Optional[PowerTrace] = None

    def start(self) -> None:
        """Begin power capture."""
        self._running = True
        self._start_time = time.perf_counter()
        self._trace = None

    def stop(self) -> PowerTrace:
        """Stop power capture and return the trace."""
        if not self._running:
            raise RuntimeError("PowerProfiler is not running. Call start() first.")

        elapsed_s = time.perf_counter() - self._start_time
        self._running = False

        if self._mock:
            self._trace = self._generate_mock_trace(elapsed_s)
        else:
            self._trace = PowerTrace()

        return self._trace

    def __enter__(self) -> "PowerProfiler":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._running:
            self.stop()

    @property
    def trace(self) -> Optional[PowerTrace]:
        """Access the last captured power trace."""
        return self._trace

    def _generate_mock_trace(self, elapsed_s: float) -> PowerTrace:
        """Generate realistic mock power measurements."""
        duration_ms = max(elapsed_s * 1000.0, 20.0)
        num_samples = max(int(duration_ms / self._sample_interval_ms), 20)

        # Realistic power profiles for Snapdragon platform
        # NPU (Hexagon): 2-8W during active inference
        # GPU (Adreno): 1-5W during compute
        # CPU: 0.5-3W
        npu_base_mw = random.uniform(2000, 4000)
        gpu_base_mw = random.uniform(500, 1500)
        cpu_base_mw = random.uniform(400, 1000)

        samples: list[PowerSample] = []
        for i in range(num_samples):
            ts = (i / num_samples) * duration_ms
            # Add realistic variation (thermal noise + workload spikes)
            npu_mw = npu_base_mw + random.gauss(0, npu_base_mw * 0.1)
            gpu_mw = gpu_base_mw + random.gauss(0, gpu_base_mw * 0.15)
            cpu_mw = cpu_base_mw + random.gauss(0, cpu_base_mw * 0.08)

            # Occasional workload spikes
            if random.random() < 0.05:
                npu_mw *= 1.4

            npu_mw = max(npu_mw, 100.0)
            gpu_mw = max(gpu_mw, 50.0)
            cpu_mw = max(cpu_mw, 100.0)
            total_mw = npu_mw + gpu_mw + cpu_mw

            samples.append(PowerSample(
                timestamp_ms=round(ts, 2),
                npu_mw=round(npu_mw, 1),
                gpu_mw=round(gpu_mw, 1),
                cpu_mw=round(cpu_mw, 1),
                total_mw=round(total_mw, 1),
            ))

        # Compute aggregates
        total_powers = [s.total_mw for s in samples]
        avg_power = sum(total_powers) / len(total_powers)
        peak_power = max(total_powers)

        # Energy = avg power * duration
        energy_mj = avg_power * (duration_ms / 1000.0)  # mW * s = mJ

        # Breakdown
        avg_npu = sum(s.npu_mw for s in samples) / len(samples)
        avg_gpu = sum(s.gpu_mw for s in samples) / len(samples)
        avg_cpu = sum(s.cpu_mw for s in samples) / len(samples)
        total_avg = avg_npu + avg_gpu + avg_cpu

        breakdown_pct = {
            "npu": round((avg_npu / total_avg) * 100.0, 1),
            "gpu": round((avg_gpu / total_avg) * 100.0, 1),
            "cpu": round((avg_cpu / total_avg) * 100.0, 1),
        }

        # Thermal headroom: how far from thermal throttle point
        # Typical TDP for mobile SoC ~10-15W, throttle around 12W
        thermal_limit_mw = 12000.0
        thermal_headroom = max(0.0, (thermal_limit_mw - peak_power) / thermal_limit_mw * 100.0)

        return PowerTrace(
            samples=samples,
            avg_power_mw=round(avg_power, 1),
            peak_power_mw=round(peak_power, 1),
            energy_mj=round(energy_mj, 2),
            breakdown_pct=breakdown_pct,
            thermal_headroom_pct=round(thermal_headroom, 1),
        )
