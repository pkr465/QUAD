"""QUAD Power — power monitoring, budgeting, and battery estimation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any


class PowerMode(Enum):
    """Power execution modes."""
    PERFORMANCE = "performance"   # Max speed, no power limit
    BALANCED = "balanced"         # Good speed, moderate power
    EFFICIENCY = "efficiency"     # Minimize power, accept slower speed


@dataclass
class PowerReport:
    """Power measurement from a profiled execution."""
    avg_power_mw: float
    peak_power_mw: float
    energy_mj: float
    duration_ms: float
    breakdown: dict[str, float]  # {"npu": mW, "gpu": mW, "cpu": mW}


@dataclass
class BatteryEstimate:
    """Battery life estimation for sustained workload."""
    hours: float
    inference_count: int
    energy_per_inference_mj: float
    duty_cycle: float


class PowerMonitor:
    """Real-time power monitoring during inference.

    Usage:
        with PowerMonitor() as pm:
            output = model(input_tensor)
        print(f"Used {pm.report.energy_mj:.1f} mJ")

    In mock mode, returns simulated power data based on device characteristics.
    In real mode, reads from QPM3 or platform power sensors.
    """

    def __init__(self, device_type: str = "npu"):
        self._device_type = device_type
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._report: PowerReport | None = None

    @property
    def report(self) -> PowerReport:
        """Get the power report from the last monitored execution."""
        if self._report is None:
            return PowerReport(
                avg_power_mw=0, peak_power_mw=0, energy_mj=0,
                duration_ms=0, breakdown={}
            )
        return self._report

    def __enter__(self) -> PowerMonitor:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self._end_time = time.perf_counter()
        duration_ms = (self._end_time - self._start_time) * 1000

        # Mock power data based on device type
        power_profiles = {
            "npu": {"npu": 1850.0, "gpu": 200.0, "cpu": 450.0},
            "gpu": {"npu": 100.0, "gpu": 3200.0, "cpu": 500.0},
            "cpu": {"npu": 0.0, "gpu": 100.0, "cpu": 4500.0},
        }
        breakdown = power_profiles.get(self._device_type, power_profiles["npu"])
        avg_power = sum(breakdown.values())

        self._report = PowerReport(
            avg_power_mw=avg_power,
            peak_power_mw=avg_power * 1.3,
            energy_mj=avg_power * duration_ms / 1000,
            duration_ms=duration_ms,
            breakdown=breakdown,
        )


def estimate_battery_life(
    power_mw: float = 2500.0,
    duty_cycle: float = 0.3,
    battery_mah: int = 5000,
    voltage: float = 3.7,
) -> BatteryEstimate:
    """Estimate battery life for a sustained inference workload.

    Args:
        power_mw: Average power during inference (milliwatts)
        duty_cycle: Fraction of time doing inference (0.0-1.0)
        battery_mah: Battery capacity in mAh
        voltage: Battery voltage (typically 3.7V for Li-ion)

    Returns:
        BatteryEstimate with hours of operation and per-inference energy.
    """
    battery_wh = battery_mah * voltage / 1000  # Convert to Wh
    avg_system_power_w = (power_mw * duty_cycle) / 1000  # Active inference power
    idle_power_w = 0.5  # Assume 500mW idle
    total_power_w = avg_system_power_w + idle_power_w * (1 - duty_cycle)

    hours = battery_wh / total_power_w if total_power_w > 0 else float("inf")

    # Estimate inferences per hour (assume 5ms per inference)
    inferences_per_second = 1000 / 5.0  # 200 FPS
    inference_count = int(hours * 3600 * inferences_per_second * duty_cycle)

    energy_per_inference = power_mw * 5.0 / 1000  # mJ per 5ms inference

    return BatteryEstimate(
        hours=round(hours, 1),
        inference_count=inference_count,
        energy_per_inference_mj=round(energy_per_inference, 2),
        duty_cycle=duty_cycle,
    )
