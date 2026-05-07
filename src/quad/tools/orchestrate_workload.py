"""orchestrate_workload tool — Allocate layers across CPU/GPU/NPU."""

from __future__ import annotations

from typing import Any, Literal

from quad.adapters.factory import AdapterFactory
from quad.models.orchestration import AllocationMap
from quad.models.profiling import ProfileRequest


async def orchestrate_workload_impl(
    model_path: str,
    power_mode: Literal["performance", "balanced", "efficiency"],
    factory: AdapterFactory,
) -> dict[str, Any]:
    """Orchestrate workload allocation and return AllocationMap as dict.

    First profiles the model, then allocates layers based on power mode.
    """
    adapter = factory.get_adapter("auto")

    # Profile first to get layer data
    request = ProfileRequest(model_path=model_path, runtime="auto")
    report = await adapter.profile(request)

    # Get supported ops for NPU
    supported_ops = await adapter.get_supported_ops()

    # Allocation algorithm
    allocation: dict[str, Literal["cpu", "gpu", "npu"]] = {}
    fallback_layers: list[str] = []

    for layer in report.layers:
        op_base = layer.op_type.lower()
        # Check NPU compatibility
        npu_compatible = any(op_base in supported.lower() for supported in supported_ops)

        if not npu_compatible:
            allocation[layer.name] = "cpu"
            fallback_layers.append(layer.name)
        elif power_mode == "performance":
            allocation[layer.name] = "npu"
        elif power_mode == "efficiency":
            # Only high-compute layers go to NPU
            if layer.latency_ms > report.latency.mean_ms / len(report.layers):
                allocation[layer.name] = "npu"
            else:
                allocation[layer.name] = "cpu"
        else:  # balanced
            allocation[layer.name] = "npu"

    # Calculate utilization percentages
    total = len(allocation) or 1
    npu_count = sum(1 for v in allocation.values() if v == "npu")
    gpu_count = sum(1 for v in allocation.values() if v == "gpu")
    cpu_count = sum(1 for v in allocation.values() if v == "cpu")

    # Project metrics based on allocation
    npu_pct = (npu_count / total) * 100
    latency_factor = 1.0 - (npu_pct / 100 * 0.7)  # NPU is faster
    projected_latency = report.latency.mean_ms * latency_factor
    projected_power = report.power_mw * (0.5 + (npu_pct / 100 * 0.3))

    result = AllocationMap(
        allocation=allocation,
        projected_latency_ms=round(projected_latency, 2),
        projected_power_mw=round(projected_power, 1),
        projected_memory_mb=report.memory_peak_mb,
        power_mode=power_mode,
        fallback_layers=fallback_layers,
        npu_utilization_pct=round(npu_pct, 1),
        gpu_utilization_pct=round((gpu_count / total) * 100, 1),
        cpu_utilization_pct=round((cpu_count / total) * 100, 1),
    )
    return result.model_dump()
