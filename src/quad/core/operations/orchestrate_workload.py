"""Pure orchestration operation.

Allocates inference graph nodes across CPU/GPU/NPU based on profile
data + power mode. Pure data-in / data-out — no MCP enrichment.

The algorithm lives entirely in this module so callers in
:mod:`quad.core` don't depend on the legacy ``quad.tools`` shim layer.
"""

from __future__ import annotations

from typing import Any, Literal

from quad.adapters.factory import AdapterFactory
from quad.exceptions import InvalidProfileError
from quad.models.orchestration import AllocationMap
from quad.models.profiling import ProfileRequest, ProfilingReport


async def _profile_with_layers(adapter: Any, model_path: str) -> ProfilingReport:
    """Profile + auto-fallback to detailed mode if no per-layer data."""
    request = ProfileRequest(model_path=model_path, runtime="auto")
    report = await adapter.profile(request)
    if report.layers:
        return report
    detailed_request = ProfileRequest(
        model_path=model_path,
        runtime="auto",
        profiling_level="detailed",
    )
    return await adapter.profile(detailed_request)


async def orchestrate_workload(
    model_path: str,
    power_mode: Literal["performance", "balanced", "efficiency"],
    factory: AdapterFactory,
) -> dict[str, Any]:
    """Allocate inference graph nodes across CPU/GPU/NPU.

    Returns AllocationMap dict (no MCP enrichment).

    Raises:
        InvalidProfileError: when the upstream profile has no per-layer
            data even after re-profiling in detailed mode.
    """
    adapter = factory.get_adapter("auto")
    report = await _profile_with_layers(adapter, model_path)

    if not report.layers:
        raise InvalidProfileError(
            "profile returned no per-layer data even after re-profiling in 'detailed' mode. "
            "Cannot allocate without per-layer timings — check that the model is profilable "
            "(non-encrypted DLC, supported runtime). For real hardware mode, ensure "
            "snpe-net-run is on PATH and the model file is reachable from the test machine."
        )

    supported_ops = await adapter.get_supported_ops()
    allocation: dict[str, Literal["cpu", "gpu", "npu"]] = {}
    fallback_layers: list[str] = []
    layer_count = max(len(report.layers), 1)
    avg_latency = report.latency.mean_ms / layer_count

    for layer in report.layers:
        op_base = layer.op_type.lower()
        npu_compatible = any(op_base in supported.lower() for supported in supported_ops)
        if not npu_compatible:
            allocation[layer.name] = "cpu"
            fallback_layers.append(layer.name)
        elif power_mode == "performance":
            allocation[layer.name] = "npu"
        elif power_mode == "efficiency":
            allocation[layer.name] = "npu" if layer.latency_ms > avg_latency else "cpu"
        else:  # balanced
            allocation[layer.name] = "npu"

    total = len(allocation) or 1
    npu_count = sum(1 for v in allocation.values() if v == "npu")
    gpu_count = sum(1 for v in allocation.values() if v == "gpu")
    cpu_count = sum(1 for v in allocation.values() if v == "cpu")

    npu_pct = (npu_count / total) * 100
    latency_factor = 1.0 - (npu_pct / 100 * 0.7)
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
