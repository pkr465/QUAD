"""orchestrate_workload tool — Allocate layers across CPU/GPU/NPU.

Resolution flow:

  1. Profile the model in `detailed` mode (necessary — `linting` and
     `qhas` profiles return cycle-level data without ms-per-layer
     timings, so they can't drive a heuristic allocation).
  2. If the profile comes back with no layers (e.g. caller forced a
     non-detailed profiling level upstream), automatically re-profile
     in `detailed` mode rather than producing a degenerate empty
     allocation.
  3. Allocate each layer to NPU / GPU / CPU using a heuristic that
     considers the layer's NPU compatibility, latency, and the chosen
     power mode.
  4. Project end-to-end latency / power / utilisation from the per-
     layer assignment and the original mean latency.

If even after re-profiling the layer list is still empty, raise
``InvalidProfileError`` so the caller gets a clear signal — better
than returning 0% utilisation across the board.
"""

from __future__ import annotations

from typing import Any, Literal

from quad.adapters.factory import AdapterFactory
from quad.exceptions import InvalidProfileError
from quad.models.orchestration import AllocationMap
from quad.models.profiling import ProfileRequest, ProfilingReport


async def _profile_with_layers(
    adapter: Any,
    model_path: str,
) -> ProfilingReport:
    """Profile the model and ensure the result has per-layer timings.

    First tries the adapter's default profiling level. If that yields no
    layers (the caller likely upstream-defaulted to `linting`), re-runs
    in `detailed` mode automatically so allocation has real data.
    """
    request = ProfileRequest(model_path=model_path, runtime="auto")
    report = await adapter.profile(request)
    if report.layers:
        return report

    # Fall back to detailed profiling — the only level that emits
    # per-layer ms timings.
    detailed_request = ProfileRequest(
        model_path=model_path,
        runtime="auto",
        profiling_level="detailed",
    )
    detailed = await adapter.profile(detailed_request)
    return detailed


async def orchestrate_workload_impl(
    model_path: str,
    power_mode: Literal["performance", "balanced", "efficiency"],
    factory: AdapterFactory,
) -> dict[str, Any]:
    """Orchestrate workload allocation and return AllocationMap as dict.

    First profiles the model, then allocates layers based on power mode.

    Args:
        model_path: Path to the model to orchestrate.
        power_mode: 'performance' (max NPU), 'balanced' (NPU for heavy
            layers + CPU for light), 'efficiency' (CPU first, NPU only
            for the costliest ops).
        factory: AdapterFactory that yields the SDK adapter.

    Raises:
        InvalidProfileError: if the profile has no per-layer data even
            after re-profiling in detailed mode. This usually means the
            model is genuinely opaque to the profiler (e.g. encrypted
            DLC) — orchestration cannot proceed.
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

    # Get supported ops for NPU
    supported_ops = await adapter.get_supported_ops()

    # Allocation algorithm
    allocation: dict[str, Literal["cpu", "gpu", "npu"]] = {}
    fallback_layers: list[str] = []

    # Compute the threshold ONCE outside the loop (was recomputed
    # per-layer before, which was slightly wrong for very small layers
    # with non-uniform timings).
    layer_count = max(len(report.layers), 1)
    avg_latency = report.latency.mean_ms / layer_count

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
            # Only high-compute layers go to NPU; everything else stays on CPU
            if layer.latency_ms > avg_latency:
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
    latency_factor = 1.0 - (npu_pct / 100 * 0.7)  # NPU is ~3.3x faster on heavy ops
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
    payload = result.model_dump()

    # Enrich with UI summary + power-mode tips
    try:
        from quad.tips import get_tips_for
        from quad.ui import format_allocation

        payload["ui"] = format_allocation(payload)
        payload["tips"] = [t.text for t in get_tips_for("orchestrate", n=2)]
    except Exception:
        pass

    return payload
