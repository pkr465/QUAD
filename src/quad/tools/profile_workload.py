"""profile_workload tool — Run profiler and return structured metrics."""

from __future__ import annotations

from typing import Any, Literal

from quad.adapters.factory import AdapterFactory
from quad.models.profiling import ProfileRequest


async def profile_workload_impl(
    model_path: str,
    platform: Literal["windows", "linux", "android"],
    runtime: Literal["cpu", "gpu", "npu", "auto"],
    duration_s: int,
    factory: AdapterFactory,
    profiling_level: Literal["basic", "detailed", "linting", "qhas"] = "detailed",
    htp_soc: str = "sm8750",
    sdk_root: str | None = None,
) -> dict[str, Any]:
    """Profile workload and return ProfilingReport as dict.

    For linting and qhas profiling levels the report includes cycle-based
    per-op metrics (linting_layers, linting_bottleneck_count, etc.) in addition
    to the standard latency/power/memory fields.
    """
    adapter = factory.get_adapter("auto")
    request = ProfileRequest(
        model_path=model_path,
        platform=platform,
        runtime=runtime,
        duration_s=duration_s,
        profiling_level=profiling_level,
        htp_soc=htp_soc,
        sdk_root=sdk_root,
    )
    report = await adapter.profile(request)
    return report.model_dump()
