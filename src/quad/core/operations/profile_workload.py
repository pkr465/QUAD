"""Pure profiling operation."""

from __future__ import annotations

from typing import Any, Literal

from quad.adapters.factory import AdapterFactory
from quad.models.profiling import ProfileRequest


async def profile_workload(
    model_path: str,
    platform: Literal["windows", "linux", "android"],
    runtime: Literal["cpu", "gpu", "npu", "auto"],
    duration_s: int,
    factory: AdapterFactory,
    *,
    profiling_level: Literal["basic", "detailed", "linting", "qhas"] = "detailed",
    htp_soc: str = "sm8750",
    sdk_root: str | None = None,
) -> dict[str, Any]:
    """Profile a model. Returns ProfilingReport dict (no MCP enrichment)."""
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
