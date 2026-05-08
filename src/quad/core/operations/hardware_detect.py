"""Pure hardware-detection operation."""

from __future__ import annotations

from typing import Any, Literal

from quad.adapters.factory import AdapterFactory


async def hardware_detect(
    platform: Literal["windows", "linux", "android"],
    factory: AdapterFactory,
) -> dict[str, Any]:
    """Detect Qualcomm chipset and compute units.

    Returns a clean DeviceProfile dict — no MCP enrichment. Use
    ``quad.mcp.tools.hardware_detect_impl`` if you want the MCP
    version with ``ui`` / ``tips`` keys.
    """
    adapter = factory.get_adapter("auto")
    profile = await adapter.detect_hardware(platform)
    return profile.model_dump()
