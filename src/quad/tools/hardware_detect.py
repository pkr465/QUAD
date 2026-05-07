"""hardware_detect tool — Detect Qualcomm chipset and compute units."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from quad.adapters.factory import AdapterFactory
from quad.models.config import ServerConfig


async def hardware_detect_impl(
    platform: Literal["windows", "linux", "android"],
    factory: AdapterFactory,
) -> dict[str, Any]:
    """Detect hardware and return DeviceProfile as dict."""
    adapter = factory.get_adapter("auto")
    profile = await adapter.detect_hardware(platform)
    return profile.model_dump()
