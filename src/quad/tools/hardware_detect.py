"""hardware_detect tool — Detect Qualcomm chipset and compute units."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from quad.adapters.factory import AdapterFactory
from quad.models.config import ServerConfig


async def hardware_detect_impl(
    platform: Literal["windows", "linux", "android"],
    factory: AdapterFactory,
    *,
    enrich: bool = True,
) -> dict[str, Any]:
    """Detect hardware and return DeviceProfile as dict.

    Args:
        platform: target platform
        factory: adapter factory
        enrich: if True (default), include `ui` (markdown summary) and
            `tips` (catalogue snippets) keys in the response. Disable
            for raw-data-only callers (e.g. some tests).
    """
    adapter = factory.get_adapter("auto")
    profile = await adapter.detect_hardware(platform)
    result = profile.model_dump()

    if enrich:
        try:
            from quad.tips import format_tips_markdown, get_tips_for
            from quad.ui import format_device

            result["ui"] = format_device(result)
            tips = get_tips_for("detect", n=2)
            result["tips"] = [t.text for t in tips]
        except Exception:
            # Enrichment is best-effort — never block the data path
            pass

    return result
