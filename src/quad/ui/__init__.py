"""QUAD UI helpers — rich markdown formatters for MCP tool responses.

When QUAD is invoked through Claude Code (the typical flow), the
MCP tool responses are markdown-rendered in the chat. These helpers
turn raw dataclass / dict outputs into readable summaries with
tables, percentile breakdowns, utilisation bars, and contextual
callouts.

All formatters return a markdown string. They are pure functions of
their input — no side effects. Each formatter is independently
testable and composable (e.g. ``format_device + format_profile``
becomes the rendering inside the ``hardware_detect`` skill).

Public API:

    from quad.ui import (
        format_device,
        format_profile,
        format_conversion,
        format_allocation,
        format_doctor,
        format_coverage,
    )
"""

from quad.ui.formatters import (
    format_allocation,
    format_conversion,
    format_coverage,
    format_device,
    format_doctor,
    format_profile,
    format_sdk_status,
    format_table,
    format_utilization_bar,
)

__all__ = [
    "format_allocation",
    "format_conversion",
    "format_coverage",
    "format_device",
    "format_doctor",
    "format_profile",
    "format_sdk_status",
    "format_table",
    "format_utilization_bar",
]
