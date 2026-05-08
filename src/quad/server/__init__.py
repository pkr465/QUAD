"""Backward-compat shim — ``quad.server`` now forwards to :mod:`quad.mcp.server`.

The MCP server logic was moved to :mod:`quad.mcp.server` to make the
client-server-core layering explicit. This module preserves all the
old import paths so existing code keeps working — including the
FastMCP-decorated tool functions imported by name in tests.

Prefer importing from ``quad.mcp.server`` in new code.
"""

from quad.mcp.server import (
    cli,
    convert_model,
    generate_code,
    hardware_detect,
    mcp,
    orchestrate_workload,
    profile_workload,
)

__all__ = [
    "cli",
    "convert_model",
    "generate_code",
    "hardware_detect",
    "mcp",
    "orchestrate_workload",
    "profile_workload",
]
