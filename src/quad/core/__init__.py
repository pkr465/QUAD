"""QUAD core — pure business logic, no MCP, no client integration.

Anything in this package can be safely imported from any caller (CLI,
MCP server, future Python SDK, tests). Operations here return clean
dataclass-shaped dicts — no UI markdown, no contextual tips, no
suggestion lists. Those are added at higher layers (``quad.mcp.``,
``quad.cli.``).

Public API:
    from quad.core.operations import (
        hardware_detect,
        convert_model,
        profile_workload,
        orchestrate_workload,
        generate_code,
    )
"""

from quad.core import operations

__all__ = ["operations"]
