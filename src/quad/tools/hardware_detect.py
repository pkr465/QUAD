"""Backward-compat shim — re-exports from :mod:`quad.mcp.tools`.

The MCP tool wrappers moved to :mod:`quad.mcp.tools` so the layering
between pure logic (``quad.core.operations``) and MCP enrichment is
explicit. Prefer importing from there in new code; this module is a
compatibility re-export.
"""

from quad.mcp.tools import hardware_detect_impl

__all__ = ["hardware_detect_impl"]
