"""QUAD MCP — Model Context Protocol server layer.

Wraps :mod:`quad.core.operations` with FastMCP tool registration plus
the response-enrichment layer (markdown ``ui`` summary + contextual
``tips`` + per-bottleneck ``suggestions``). Anything client-specific
(Claude Code skills, settings.json) lives in :mod:`quad.client`.

Public API:

    from quad.mcp.server import cli, mcp
    from quad.mcp.enrichment import enrich_hardware_detect, ...
    from quad.mcp.tools import hardware_detect_impl, ...
"""

from quad.mcp.server import cli, mcp

__all__ = ["cli", "mcp"]
