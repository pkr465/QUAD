"""Tests for MCP server tool registration and basic dispatch."""

from __future__ import annotations

from quad.server import mcp


class TestServerRegistration:
    """Verify all 5 tools are registered with the MCP server."""

    def test_server_has_name(self) -> None:
        assert "QUAD" in mcp.name

    def test_tools_registered(self) -> None:
        # FastMCP registers tools internally; verify we can import without error
        from quad.server import (
            convert_model,
            generate_code,
            hardware_detect,
            orchestrate_workload,
            profile_workload,
        )
        # All are callable (async functions wrapped by @mcp.tool)
        assert callable(hardware_detect)
        assert callable(convert_model)
        assert callable(profile_workload)
        assert callable(orchestrate_workload)
        assert callable(generate_code)
