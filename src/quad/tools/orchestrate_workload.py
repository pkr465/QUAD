"""Backward-compat shim — re-exports from :mod:`quad.mcp.tools` and
:mod:`quad.core.operations`.

The orchestration algorithm now lives in
:mod:`quad.core.operations.orchestrate_workload`; the MCP wrapper is in
:mod:`quad.mcp.tools`. This module preserves the import paths used by
existing tests (``_profile_with_layers``, ``orchestrate_workload_impl``).
"""

from quad.core.operations.orchestrate_workload import _profile_with_layers
from quad.mcp.tools import orchestrate_workload_impl

__all__ = ["_profile_with_layers", "orchestrate_workload_impl"]
