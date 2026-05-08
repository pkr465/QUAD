"""Pure operations — the underlying logic of each MCP tool.

Each function here returns a clean Pydantic-dataclass-derived dict
with no UI / tips / suggestions enrichment. Higher layers
(``quad.mcp``, ``quad.cli``) add presentational concerns on top of
these.

This is the API a future ``quad-python-sdk`` would expose to users
who want to call QUAD operations programmatically without going
through the MCP protocol.
"""

from quad.core.operations.convert_model import convert_model
from quad.core.operations.generate_code import generate_code
from quad.core.operations.hardware_detect import hardware_detect
from quad.core.operations.orchestrate_workload import orchestrate_workload
from quad.core.operations.profile_workload import profile_workload

__all__ = [
    "convert_model",
    "generate_code",
    "hardware_detect",
    "orchestrate_workload",
    "profile_workload",
]
