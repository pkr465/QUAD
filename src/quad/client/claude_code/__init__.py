"""Claude Code-specific provisioning for the QUAD MCP server.

Generates ``.claude/settings.json`` and installs the bundled skill
files into ``.claude/skills/``. The MCP server itself
(``quad.mcp.server``) is unchanged.

Public API::

    from quad.client.claude_code import ClaudeCodeProvisioner
    p = ClaudeCodeProvisioner()
    p.install(Path.cwd())
    p.status(Path.cwd())
"""

from quad.client.claude_code.provisioner import ClaudeCodeProvisioner
from quad.client.claude_code.settings import (
    DEFAULT_PERMISSIONS,
    build_settings,
    settings_template,
)
from quad.client.claude_code.skills import (
    bundled_skills_dir,
    install_skills,
    list_bundled_skills,
)

__all__ = [
    "ClaudeCodeProvisioner",
    "DEFAULT_PERMISSIONS",
    "build_settings",
    "settings_template",
    "bundled_skills_dir",
    "install_skills",
    "list_bundled_skills",
]
