"""QUAD client integrations — IDE / agent-specific provisioning.

The MCP server (``quad.mcp.server``) is the same regardless of which
client (Claude Code, Cursor, Continue, …) connects to it. What
differs per client is **how that client discovers and authorises the
server** — typically via a config file (e.g. ``.claude/settings.json``)
and a set of skills / instruction files.

This package isolates that client-specific provisioning so the MCP
server stays generic, and adding support for a new client is a focused
change.

Public abstract base:

    from quad.client import MCPClientProvisioner

Concrete implementations:

    from quad.client.claude_code import ClaudeCodeProvisioner

CLI entry point: ``quad client install [--client claude_code]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProvisionResult:
    """Result of installing client-side config + skills."""

    client: str
    files_written: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    settings_path: str | None = None
    skills_dir: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "client": self.client,
            "files_written": self.files_written,
            "files_skipped": self.files_skipped,
            "settings_path": self.settings_path,
            "skills_dir": self.skills_dir,
            "notes": self.notes,
        }


class MCPClientProvisioner(ABC):
    """Abstract base for client-side QUAD MCP integrations.

    Subclasses implement how to:
      * Generate the client's config file (e.g. settings.json)
      * Install skill / instruction files into the client's expected
        location
      * Verify an existing install
      * Uninstall

    The MCP server itself is always the same: ``quad.mcp.server:cli``,
    invoked by the client via stdio.
    """

    name: str = "abstract"

    @abstractmethod
    def install(self, project_root: Path, *, force: bool = False) -> ProvisionResult:
        """Install client config + skills under ``project_root``."""

    @abstractmethod
    def uninstall(self, project_root: Path) -> ProvisionResult:
        """Remove the client config + skills (does not affect QUAD itself)."""

    @abstractmethod
    def status(self, project_root: Path) -> dict[str, Any]:
        """Report what's currently installed under ``project_root``."""


def get_provisioner(client: str = "claude_code") -> MCPClientProvisioner:
    """Factory — returns a provisioner for the given client name."""
    if client == "claude_code":
        from quad.client.claude_code.provisioner import ClaudeCodeProvisioner

        return ClaudeCodeProvisioner()
    raise ValueError(
        f"Unknown MCP client: {client!r}. Supported: 'claude_code'. "
        "Other clients (cursor, continue, cline) can be added by implementing "
        "MCPClientProvisioner in a new submodule under quad.client."
    )


__all__ = [
    "MCPClientProvisioner",
    "ProvisionResult",
    "get_provisioner",
]
