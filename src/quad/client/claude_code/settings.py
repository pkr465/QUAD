"""Generate ``.claude/settings.json`` for the QUAD MCP server.

Replaces the install.sh heredoc that hardcoded the JSON structure.
This module is the single source of truth for what a Claude Code
client needs to discover + authorise the QUAD MCP server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Pre-approved tool names. Claude Code grants these without prompting
# the user. Anything not listed here will require explicit approval.
DEFAULT_PERMISSIONS: tuple[str, ...] = (
    "mcp__quad__hardware_detect",
    "mcp__quad__convert_model",
    "mcp__quad__profile_workload",
    "mcp__quad__orchestrate_workload",
    "mcp__quad__generate_code",
)


def settings_template(
    *,
    server_command: str = "python",
    server_args: list[str] | None = None,
    cwd: str = "${workspaceFolder}",
    adapter_mode: str = "mock",
    permissions: tuple[str, ...] | list[str] = DEFAULT_PERMISSIONS,
) -> dict[str, Any]:
    """Local-stdio settings.json — Claude Code spawns the MCP server here.

    This is the default for the most common topology: server + client on
    the same machine. For SSH-tunnelled or HTTP/SSE servers, use
    :func:`settings_template_ssh` or :func:`settings_template_sse`.

    Args:
        server_command: how Claude Code spawns the MCP server (default ``python``).
        server_args: ``-m quad.mcp.server`` by default; set to invoke
            a custom entry point.
        cwd: working directory token Claude Code substitutes.
        adapter_mode: ``mock`` or ``real`` — written to env so the
            server picks it up at startup.
        permissions: pre-approved MCP tool names.
    """
    if server_args is None:
        server_args = ["-m", "quad.mcp.server"]

    return {
        "permissions": {"allow": list(permissions)},
        "mcpServers": {
            "quad": {
                "command": server_command,
                "args": list(server_args),
                "cwd": cwd,
                "env": {
                    "QUAD_ADAPTER_MODE": adapter_mode,
                },
            }
        },
    }


def settings_template_ssh(
    *,
    ssh_user: str,
    ssh_host: str,
    ssh_port: int = 22,
    ssh_key: str | None = None,
    server_command: str = "python -m quad.mcp.server",
    permissions: tuple[str, ...] | list[str] = DEFAULT_PERMISSIONS,
) -> dict[str, Any]:
    """SSH-tunnelled stdio settings.json — Claude Code talks to a remote server over SSH.

    The MCP protocol is stdio; SSH proxies it. The user must have
    passwordless SSH (key-based) auth set up to ``ssh_user@ssh_host``,
    otherwise Claude Code will hang on the password prompt.

    Args:
        ssh_user: remote login username
        ssh_host: hostname or IP of the server machine
        ssh_port: SSH port (default 22)
        ssh_key: path to a specific private key (default: SSH agent / ~/.ssh/id_rsa)
        server_command: command run on the remote machine to start the MCP server
        permissions: pre-approved MCP tool names
    """
    args = ["-p", str(ssh_port)]
    if ssh_key:
        args += ["-i", ssh_key]
    args += [
        "-o", "BatchMode=yes",
        "-o", "ServerAliveInterval=30",
        f"{ssh_user}@{ssh_host}",
        server_command,
    ]
    return {
        "permissions": {"allow": list(permissions)},
        "mcpServers": {
            "quad": {
                "command": "ssh",
                "args": args,
            }
        },
    }


def settings_template_sse(
    *,
    url: str,
    auth_token_env: str | None = None,
    permissions: tuple[str, ...] | list[str] = DEFAULT_PERMISSIONS,
) -> dict[str, Any]:
    """SSE/HTTP settings.json — remote MCP server over HTTP/Server-Sent Events.

    Used for hosted MCP servers (e.g. behind a load balancer / TLS-terminating
    reverse proxy). The MCP protocol is wrapped in SSE; Claude Code maintains
    an HTTP connection rather than spawning a subprocess.

    Args:
        url: full URL to the MCP server's SSE endpoint (e.g. https://mcp.example.com/sse)
        auth_token_env: name of an env var that holds the bearer token. Claude Code
            will read it and inject as Authorization: Bearer header. Optional.
        permissions: pre-approved MCP tool names
    """
    server_block: dict[str, Any] = {
        "transport": {
            "type": "sse",
            "url": url,
        },
    }
    if auth_token_env:
        server_block["transport"]["headers"] = {
            "Authorization": f"Bearer ${{env:{auth_token_env}}}",
        }
    return {
        "permissions": {"allow": list(permissions)},
        "mcpServers": {"quad": server_block},
    }


def build_settings(
    *,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Like :func:`settings_template` but accepts a final ``overrides`` dict.

    The overrides dict is merged on top of the template so callers can
    customise any field (e.g. add a second MCP server, change permissions)
    without re-implementing the whole template.
    """
    settings = settings_template(**kwargs)
    if overrides:
        _deep_merge(settings, overrides)
    return settings


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """In-place recursive dict merge — overlay wins."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def render_to_json(settings: dict[str, Any], *, indent: int = 2) -> str:
    """Serialise settings to JSON with a trailing newline."""
    return json.dumps(settings, indent=indent) + "\n"


def write_settings(
    path: Path,
    *,
    settings: dict[str, Any] | None = None,
    force: bool = False,
    **kwargs: Any,
) -> bool:
    """Write the settings.json file. Returns True if the file was written.

    If the file already exists and ``force`` is False, returns False
    without overwriting.
    """
    if path.exists() and not force:
        return False
    if settings is None:
        settings = build_settings(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_to_json(settings))
    return True
