"""Standalone lightweight CLI for the QUAD client.

Exposed as the ``quad-client`` entry point. Intentionally separate from
``quad.cli.main`` (which pulls in adapters / runtime / compiler /
codegen) — this CLI only depends on ``quad.client``, so a pure
client-side install via the ``[client]`` extras has minimal surface.

Commands:
    quad-client install        # interactive provisioning + connection test
    quad-client status         # what's installed
    quad-client preview        # print settings.json that install would write
    quad-client uninstall      # remove bundled skills
    quad-client connect-test   # verify a transport without writing files

Used by:
    install-client.sh          # interactive client-machine setup
    docs/CLIENT_INSTALL.md     # walkthrough of the manual flow
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from quad.client import get_provisioner
from quad.client.claude_code import ClaudeCodeProvisioner
from quad.client.claude_code.provisioner import TransportConfig
from quad.client.connection import probe

app = typer.Typer(
    name="quad-client",
    help=(
        "Lightweight QUAD MCP client provisioner. Configures Claude Code "
        "(or another MCP-compatible IDE) to talk to a QUAD MCP server. "
        "The MCP server itself can run locally, on a remote machine over SSH, "
        "or as a hosted HTTP/SSE service — see `quad-client install --help`."
    ),
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


# ─── install ────────────────────────────────────────────────────────────────


@app.command()
def install(
    transport: str = typer.Option(
        "stdio-local",
        help="MCP transport: 'stdio-local' | 'stdio-ssh' | 'sse-http'",
    ),
    # stdio-local
    server_command: str = typer.Option("python", help="(stdio-local) command to spawn the server"),
    adapter_mode: str = typer.Option("mock", help="(stdio-local) adapter mode for the server"),
    # stdio-ssh
    ssh_user: str = typer.Option("", help="(stdio-ssh) remote SSH username"),
    ssh_host: str = typer.Option("", help="(stdio-ssh) remote SSH hostname / IP"),
    ssh_port: int = typer.Option(22, help="(stdio-ssh) SSH port"),
    ssh_key: Optional[str] = typer.Option(None, help="(stdio-ssh) path to private key"),
    ssh_server_command: str = typer.Option(
        "python -m quad.mcp.server", help="(stdio-ssh) command run on the remote machine"
    ),
    # sse-http
    sse_url: str = typer.Option("", help="(sse-http) full URL to the MCP SSE endpoint"),
    sse_auth_token_env: Optional[str] = typer.Option(
        None, help="(sse-http) name of env var holding the bearer token"
    ),
    # common
    client: str = typer.Option("claude_code", help="Which IDE/agent client to provision"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config"),
    skip_test: bool = typer.Option(False, help="Skip the connection test"),
    project_root: Optional[str] = typer.Option(
        None, help="Project root (default: cwd). The .claude/ dir is created here."
    ),
) -> None:
    """Provision Claude Code (or another MCP client) to talk to the QUAD server."""
    root = Path(project_root) if project_root else Path.cwd()

    if transport == "stdio-local":
        cfg = TransportConfig(
            transport="stdio-local",
            server_command=server_command,
            adapter_mode=adapter_mode,
        )
        probe_kwargs: dict[str, Any] = {"command": server_command}
    elif transport == "stdio-ssh":
        if not ssh_host or not ssh_user:
            typer.echo("Error: stdio-ssh requires --ssh-user and --ssh-host", err=True)
            raise typer.Exit(code=2)
        cfg = TransportConfig(
            transport="stdio-ssh",
            ssh_user=ssh_user,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_key=ssh_key,
            ssh_server_command=ssh_server_command,
        )
        probe_kwargs = {
            "ssh_user": ssh_user,
            "ssh_host": ssh_host,
            "ssh_port": ssh_port,
            "ssh_key": ssh_key,
            "server_command": ssh_server_command,
        }
    elif transport == "sse-http":
        if not sse_url:
            typer.echo("Error: sse-http requires --sse-url", err=True)
            raise typer.Exit(code=2)
        cfg = TransportConfig(
            transport="sse-http",
            sse_url=sse_url,
            sse_auth_token_env=sse_auth_token_env,
        )
        probe_kwargs = {"url": sse_url}
    else:
        typer.echo(f"Error: unknown transport {transport!r}", err=True)
        typer.echo("Valid choices: stdio-local, stdio-ssh, sse-http", err=True)
        raise typer.Exit(code=2)

    # Connection test
    if not skip_test:
        typer.echo(f"Testing {transport} connection…")
        result = probe(transport, **probe_kwargs)  # type: ignore[arg-type]
        typer.echo(result.render())
        if not result.ok:
            if not force:
                typer.echo(
                    "\nConnection test failed. Re-run with --force to provision anyway, "
                    "or fix the issue above and retry.",
                    err=True,
                )
                raise typer.Exit(code=1)
            typer.echo("--force given; provisioning anyway despite failed connection test.")

    # Provision
    try:
        prov = get_provisioner(client)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    if isinstance(prov, ClaudeCodeProvisioner):
        outcome = prov.install(root, force=force, transport_config=cfg)
    else:
        outcome = prov.install(root, force=force)

    typer.echo("")
    typer.echo(f"Provisioned {outcome.client}:")
    typer.echo(f"  settings.json: {outcome.settings_path}")
    typer.echo(f"  skills dir:    {outcome.skills_dir}")
    typer.echo(f"  written:       {len(outcome.files_written)} file(s)")
    if outcome.files_skipped:
        typer.echo(f"  skipped:       {len(outcome.files_skipped)} file(s) — use --force to overwrite")
    for note in outcome.notes:
        typer.echo(f"  note: {note}")
    typer.echo("")
    typer.echo("Next: open Claude Code in this project. The QUAD MCP tools will appear in the tool list.")


# ─── status ─────────────────────────────────────────────────────────────────


@app.command()
def status(
    client: str = typer.Option("claude_code", help="Which IDE/agent client to check"),
    project_root: Optional[str] = typer.Option(None, help="Project root (default: cwd)"),
) -> None:
    """Show what's currently installed."""
    root = Path(project_root) if project_root else Path.cwd()
    try:
        prov = get_provisioner(client)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)
    s = prov.status(root)
    typer.echo(f"Client:           {s['client']}")
    typer.echo(f"settings.json:    {'present' if s['settings_exists'] else 'missing'} @ {s['settings_path']}")
    typer.echo(f"Skills dir:       {'present' if s['skills_dir_exists'] else 'missing'} @ {s['skills_dir']}")
    typer.echo(f"Bundled skills:   {s['bundled_skill_count']}")
    typer.echo(f"Installed skills: {s['installed_skill_count']}")
    if s["missing_skills"]:
        typer.echo(f"Missing:          {', '.join(s['missing_skills'])}")
    if s["extra_user_skills"]:
        typer.echo(f"User-added:       {', '.join(s['extra_user_skills'])}")


# ─── preview ────────────────────────────────────────────────────────────────


@app.command()
def preview(
    transport: str = typer.Option("stdio-local"),
    adapter_mode: str = typer.Option("mock"),
    ssh_user: str = typer.Option(""),
    ssh_host: str = typer.Option(""),
    ssh_port: int = typer.Option(22),
    ssh_key: Optional[str] = typer.Option(None),
    sse_url: str = typer.Option(""),
    sse_auth_token_env: Optional[str] = typer.Option(None),
) -> None:
    """Print the settings.json content that ``install`` would write."""
    if transport == "stdio-local":
        cfg = TransportConfig(transport="stdio-local", adapter_mode=adapter_mode)
    elif transport == "stdio-ssh":
        cfg = TransportConfig(
            transport="stdio-ssh",
            ssh_user=ssh_user, ssh_host=ssh_host, ssh_port=ssh_port, ssh_key=ssh_key,
        )
    elif transport == "sse-http":
        cfg = TransportConfig(
            transport="sse-http", sse_url=sse_url, sse_auth_token_env=sse_auth_token_env,
        )
    else:
        typer.echo(f"Unknown transport: {transport}", err=True)
        raise typer.Exit(code=2)
    typer.echo(json.dumps(cfg.to_settings(), indent=2))


# ─── uninstall ──────────────────────────────────────────────────────────────


@app.command()
def uninstall(
    client: str = typer.Option("claude_code"),
    project_root: Optional[str] = typer.Option(None),
) -> None:
    """Remove the bundled skill files (settings.json is preserved)."""
    root = Path(project_root) if project_root else Path.cwd()
    try:
        prov = get_provisioner(client)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)
    result = prov.uninstall(root)
    typer.echo(f"Uninstalled {result.client} from {result.skills_dir}")
    for note in result.notes:
        typer.echo(f"  {note}")


# ─── connect-test ───────────────────────────────────────────────────────────


@app.command("connect-test")
def connect_test(
    transport: str = typer.Argument(..., help="stdio-local | stdio-ssh | sse-http"),
    server_command: str = typer.Option("python", help="(stdio-local) command to test"),
    ssh_user: str = typer.Option(""),
    ssh_host: str = typer.Option(""),
    ssh_port: int = typer.Option(22),
    ssh_key: Optional[str] = typer.Option(None),
    server: str = typer.Option("python -m quad.mcp.server", help="(stdio-ssh) remote command"),
    sse_url: str = typer.Option("", help="(sse-http) URL"),
    auth_token: Optional[str] = typer.Option(None, help="(sse-http) bearer token"),
) -> None:
    """Verify a transport works without writing any files. Exits non-zero on failure."""
    kwargs: dict[str, Any] = {}
    if transport == "stdio-local":
        kwargs = {"command": server_command}
    elif transport == "stdio-ssh":
        if not ssh_host or not ssh_user:
            typer.echo("stdio-ssh requires --ssh-user and --ssh-host", err=True)
            raise typer.Exit(code=2)
        kwargs = {
            "ssh_user": ssh_user,
            "ssh_host": ssh_host,
            "ssh_port": ssh_port,
            "ssh_key": ssh_key,
            "server_command": server,
        }
    elif transport == "sse-http":
        if not sse_url:
            typer.echo("sse-http requires --sse-url", err=True)
            raise typer.Exit(code=2)
        kwargs = {"url": sse_url, "auth_token": auth_token}
    else:
        typer.echo(f"Unknown transport: {transport}", err=True)
        raise typer.Exit(code=2)

    result = probe(transport, **kwargs)  # type: ignore[arg-type]
    typer.echo(result.render())
    if not result.ok:
        raise typer.Exit(code=1)


def cli() -> None:
    """Entry point for the ``quad-client`` script."""
    app()


if __name__ == "__main__":
    cli()
