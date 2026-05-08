"""MCP server connection probing — verify reachability before provisioning.

The client install needs to confirm that the MCP server it's about to
configure Claude Code to invoke is actually reachable. Three transports
are supported:

1. **stdio-local**  — the server runs on the same machine; Claude Code
                       spawns it as a child process. Probe = launch the
                       command and verify it starts cleanly.

2. **stdio-ssh**    — the server runs on a remote machine; Claude Code
                       proxies through SSH (``ssh user@host python -m quad.mcp.server``).
                       Probe = SSH to the host and run the same command,
                       verify ``python -c "import quad.mcp.server"`` succeeds.

3. **sse-http**     — the server runs as a remote HTTP service speaking
                       MCP over Server-Sent Events. Probe = HTTP GET on
                       the URL, verify a 200 / 405 / 404 (any 4xx that
                       isn't 401/403 means the URL is reachable; auth
                       failure is also ``reachable``).

Every probe returns a :class:`ConnectionProbeResult` with:
  * ``ok`` — boolean
  * ``transport`` — which transport was tested
  * ``latency_ms`` — measured handshake time
  * ``error`` — exception message on failure
  * ``hint`` — human-readable next-step suggestion when ok is False
  * ``details`` — extra data (server version, tool count, etc.) when known

The probe is best-effort — it never raises. Failures should be reported
to the user with the hint, but shouldn't block them from completing the
install (the user may be intentionally provisioning before the server is
running).
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


Transport = Literal["stdio-local", "stdio-ssh", "sse-http"]


@dataclass
class ConnectionProbeResult:
    """Outcome of a connection probe."""

    ok: bool
    transport: Transport
    latency_ms: float = 0.0
    error: str = ""
    hint: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        """Human-readable single-string summary."""
        if self.ok:
            return (
                f"✓ {self.transport} reachable in {self.latency_ms:.0f} ms"
                + (f" — {self.details}" if self.details else "")
            )
        msg = f"✗ {self.transport} probe failed: {self.error}"
        if self.hint:
            msg += f"\n  → {self.hint}"
        return msg


# ─── Probe: stdio-local ─────────────────────────────────────────────────────


def probe_stdio_local(
    command: str = "python",
    args: list[str] | None = None,
    *,
    timeout_s: float = 8.0,
    env: dict[str, str] | None = None,
) -> ConnectionProbeResult:
    """Verify the local MCP server can start.

    Launches ``command args`` as a subprocess. Doesn't actually speak
    MCP — just verifies the process starts, stays running for ~1 second,
    and exits cleanly when terminated. This catches the common failures:
    Python missing, package not importable, syntax errors at import time.
    """
    if args is None:
        args = ["-m", "quad.mcp.server"]

    cmd_path = shutil.which(command)
    if cmd_path is None:
        return ConnectionProbeResult(
            ok=False,
            transport="stdio-local",
            error=f"Command not found on PATH: {command!r}",
            hint=(
                f"Install Python 3.10+ and ensure 'python' (or {command!r}) is on PATH. "
                "Run: where python (Windows) or which python (POSIX)."
            ),
        )

    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    start = time.perf_counter()
    try:
        proc = subprocess.Popen(
            [command, *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
        )
    except FileNotFoundError as e:
        return ConnectionProbeResult(
            ok=False,
            transport="stdio-local",
            error=str(e),
            hint=f"Could not exec {command!r}. Check PATH and permissions.",
        )

    # Give it up to ~1 second to confirm it didn't crash on startup
    try:
        # If it exits within 1s, that's a startup error
        rc = proc.wait(timeout=1.0)
        elapsed = (time.perf_counter() - start) * 1000.0
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        return ConnectionProbeResult(
            ok=False,
            transport="stdio-local",
            latency_ms=elapsed,
            error=f"server exited immediately with code {rc}",
            hint=(
                "MCP server crashed during startup. Check that the package is "
                f"importable: python -c 'import quad.mcp.server'. stderr:\n  {stderr.strip()[:500]}"
            ),
            details={"returncode": rc, "stderr_excerpt": stderr.strip()[:500]},
        )
    except subprocess.TimeoutExpired:
        # Good — it didn't crash. Now terminate cleanly.
        elapsed = (time.perf_counter() - start) * 1000.0
        proc.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
        return ConnectionProbeResult(
            ok=True,
            transport="stdio-local",
            latency_ms=elapsed,
            details={"command": command, "args": args},
        )


# ─── Probe: stdio-ssh ───────────────────────────────────────────────────────


def probe_stdio_ssh(
    ssh_user: str,
    ssh_host: str,
    *,
    ssh_port: int = 22,
    ssh_key: str | None = None,
    server_command: str = "python -m quad.mcp.server",
    timeout_s: float = 12.0,
) -> ConnectionProbeResult:
    """Verify a remote MCP server is reachable + the package is importable there.

    Runs ``ssh user@host python -c 'import quad.mcp.server'`` and checks
    the exit code. Doesn't actually speak MCP — just confirms the
    plumbing works.
    """
    if not shutil.which("ssh"):
        return ConnectionProbeResult(
            ok=False,
            transport="stdio-ssh",
            error="ssh command not found on PATH",
            hint=(
                "Install OpenSSH client. On Windows: winget install Microsoft.OpenSSH.Client. "
                "On macOS/Linux: usually preinstalled."
            ),
        )

    # We can probe with: ssh -o BatchMode=yes -o ConnectTimeout=N user@host
    #     python -c "import quad.mcp.server"
    ssh_cmd = ["ssh", "-p", str(ssh_port)]
    if ssh_key:
        ssh_cmd += ["-i", ssh_key]
    # BatchMode prevents prompting for password — fail fast if key auth doesn't work
    ssh_cmd += [
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(timeout_s)}",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{ssh_user}@{ssh_host}",
    ]
    # Build the remote probe — just import the server module
    probe = "python -c 'import quad.mcp.server; print(\"OK\")'"
    ssh_cmd.append(probe)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return ConnectionProbeResult(
            ok=False,
            transport="stdio-ssh",
            latency_ms=timeout_s * 1000.0,
            error=f"SSH connection timed out after {timeout_s}s",
            hint=(
                f"Could not reach {ssh_host}:{ssh_port}. Check the host is up, "
                "the port is open, and your SSH key is configured. "
                f"Test manually: ssh -p {ssh_port} {ssh_user}@{ssh_host}"
            ),
        )

    elapsed = (time.perf_counter() - start) * 1000.0
    if result.returncode == 0 and "OK" in result.stdout:
        return ConnectionProbeResult(
            ok=True,
            transport="stdio-ssh",
            latency_ms=elapsed,
            details={
                "host": ssh_host,
                "user": ssh_user,
                "port": ssh_port,
            },
        )

    # Diagnose common failures
    stderr = result.stderr.strip()
    if "Permission denied" in stderr or "publickey" in stderr.lower():
        hint = (
            f"SSH key auth to {ssh_user}@{ssh_host} failed. Add your public key "
            "to the server's ~/.ssh/authorized_keys, or specify a key with --ssh-key."
        )
    elif "Could not resolve hostname" in stderr or "Name or service not known" in stderr:
        hint = f"Hostname {ssh_host!r} did not resolve. Check spelling and DNS."
    elif "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
        hint = (
            f"SSH worked but quad.mcp.server isn't importable on {ssh_host}. "
            "Install the QUAD package on the server: pip install quad-agent."
        )
    else:
        hint = "Check stderr above + run the ssh command manually to debug."

    return ConnectionProbeResult(
        ok=False,
        transport="stdio-ssh",
        latency_ms=elapsed,
        error=f"ssh exit {result.returncode}: {stderr[:300]}",
        hint=hint,
        details={
            "returncode": result.returncode,
            "stderr_excerpt": stderr[:500],
        },
    )


# ─── Probe: sse-http ────────────────────────────────────────────────────────


def probe_sse_http(
    url: str,
    *,
    auth_token: str | None = None,
    timeout_s: float = 8.0,
) -> ConnectionProbeResult:
    """Verify a remote MCP-over-HTTP/SSE endpoint is reachable.

    Issues an HTTP GET. The server may respond with any of:
      * 200 — fully reachable + likely speaks MCP
      * 401/403 — reachable but auth failed (still counts as "reachable",
                  the user just needs the right token)
      * 404/405 — reachable but the URL path is wrong (counted as reachable
                  with a hint)
      * connection refused / timeout — server unreachable
    """
    try:
        import httpx
    except ImportError:
        return ConnectionProbeResult(
            ok=False,
            transport="sse-http",
            error="httpx not installed",
            hint=(
                "Install via: pip install httpx. Or use the [client] extras: "
                "pip install quad-agent[client]"
            ),
        )

    headers: dict[str, str] = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    start = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
            response = client.get(url, headers=headers)
        elapsed = (time.perf_counter() - start) * 1000.0
    except httpx.ConnectError as e:
        return ConnectionProbeResult(
            ok=False,
            transport="sse-http",
            error=f"connection refused: {e}",
            hint=(
                f"Could not connect to {url}. Check the server is running, "
                "the URL is correct (scheme + host + port), and any firewall "
                "rules allow your IP."
            ),
        )
    except httpx.TimeoutException:
        return ConnectionProbeResult(
            ok=False,
            transport="sse-http",
            latency_ms=timeout_s * 1000.0,
            error=f"connection timed out after {timeout_s}s",
            hint=(
                f"Server at {url} did not respond within {timeout_s}s. "
                "Check it's running and reachable from this network."
            ),
        )
    except Exception as e:
        return ConnectionProbeResult(
            ok=False,
            transport="sse-http",
            error=f"{type(e).__name__}: {e}",
            hint="See error above. Run the request manually with curl to debug.",
        )

    code = response.status_code
    server_header = response.headers.get("server", "")

    if 200 <= code < 300:
        return ConnectionProbeResult(
            ok=True,
            transport="sse-http",
            latency_ms=elapsed,
            details={"status_code": code, "server": server_header},
        )

    if code in (401, 403):
        return ConnectionProbeResult(
            ok=False,
            transport="sse-http",
            latency_ms=elapsed,
            error=f"auth failed: HTTP {code}",
            hint=(
                "Server is reachable but the auth token was rejected. "
                "Check QAI_HUB_API_KEY / Bearer token / cookies."
            ),
            details={"status_code": code, "server": server_header},
        )

    if code in (404, 405):
        return ConnectionProbeResult(
            ok=False,
            transport="sse-http",
            latency_ms=elapsed,
            error=f"endpoint not found: HTTP {code}",
            hint=(
                f"Server is reachable but {url} returned {code}. "
                "Check the URL path — MCP servers typically expose /sse, /mcp, or /."
            ),
            details={"status_code": code, "server": server_header},
        )

    return ConnectionProbeResult(
        ok=False,
        transport="sse-http",
        latency_ms=elapsed,
        error=f"HTTP {code}",
        hint=f"Server responded with an unexpected status. Check {url} manually.",
        details={"status_code": code, "server": server_header},
    )


# ─── Public dispatch ────────────────────────────────────────────────────────


def probe(
    transport: Transport,
    /,
    **kwargs: Any,
) -> ConnectionProbeResult:
    """Dispatch to the appropriate probe based on transport type."""
    if transport == "stdio-local":
        return probe_stdio_local(**kwargs)
    if transport == "stdio-ssh":
        return probe_stdio_ssh(**kwargs)
    if transport == "sse-http":
        return probe_sse_http(**kwargs)
    return ConnectionProbeResult(
        ok=False,
        transport=transport,  # type: ignore[arg-type]
        error=f"Unknown transport: {transport!r}",
        hint="Supported transports: stdio-local, stdio-ssh, sse-http",
    )
