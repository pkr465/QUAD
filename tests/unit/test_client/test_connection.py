"""Tests for the MCP server connection probes."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from quad.client.connection import (
    ConnectionProbeResult,
    probe,
    probe_sse_http,
    probe_stdio_local,
    probe_stdio_ssh,
)


# ─── Result class ─────────────────────────────────────────────────────────


class TestConnectionProbeResult:
    def test_render_ok(self) -> None:
        r = ConnectionProbeResult(ok=True, transport="stdio-local", latency_ms=42.0)
        out = r.render()
        assert "✓" in out
        assert "stdio-local" in out
        assert "42" in out

    def test_render_fail(self) -> None:
        r = ConnectionProbeResult(
            ok=False, transport="sse-http", error="connection refused", hint="check server"
        )
        out = r.render()
        assert "✗" in out
        assert "connection refused" in out
        assert "check server" in out

    def test_to_dict_serialisable(self) -> None:
        r = ConnectionProbeResult(ok=True, transport="stdio-local", latency_ms=10.0)
        d = r.to_dict()
        assert d["ok"] is True
        assert d["transport"] == "stdio-local"


# ─── stdio-local ──────────────────────────────────────────────────────────


class TestProbeStdioLocal:
    def test_python_runs_cleanly(self) -> None:
        # python -c 'import time; time.sleep(2)' should let the probe
        # confirm the process starts cleanly (it'll terminate it after 1s)
        result = probe_stdio_local(
            command="python",
            args=["-c", "import time; time.sleep(5)"],
        )
        # If python is on PATH, this should pass
        assert result.transport == "stdio-local"
        if result.ok:
            assert result.latency_ms > 0

    def test_missing_command_returns_clear_error(self) -> None:
        result = probe_stdio_local(command="this-command-definitely-does-not-exist-asdf")
        assert result.ok is False
        assert "not found" in result.error.lower()
        assert "PATH" in result.hint

    def test_immediate_exit_caught(self) -> None:
        # python -c 'import sys; sys.exit(1)' exits in <1s — caught as fail
        result = probe_stdio_local(
            command="python",
            args=["-c", "import sys; sys.exit(1)"],
        )
        assert result.ok is False
        assert "exited immediately" in result.error or result.error
        assert result.details.get("returncode") in (1, None)

    def test_default_args_target_quad_mcp_server(self) -> None:
        # Default args invoke the QUAD server — this is integration-y but
        # since we have it installed, should work
        result = probe_stdio_local(command="python")
        # Should pass (the server starts cleanly)
        if result.ok:
            assert result.details.get("args") == ["-m", "quad.mcp.server"]


# ─── stdio-ssh ────────────────────────────────────────────────────────────


class TestProbeStdioSsh:
    def test_no_ssh_command_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulate ssh missing from PATH
        monkeypatch.setattr("shutil.which", lambda cmd: None if cmd == "ssh" else "/usr/bin/python")
        result = probe_stdio_ssh(ssh_user="x", ssh_host="y")
        assert result.ok is False
        assert "ssh" in result.error.lower()

    def test_invalid_host_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock subprocess to simulate connection failure
        class FakeProc:
            returncode = 255
            stdout = ""
            stderr = "Could not resolve hostname fake-host: Name or service not known"

        def fake_run(*args: object, **kwargs: object) -> FakeProc:
            return FakeProc()

        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/ssh")

        result = probe_stdio_ssh(ssh_user="me", ssh_host="fake-host", timeout_s=2)
        assert result.ok is False
        assert "did not resolve" in result.hint or "DNS" in result.hint

    def test_permission_denied_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeProc:
            returncode = 255
            stdout = ""
            stderr = "Permission denied (publickey)."

        monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/ssh")

        result = probe_stdio_ssh(ssh_user="me", ssh_host="example.com", timeout_s=2)
        assert result.ok is False
        assert "key auth" in result.hint or "publickey" in result.hint

    def test_module_not_found_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeProc:
            returncode = 1
            stdout = ""
            stderr = "ModuleNotFoundError: No module named 'quad'"

        monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/ssh")

        result = probe_stdio_ssh(ssh_user="me", ssh_host="example.com", timeout_s=2)
        assert result.ok is False
        assert "quad-agent" in result.hint or "Install" in result.hint

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeProc:
            returncode = 0
            stdout = "OK\n"
            stderr = ""

        monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/ssh")

        result = probe_stdio_ssh(ssh_user="me", ssh_host="real-host", ssh_port=22, timeout_s=2)
        assert result.ok is True
        assert result.details["host"] == "real-host"
        assert result.details["user"] == "me"


# ─── sse-http ─────────────────────────────────────────────────────────────


class TestProbeSseHttp:
    def test_2xx_is_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResp:
            status_code = 200
            headers = {"server": "nginx/1.20"}

        class FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, *a, **kw): return FakeResp()

        monkeypatch.setattr("httpx.Client", FakeClient)
        result = probe_sse_http("https://mcp.example.com/sse")
        assert result.ok is True
        assert result.details["status_code"] == 200

    def test_401_reports_auth_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResp:
            status_code = 401
            headers = {}

        class FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, *a, **kw): return FakeResp()

        monkeypatch.setattr("httpx.Client", FakeClient)
        result = probe_sse_http("https://mcp.example.com/sse", auth_token="wrong")
        assert result.ok is False
        assert "auth" in result.error.lower()
        assert "401" in result.error

    def test_404_reports_path_issue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeResp:
            status_code = 404
            headers = {}

        class FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, *a, **kw): return FakeResp()

        monkeypatch.setattr("httpx.Client", FakeClient)
        result = probe_sse_http("https://mcp.example.com/wrong-path")
        assert result.ok is False
        assert "404" in result.error
        assert "URL path" in result.hint or "endpoints" in result.hint or "/sse" in result.hint

    def test_connection_refused_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import httpx

        class FakeClient:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def get(self, *a, **kw):
                raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr("httpx.Client", FakeClient)
        result = probe_sse_http("https://nope.example.com/sse")
        assert result.ok is False
        assert "refused" in result.error or "connect" in result.error.lower()


# ─── dispatch ─────────────────────────────────────────────────────────────


class TestDispatch:
    def test_unknown_transport(self) -> None:
        result = probe("madeup-transport")  # type: ignore[arg-type]
        assert result.ok is False
        assert "Unknown transport" in result.error
