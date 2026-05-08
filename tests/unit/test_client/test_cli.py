"""Tests for the standalone ``quad-client`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quad.client.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ─── preview ──────────────────────────────────────────────────────────────


class TestPreview:
    def test_default_stdio_local(self, runner: CliRunner) -> None:
        r = runner.invoke(app, ["preview"])
        assert r.exit_code == 0
        parsed = json.loads(r.stdout)
        assert parsed["mcpServers"]["quad"]["command"] == "python"

    def test_stdio_ssh(self, runner: CliRunner) -> None:
        r = runner.invoke(
            app,
            ["preview", "--transport", "stdio-ssh", "--ssh-user", "alice", "--ssh-host", "x.lan"],
        )
        assert r.exit_code == 0
        parsed = json.loads(r.stdout)
        assert parsed["mcpServers"]["quad"]["command"] == "ssh"
        assert "alice@x.lan" in parsed["mcpServers"]["quad"]["args"]

    def test_sse_http(self, runner: CliRunner) -> None:
        r = runner.invoke(
            app,
            [
                "preview",
                "--transport", "sse-http",
                "--sse-url", "https://m.example.com/sse",
                "--sse-auth-token-env", "MY_TOKEN",
            ],
        )
        assert r.exit_code == 0
        parsed = json.loads(r.stdout)
        srv = parsed["mcpServers"]["quad"]
        assert srv["transport"]["type"] == "sse"
        assert srv["transport"]["url"] == "https://m.example.com/sse"

    def test_unknown_transport_returns_2(self, runner: CliRunner) -> None:
        r = runner.invoke(app, ["preview", "--transport", "bogus"])
        assert r.exit_code == 2


# ─── status ───────────────────────────────────────────────────────────────


class TestStatus:
    def test_status_when_nothing_installed(self, runner: CliRunner, tmp_path: Path) -> None:
        r = runner.invoke(app, ["status", "--project-root", str(tmp_path)])
        assert r.exit_code == 0
        assert "settings.json" in r.stdout
        assert "missing" in r.stdout

    def test_unknown_client_errors(self, runner: CliRunner) -> None:
        r = runner.invoke(app, ["status", "--client", "cursor"])
        assert r.exit_code == 2


# ─── install ──────────────────────────────────────────────────────────────


class TestInstall:
    def test_stdio_local_install_with_skip_test(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        r = runner.invoke(
            app,
            [
                "install",
                "--project-root", str(tmp_path),
                "--skip-test",
            ],
        )
        assert r.exit_code == 0, r.stdout
        assert (tmp_path / ".claude" / "settings.json").is_file()

    def test_stdio_ssh_requires_user_and_host(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        r = runner.invoke(
            app,
            [
                "install",
                "--transport", "stdio-ssh",
                "--project-root", str(tmp_path),
            ],
        )
        assert r.exit_code == 2

    def test_sse_http_requires_url(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        r = runner.invoke(
            app,
            [
                "install",
                "--transport", "sse-http",
                "--project-root", str(tmp_path),
            ],
        )
        assert r.exit_code == 2

    def test_install_provisions_sse_settings(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        r = runner.invoke(
            app,
            [
                "install",
                "--transport", "sse-http",
                "--sse-url", "https://m.example.com/sse",
                "--project-root", str(tmp_path),
                "--skip-test",
            ],
        )
        assert r.exit_code == 0, r.stdout
        settings = json.loads(
            (tmp_path / ".claude" / "settings.json").read_text()
        )
        assert settings["mcpServers"]["quad"]["transport"]["type"] == "sse"


# ─── connect-test ─────────────────────────────────────────────────────────


class TestConnectTest:
    def test_unknown_transport(self, runner: CliRunner) -> None:
        r = runner.invoke(app, ["connect-test", "fake-transport"])
        assert r.exit_code == 2

    def test_stdio_ssh_requires_args(self, runner: CliRunner) -> None:
        r = runner.invoke(app, ["connect-test", "stdio-ssh"])
        assert r.exit_code == 2

    def test_sse_http_requires_url(self, runner: CliRunner) -> None:
        r = runner.invoke(app, ["connect-test", "sse-http"])
        assert r.exit_code == 2


# ─── lightweight import — does the client CLI pull heavy modules? ─────────


class TestLightweightImport:
    """The whole point of the standalone CLI: importing ``quad.client.cli``
    should NOT pull in adapters / runtime / compiler / codegen / serve
    / profiler. Runs in a subprocess so we don't pollute the test
    process's module state.
    """

    def test_client_cli_does_not_pull_heavy_modules(self) -> None:
        import subprocess
        import sys

        # Run a fresh interpreter that imports only quad.client.cli and
        # then prints any heavy modules that came along for the ride.
        probe = (
            "import sys; "
            "import quad.client.cli; "
            "heavy = [m for m in sys.modules "
            "  if m.startswith('quad.') and any("
            "    bad in m for bad in ('adapters', 'runtime', 'compiler', "
            "                          'codegen', 'serve.http', 'profiler'))]; "
            "print('|'.join(heavy))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"probe failed: {result.stderr}"
        heavy = [m for m in result.stdout.strip().split("|") if m]
        assert not heavy, (
            f"quad.client.cli unexpectedly pulled heavy modules in a fresh interpreter:\n"
            f"  {heavy}"
        )
