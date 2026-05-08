"""Tests for the multi-transport settings.json templates + TransportConfig."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quad.client.claude_code import (
    ClaudeCodeProvisioner,
    settings_template,
    settings_template_sse,
    settings_template_ssh,
)
from quad.client.claude_code.provisioner import TransportConfig


# ─── stdio-ssh template ───────────────────────────────────────────────────


class TestSettingsTemplateSSH:
    def test_basic_ssh_settings(self) -> None:
        s = settings_template_ssh(ssh_user="alice", ssh_host="server.example.com")
        srv = s["mcpServers"]["quad"]
        assert srv["command"] == "ssh"
        assert "alice@server.example.com" in srv["args"]
        # Default server command on remote
        assert "python -m quad.mcp.server" in srv["args"]

    def test_custom_port(self) -> None:
        s = settings_template_ssh(ssh_user="x", ssh_host="y", ssh_port=2222)
        args = s["mcpServers"]["quad"]["args"]
        # -p 2222 should appear
        assert "-p" in args
        assert "2222" in args

    def test_custom_key(self) -> None:
        s = settings_template_ssh(
            ssh_user="x", ssh_host="y", ssh_key="/home/me/.ssh/special_key"
        )
        args = s["mcpServers"]["quad"]["args"]
        assert "-i" in args
        assert "/home/me/.ssh/special_key" in args

    def test_batch_mode_set(self) -> None:
        # BatchMode=yes prevents Claude Code from hanging on a password prompt
        s = settings_template_ssh(ssh_user="x", ssh_host="y")
        args = s["mcpServers"]["quad"]["args"]
        assert "BatchMode=yes" in args

    def test_permissions_default(self) -> None:
        s = settings_template_ssh(ssh_user="x", ssh_host="y")
        for tool in (
            "mcp__quad__hardware_detect",
            "mcp__quad__convert_model",
            "mcp__quad__profile_workload",
            "mcp__quad__orchestrate_workload",
            "mcp__quad__generate_code",
        ):
            assert tool in s["permissions"]["allow"]


# ─── sse-http template ────────────────────────────────────────────────────


class TestSettingsTemplateSSE:
    def test_basic_sse_settings(self) -> None:
        s = settings_template_sse(url="https://mcp.example.com/sse")
        srv = s["mcpServers"]["quad"]
        assert "transport" in srv
        assert srv["transport"]["type"] == "sse"
        assert srv["transport"]["url"] == "https://mcp.example.com/sse"
        # No subprocess command for SSE
        assert "command" not in srv

    def test_auth_token_env(self) -> None:
        s = settings_template_sse(
            url="https://mcp.example.com/sse",
            auth_token_env="QUAD_MCP_TOKEN",
        )
        headers = s["mcpServers"]["quad"]["transport"]["headers"]
        assert "Authorization" in headers
        # Claude Code env-var substitution syntax
        assert "QUAD_MCP_TOKEN" in headers["Authorization"]

    def test_no_auth_token_omits_headers(self) -> None:
        s = settings_template_sse(url="https://mcp.example.com/sse")
        # No auth → no headers field
        srv = s["mcpServers"]["quad"]
        assert "headers" not in srv["transport"]


# ─── TransportConfig dispatch ────────────────────────────────────────────


class TestTransportConfig:
    def test_stdio_local_default(self) -> None:
        cfg = TransportConfig()
        s = cfg.to_settings()
        assert s["mcpServers"]["quad"]["command"] == "python"

    def test_stdio_ssh(self) -> None:
        cfg = TransportConfig(
            transport="stdio-ssh",
            ssh_user="bob",
            ssh_host="snapdragon-test.lan",
            ssh_port=22,
        )
        s = cfg.to_settings()
        assert s["mcpServers"]["quad"]["command"] == "ssh"
        assert "bob@snapdragon-test.lan" in s["mcpServers"]["quad"]["args"]

    def test_sse_http(self) -> None:
        cfg = TransportConfig(
            transport="sse-http",
            sse_url="https://hosted.mcp.cloud/sse",
        )
        s = cfg.to_settings()
        assert s["mcpServers"]["quad"]["transport"]["type"] == "sse"
        assert s["mcpServers"]["quad"]["transport"]["url"] == "https://hosted.mcp.cloud/sse"

    def test_unknown_raises(self) -> None:
        cfg = TransportConfig(transport="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown transport"):
            cfg.to_settings()


# ─── Provisioner end-to-end with non-default transport ───────────────────


class TestProvisionerWithTransport:
    def test_install_with_ssh_transport(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        cfg = TransportConfig(
            transport="stdio-ssh",
            ssh_user="bob",
            ssh_host="server.example.com",
        )
        result = prov.install(tmp_path, transport_config=cfg)
        assert result.client == "claude_code"

        settings_file = tmp_path / ".claude" / "settings.json"
        settings = json.loads(settings_file.read_text())
        srv = settings["mcpServers"]["quad"]
        assert srv["command"] == "ssh"
        assert "bob@server.example.com" in srv["args"]

    def test_install_with_sse_transport(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        cfg = TransportConfig(
            transport="sse-http",
            sse_url="https://hosted.mcp.com/sse",
            sse_auth_token_env="MY_TOKEN",
        )
        prov.install(tmp_path, transport_config=cfg)

        settings_file = tmp_path / ".claude" / "settings.json"
        settings = json.loads(settings_file.read_text())
        srv = settings["mcpServers"]["quad"]
        assert srv["transport"]["type"] == "sse"
        assert srv["transport"]["url"] == "https://hosted.mcp.com/sse"
        assert "MY_TOKEN" in srv["transport"]["headers"]["Authorization"]

    def test_install_default_transport_unchanged(self, tmp_path: Path) -> None:
        # Backward-compat: install with no transport_config still produces
        # the original stdio-local settings shape
        prov = ClaudeCodeProvisioner()
        prov.install(tmp_path)

        settings_file = tmp_path / ".claude" / "settings.json"
        settings = json.loads(settings_file.read_text())
        srv = settings["mcpServers"]["quad"]
        assert srv["command"] == "python"
        assert srv["args"] == ["-m", "quad.mcp.server"]
