"""Tests for the Claude Code MCP client provisioner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quad.client import MCPClientProvisioner, ProvisionResult, get_provisioner
from quad.client.claude_code import (
    ClaudeCodeProvisioner,
    DEFAULT_PERMISSIONS,
    bundled_skills_dir,
    build_settings,
    list_bundled_skills,
    settings_template,
)
from quad.client.claude_code.skills import (
    install_skills,
    list_installed_skills,
    uninstall_skills,
)


# ─── settings.py ──────────────────────────────────────────────────────────


class TestSettingsTemplate:
    def test_default_shape(self) -> None:
        s = settings_template()
        assert "permissions" in s
        assert "mcpServers" in s
        assert "quad" in s["mcpServers"]
        assert s["mcpServers"]["quad"]["command"] == "python"
        assert s["mcpServers"]["quad"]["args"] == ["-m", "quad.mcp.server"]

    def test_default_permissions_includes_all_5_tools(self) -> None:
        s = settings_template()
        allowed = s["permissions"]["allow"]
        for tool in (
            "mcp__quad__hardware_detect",
            "mcp__quad__convert_model",
            "mcp__quad__profile_workload",
            "mcp__quad__orchestrate_workload",
            "mcp__quad__generate_code",
        ):
            assert tool in allowed

    def test_adapter_mode_set_in_env(self) -> None:
        s = settings_template(adapter_mode="real")
        assert s["mcpServers"]["quad"]["env"]["QUAD_ADAPTER_MODE"] == "real"

    def test_legacy_args_supported(self) -> None:
        # Tests that callers can override args to use the legacy entrypoint
        s = settings_template(server_args=["-m", "quad.server.main"])
        assert s["mcpServers"]["quad"]["args"] == ["-m", "quad.server.main"]


class TestBuildSettings:
    def test_overrides_merge_deep(self) -> None:
        s = build_settings(overrides={"mcpServers": {"other": {"command": "node"}}})
        assert "quad" in s["mcpServers"]  # original preserved
        assert s["mcpServers"]["other"]["command"] == "node"  # overlay added

    def test_overrides_replace_scalar(self) -> None:
        s = build_settings(overrides={"permissions": {"allow": ["custom"]}})
        # Deep merge replaces leaf values, so allow becomes ["custom"]
        assert s["permissions"]["allow"] == ["custom"]


# ─── skills.py ────────────────────────────────────────────────────────────


class TestSkills:
    def test_bundled_skills_dir_exists(self) -> None:
        d = bundled_skills_dir()
        assert d.is_dir()
        assert any(d.glob("*.md"))

    def test_list_bundled_skills_returns_at_least_5(self) -> None:
        skills = list_bundled_skills()
        # We bundle 11 skills as of v0.4.0
        assert len(skills) >= 5
        # Names start with quad- prefix
        assert all(s.name.startswith("quad-") for s in skills)
        assert all(s.suffix == ".md" for s in skills)

    def test_install_skills_writes_files(self, tmp_path: Path) -> None:
        target = tmp_path / "skills"
        written, skipped = install_skills(target)
        assert len(written) > 0
        assert len(skipped) == 0
        for f in written:
            assert f.is_file()
            assert f.parent == target

    def test_install_skills_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "skills"
        first_written, _ = install_skills(target)
        second_written, second_skipped = install_skills(target)
        assert len(second_written) == 0
        assert len(second_skipped) == len(first_written)

    def test_install_skills_force_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "skills"
        install_skills(target)
        # Mutate one file so we can detect overwrite
        modified = next(target.glob("*.md"))
        modified.write_text("USER MODIFIED")
        # With force=True, the bundled content should come back
        written, skipped = install_skills(target, force=True)
        assert len(written) > 0
        assert len(skipped) == 0
        assert "USER MODIFIED" not in modified.read_text()

    def test_uninstall_removes_only_bundled(self, tmp_path: Path) -> None:
        target = tmp_path / "skills"
        install_skills(target)
        # Add a user skill
        user_skill = target / "my-custom-skill.md"
        user_skill.write_text("custom content")

        removed = uninstall_skills(target)
        # Bundled removed
        assert len(removed) > 0
        # User skill survived
        assert user_skill.is_file()


# ─── provisioner.py ───────────────────────────────────────────────────────


class TestClaudeCodeProvisioner:
    def test_install_creates_files(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        result = prov.install(tmp_path)
        assert result.client == "claude_code"
        assert (tmp_path / ".claude" / "settings.json").is_file()
        assert (tmp_path / ".claude" / "skills").is_dir()
        # settings.json is well-formed JSON
        content = (tmp_path / ".claude" / "settings.json").read_text()
        parsed = json.loads(content)
        assert "mcpServers" in parsed

    def test_install_idempotent_without_force(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        prov.install(tmp_path)
        # Mutate settings.json
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.write_text('{"mine": true}')
        result = prov.install(tmp_path)
        # Without force, settings was skipped
        assert any("settings.json" in p for p in result.files_skipped)
        # And the user's content was preserved
        assert json.loads(settings_path.read_text()) == {"mine": True}

    def test_install_force_overwrites_settings(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        settings_path = tmp_path / ".claude" / "settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text('{"mine": true}')
        prov.install(tmp_path, force=True)
        parsed = json.loads(settings_path.read_text())
        assert "mcpServers" in parsed  # template took over

    def test_status_reports_complete_install(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        prov.install(tmp_path)
        s = prov.status(tmp_path)
        assert s["settings_exists"] is True
        assert s["skills_dir_exists"] is True
        assert s["installed_skill_count"] == s["bundled_skill_count"]
        assert s["missing_skills"] == []

    def test_status_reports_missing_install(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        s = prov.status(tmp_path)
        assert s["settings_exists"] is False
        assert s["skills_dir_exists"] is False
        assert s["installed_skill_count"] == 0
        # All bundled skills appear as missing
        assert len(s["missing_skills"]) == s["bundled_skill_count"]

    def test_status_detects_user_added_skills(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        prov.install(tmp_path)
        # Add a user skill
        (tmp_path / ".claude" / "skills" / "my-skill.md").write_text("hi")
        s = prov.status(tmp_path)
        assert "my-skill.md" in s["extra_user_skills"]

    def test_uninstall_preserves_user_skills(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        prov.install(tmp_path)
        user_skill = tmp_path / ".claude" / "skills" / "user.md"
        user_skill.write_text("user content")

        prov.uninstall(tmp_path)
        # Bundled skills gone
        s = prov.status(tmp_path)
        assert s["installed_skill_count"] == 1  # only user.md left
        assert user_skill.is_file()

    def test_uninstall_keeps_settings(self, tmp_path: Path) -> None:
        """Uninstall doesn't auto-remove settings.json — user may have customised it."""
        prov = ClaudeCodeProvisioner()
        prov.install(tmp_path)
        prov.uninstall(tmp_path)
        assert (tmp_path / ".claude" / "settings.json").is_file()

    def test_render_settings_preview(self, tmp_path: Path) -> None:
        prov = ClaudeCodeProvisioner()
        preview = prov.render_settings_preview(adapter_mode="real")
        parsed = json.loads(preview)
        assert parsed["mcpServers"]["quad"]["env"]["QUAD_ADAPTER_MODE"] == "real"


# ─── factory ──────────────────────────────────────────────────────────────


class TestGetProvisioner:
    def test_returns_claude_code_provisioner(self) -> None:
        p = get_provisioner("claude_code")
        assert isinstance(p, ClaudeCodeProvisioner)
        assert isinstance(p, MCPClientProvisioner)

    def test_unknown_client_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown MCP client"):
            get_provisioner("cursor")


# ─── result dataclass ─────────────────────────────────────────────────────


class TestProvisionResult:
    def test_to_dict(self) -> None:
        r = ProvisionResult(client="x", files_written=["a"], notes=["n"])
        d = r.to_dict()
        assert d["client"] == "x"
        assert d["files_written"] == ["a"]
        assert d["notes"] == ["n"]
