"""Concrete :class:`MCPClientProvisioner` for Claude Code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quad.client import MCPClientProvisioner, ProvisionResult
from quad.client.claude_code.settings import (
    DEFAULT_PERMISSIONS,
    build_settings,
    render_to_json,
    write_settings,
)
from quad.client.claude_code.skills import (
    bundled_skills_dir,
    install_skills,
    list_bundled_skills,
    list_installed_skills,
    uninstall_skills,
)

CLAUDE_DIR_NAME = ".claude"
SETTINGS_FILE = "settings.json"
SKILLS_DIR = "skills"


class ClaudeCodeProvisioner(MCPClientProvisioner):
    """Install + manage Claude Code's QUAD MCP integration.

    Layout under ``project_root``::

        .claude/
        ├── settings.json   # MCP server discovery + tool permissions
        └── skills/
            ├── quad-quickstart.md
            ├── quad-detect.md
            └── …

    ``install`` is idempotent: re-running on an already-provisioned
    project skips files that exist, unless ``force=True``.
    """

    name = "claude_code"

    def install(
        self,
        project_root: Path,
        *,
        force: bool = False,
        adapter_mode: str = "mock",
        permissions: tuple[str, ...] | None = None,
    ) -> ProvisionResult:
        claude_dir = project_root / CLAUDE_DIR_NAME
        settings_path = claude_dir / SETTINGS_FILE
        skills_target = claude_dir / SKILLS_DIR

        result = ProvisionResult(
            client=self.name,
            settings_path=str(settings_path),
            skills_dir=str(skills_target),
        )

        # 1. settings.json
        settings_written = write_settings(
            settings_path,
            force=force,
            adapter_mode=adapter_mode,
            permissions=permissions or DEFAULT_PERMISSIONS,
        )
        if settings_written:
            result.files_written.append(str(settings_path))
        else:
            result.files_skipped.append(str(settings_path))
            result.notes.append(
                f"settings.json already exists — pass force=True to overwrite. "
                f"Read at: {settings_path}"
            )

        # 2. skills
        try:
            written, skipped = install_skills(skills_target, force=force)
            result.files_written.extend(str(p) for p in written)
            result.files_skipped.extend(str(p) for p in skipped)
            if skipped and not force:
                result.notes.append(
                    f"{len(skipped)} skill file(s) already exist — pass force=True to overwrite."
                )
        except FileNotFoundError as e:
            result.notes.append(f"Skill installation failed: {e}")

        return result

    def uninstall(self, project_root: Path) -> ProvisionResult:
        claude_dir = project_root / CLAUDE_DIR_NAME
        settings_path = claude_dir / SETTINGS_FILE
        skills_target = claude_dir / SKILLS_DIR

        result = ProvisionResult(
            client=self.name,
            settings_path=str(settings_path),
            skills_dir=str(skills_target),
        )

        # Remove bundled skills (leave user-added ones alone)
        try:
            removed = uninstall_skills(skills_target)
            result.files_written = [str(p) for p in removed]  # 'removed' for the report
            result.notes.append(
                f"Removed {len(removed)} bundled skill file(s) from {skills_target}."
            )
        except Exception as e:
            result.notes.append(f"Skill uninstall failed: {e}")

        # Don't auto-delete settings.json — user may have customised it
        if settings_path.exists():
            result.notes.append(
                f"settings.json was NOT removed (may contain user customisations). "
                f"Delete manually if desired: {settings_path}"
            )

        return result

    def status(self, project_root: Path) -> dict[str, Any]:
        claude_dir = project_root / CLAUDE_DIR_NAME
        settings_path = claude_dir / SETTINGS_FILE
        skills_target = claude_dir / SKILLS_DIR

        installed_skills = list_installed_skills(skills_target)
        bundled = {p.name for p in list_bundled_skills()}
        installed_names = {p.name for p in installed_skills}
        missing = sorted(bundled - installed_names)
        extra = sorted(installed_names - bundled)

        return {
            "client": self.name,
            "settings_path": str(settings_path),
            "settings_exists": settings_path.is_file(),
            "skills_dir": str(skills_target),
            "skills_dir_exists": skills_target.is_dir(),
            "bundled_skill_count": len(bundled),
            "installed_skill_count": len(installed_skills),
            "missing_skills": missing,
            "extra_user_skills": extra,
        }

    def render_settings_preview(
        self,
        *,
        adapter_mode: str = "mock",
        permissions: tuple[str, ...] | None = None,
    ) -> str:
        """Return the JSON content that ``install()`` would write.

        Useful for ``quad client preview`` (see CLI), so users can
        inspect before committing.
        """
        return render_to_json(
            build_settings(
                adapter_mode=adapter_mode,
                permissions=permissions or DEFAULT_PERMISSIONS,
            )
        )
