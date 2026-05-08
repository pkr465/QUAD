"""Manage Claude Code skill files (``.claude/skills/*.md``).

Skills are bundled inside the Python package (under
``quad/client/claude_code/skills_content/``) so a ``pip install``
includes them. ``install_skills`` copies them to the user's
``.claude/skills/`` directory.
"""

from __future__ import annotations

import importlib.resources
import shutil
from pathlib import Path


def bundled_skills_dir() -> Path:
    """Locate the bundled skills directory inside the installed package.

    Falls back to the source-tree location when running in editable mode
    or directly from a checkout.
    """
    # Try the package-internal directory first (set by hatch's force-include
    # equivalent at install time)
    try:
        with importlib.resources.path(
            "quad.client.claude_code", "skills_content"
        ) as p:
            if p.is_dir():
                return p
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    # Source-tree fallback
    here = Path(__file__).resolve().parent
    candidate = here / "skills_content"
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        "Could not locate the bundled skills directory. Expected at "
        "quad/client/claude_code/skills_content/. If installed via pip, "
        "ensure the package was built with the skills_content directory "
        "included."
    )


def list_bundled_skills() -> list[Path]:
    """Return the list of bundled ``*.md`` skill files."""
    return sorted(bundled_skills_dir().glob("*.md"))


def install_skills(
    target_dir: Path,
    *,
    force: bool = False,
) -> tuple[list[Path], list[Path]]:
    """Copy bundled skill files into ``target_dir``.

    Args:
        target_dir: typically ``<project_root>/.claude/skills/``
        force: overwrite existing files of the same name. When False,
            existing files are left untouched and listed in the
            ``skipped`` return.

    Returns:
        ``(written, skipped)`` — both lists of Path objects in target_dir.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[Path] = []
    for src in list_bundled_skills():
        dst = target_dir / src.name
        if dst.exists() and not force:
            skipped.append(dst)
            continue
        shutil.copy2(src, dst)
        written.append(dst)
    return written, skipped


def uninstall_skills(target_dir: Path) -> list[Path]:
    """Remove the bundled skill files from ``target_dir``.

    Only removes files that exist in the bundled set — leaves
    user-added skills alone.
    """
    if not target_dir.is_dir():
        return []
    bundled_names = {p.name for p in list_bundled_skills()}
    removed: list[Path] = []
    for f in target_dir.glob("*.md"):
        if f.name in bundled_names:
            f.unlink()
            removed.append(f)
    return removed


def list_installed_skills(target_dir: Path) -> list[Path]:
    """Return all ``*.md`` files in the user's ``.claude/skills/``."""
    if not target_dir.is_dir():
        return []
    return sorted(target_dir.glob("*.md"))
