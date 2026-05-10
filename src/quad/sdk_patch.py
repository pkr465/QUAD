"""Patch QAIRT's ``qti.aisw.dlc_utils.__init__`` so it picks the .pyd
that actually loads in the active Python.

Why:
    The stock ``__init__.py`` keys off ``platform.processor()``, which on
    Snapdragon X Elite always returns ``"ARMv8 (64-bit) Family 8 …"``.
    The code then inserts ``windows-arm64ec/`` into sys.path and tries
    to import the .pyd from there. That .pyd is ARM64EC-compiled and
    refuses to load into:

      * pure ARM64 Python   → WinError 193 (architecture mismatch)
      * x86_64 emulated Python → also fails because ARM64EC modules
                                  need an ARM64EC host process and
                                  Prism translates the wrong direction

    The working path on this hardware is x86_64 Python + the
    ``windows-x86_64/`` .pyd (built for native x86_64) + a VS 2022 C++
    runtime install. We need ``__init__.py`` to choose that branch
    based on what the running *Python* is, not what the *CPU* is.

The fix is one line: prefer ``sysconfig.get_platform()`` (Python-arch
truthful) over ``platform.processor()`` (CPU-truthful). The patcher is
idempotent — it embeds a sentinel comment, and on a re-run with the
sentinel present it returns immediately.

Both ``install.sh`` (via ``scripts/adapters/setup_qairt.sh``) and
``bootstrap.ps1`` Step 5 call this after QAIRT extraction so the user
never sees the original bug.
"""
from __future__ import annotations

import os
from pathlib import Path

PATCH_SENTINEL = "# QUAD-PATCH: prefer sysconfig.get_platform() over platform.processor()"

# The replacement block. We keep the original Linux branch intact and
# rewrite only the Windows branch. The new logic:
#   1. If sysconfig.get_platform() says win-amd64 → use windows-x86_64
#   2. Else if it says win-arm64 → use windows-arm64ec (the only ARM64
#      branch QAIRT 2.46 ships; works on native ARM64 Python only if
#      Microsoft's ARM64EC bridge is loadable — uncommon)
#   3. Else fall back to platform.processor()-based heuristic for
#      compatibility with older Python builds (e.g. on Linux ARM hosts).
_NEW_WINDOWS_BLOCK = '''elif platform.system() == "Windows":
    {sentinel}
    _qp_py_arch = sysconfig.get_platform().lower()
    if "amd64" in _qp_py_arch or "x86" in _qp_py_arch:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-x86_64'))
    elif "arm64" in _qp_py_arch:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-arm64ec'))
    elif "AMD64" in platform.processor() or "Intel64" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-x86_64'))
    elif "ARMv8" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-arm64ec'))
    else:
        cpu_isa = platform.processor().split()[0]
        raise NotImplementedError('Unsupported OS Platform: {{}} {{}}'.format(platform.system(), cpu_isa))
'''.format(sentinel=PATCH_SENTINEL)


def _dlc_utils_init_paths(sdk_root: Path) -> list[Path]:
    """Return every ``__init__.py`` that needs patching under one SDK root.

    QAIRT 2.46 ships the same file in two places:
        lib/python/qti/aisw/dlc_utils/__init__.py
        lib/python/qti/aisw/converters/common/__init__.py  (sometimes)
    """
    candidates = [
        sdk_root / "lib" / "python" / "qti" / "aisw" / "dlc_utils" / "__init__.py",
    ]
    return [p for p in candidates if p.exists()]


def patch_sdk(sdk_root: Path, *, dry_run: bool = False) -> dict:
    """Patch every applicable __init__.py under ``sdk_root``.

    Idempotent: re-running on an already-patched SDK is a no-op.

    Args:
        sdk_root: Path to an unpacked QAIRT install (e.g. ./sdks/qairt-2.46.0.260424)
        dry_run:  If True, report what would change without writing.

    Returns:
        Summary dict with ``patched`` (list[str] paths), ``skipped`` (already
        patched), and ``not_found`` (no __init__.py to patch under this root).
    """
    summary = {"patched": [], "skipped": [], "not_found": []}
    inits = _dlc_utils_init_paths(sdk_root)
    if not inits:
        summary["not_found"].append(str(sdk_root))
        return summary

    for init_path in inits:
        original = init_path.read_text(encoding="utf-8")
        if PATCH_SENTINEL in original:
            summary["skipped"].append(str(init_path))
            continue

        text = original

        # Inject `import sysconfig` after the `import platform` line so
        # the rewritten Windows branch can call sysconfig.get_platform()
        # without an inline import (which would shadow test fixtures).
        if "\nimport sysconfig" not in text:
            text = text.replace(
                "import platform",
                "import platform\nimport sysconfig  # QUAD-PATCH",
                1,
            )

        # Find the Windows branch — it always starts with `elif platform.system() == "Windows":`
        anchor = 'elif platform.system() == "Windows":'
        idx = text.find(anchor)
        if idx == -1:
            # Can't safely patch — skip.
            summary["skipped"].append(str(init_path) + ":no_anchor")
            continue

        # Find the matching `else:` that closes the Windows branch.
        # Search forward from the Windows line.
        else_idx = text.find("\nelse:", idx)
        if else_idx == -1:
            summary["skipped"].append(str(init_path) + ":no_terminator")
            continue
        # The replacement covers everything from the Windows-branch start
        # through the close of its own else: clause. We don't touch the
        # outer module-level else (that's the Linux-vs-Windows top-level
        # error). Find the next top-level statement boundary instead by
        # walking until we find the line that's de-dented to column 0
        # AND starts with 'else:' AND raises NotImplementedError.
        # In QAIRT 2.46's file the structure is:
        #     elif platform.system() == "Windows":
        #         if ...
        #         elif ...
        #         else:
        #             raise NotImplementedError(...)
        #     else:
        #         raise NotImplementedError(...)
        # So we want to replace from `elif platform.system()` up to (but
        # not including) the second top-level `else:`. That second
        # `else:` follows the Windows-block's own internal `raise`.
        # Strategy: find the first newline that re-establishes column 0
        # AND begins with 'else:'.
        scan = else_idx + 1
        replacement_end = None
        while True:
            next_nl = text.find("\nelse:", scan + 1)
            if next_nl == -1:
                break
            # Confirm this `else:` is at column 0 of its line
            line_start = next_nl + 1
            if text[line_start:line_start + 5] == "else:":
                replacement_end = next_nl
                break
            scan = next_nl + 1

        if replacement_end is None:
            # Fallback: use the first newline-else: we found and trust it.
            replacement_end = else_idx

        new_text = text[:idx] + _NEW_WINDOWS_BLOCK + text[replacement_end:]

        if dry_run:
            summary["patched"].append(str(init_path) + ":dry_run")
            continue

        # Write atomically with a backup beside the original. The backup
        # captures the pristine ORIGINAL (pre-import-injection) text so
        # a later restore is faithful.
        backup = init_path.with_suffix(init_path.suffix + ".quad-bak")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        init_path.write_text(new_text, encoding="utf-8")
        summary["patched"].append(str(init_path))

    return summary


def patch_active_sdk(dry_run: bool = False) -> dict:
    """Convenience wrapper — resolve QAIRT_SDK_ROOT and patch it."""
    sdk_root = os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("SNPE_ROOT")
    if not sdk_root:
        return {"patched": [], "skipped": [], "not_found": ["QAIRT_SDK_ROOT_not_set"]}
    return patch_sdk(Path(sdk_root), dry_run=dry_run)


__all__ = ["patch_sdk", "patch_active_sdk", "PATCH_SENTINEL"]
