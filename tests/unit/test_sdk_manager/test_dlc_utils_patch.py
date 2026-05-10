"""Tests for sdk_manager.dlc_utils_patch — QAIRT __init__.py rewriter."""
from __future__ import annotations

from pathlib import Path

import pytest

from quad.sdk_patch import (
    PATCH_SENTINEL,
    patch_sdk,
)


# Real-world snippet of QAIRT 2.46's __init__.py (Windows branch only).
INIT_PY = '''# header

import os
import sys
import platform

if platform.system() == "Linux":
    if platform.machine() == "x86_64":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'linux-x86_64'))
    else:
        raise NotImplementedError("posix")
elif platform.system() == "Windows":
    if "AMD64" in platform.processor() or "Intel64" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-x86_64'))
    elif "ARMv8" in platform.processor():
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'windows-arm64ec'))
    else:
        cpu_isa = platform.processor().split()[0]
        raise NotImplementedError('Unsupported OS Platform: {} {}'.format(platform.system(), cpu_isa))
else:
    raise NotImplementedError("not supported")

# trailing content
try:
    from . import libDlModelToolsPy as modeltools
except ImportError:
    raise
'''


def _make_fake_sdk(root: Path) -> Path:
    init_dir = root / "lib" / "python" / "qti" / "aisw" / "dlc_utils"
    init_dir.mkdir(parents=True, exist_ok=True)
    init_path = init_dir / "__init__.py"
    init_path.write_text(INIT_PY, encoding="utf-8")
    return init_path


def test_patch_inserts_sentinel(tmp_path):
    init_path = _make_fake_sdk(tmp_path)
    summary = patch_sdk(tmp_path)
    assert summary["patched"] and str(init_path) in summary["patched"]
    assert not summary["skipped"]
    new_text = init_path.read_text(encoding="utf-8")
    assert PATCH_SENTINEL in new_text


def test_patch_is_idempotent(tmp_path):
    init_path = _make_fake_sdk(tmp_path)
    first = patch_sdk(tmp_path)
    second = patch_sdk(tmp_path)
    assert first["patched"]
    assert second["skipped"] and not second["patched"]
    # Sentinel appears exactly once after two patch runs.
    txt = init_path.read_text(encoding="utf-8")
    assert txt.count(PATCH_SENTINEL) == 1


def test_patch_preserves_linux_branch(tmp_path):
    init_path = _make_fake_sdk(tmp_path)
    patch_sdk(tmp_path)
    new_text = init_path.read_text(encoding="utf-8")
    assert 'platform.system() == "Linux":' in new_text
    assert "linux-x86_64" in new_text


def test_patched_init_picks_x86_for_amd64_python(tmp_path):
    """Smoke-test the patched init: simulate execution with sysconfig
    returning win-amd64 and verify it inserts windows-x86_64 into sys.path."""
    init_path = _make_fake_sdk(tmp_path)
    patch_sdk(tmp_path)
    src = init_path.read_text(encoding="utf-8")

    # Strip the trailing import-raise block (we don't want to actually
    # import libDlModelToolsPy in the test).
    src = src.split("# trailing content")[0]

    # Patch the module's view of sysconfig + platform without disturbing globals.
    class _FakeSysconfig:
        @staticmethod
        def get_platform():
            return "win-amd64"

    class _FakePlatform:
        @staticmethod
        def system():
            return "Windows"
        @staticmethod
        def machine():
            return "AMD64"
        @staticmethod
        def processor():
            return "ARMv8 (64-bit) Family 8 Model 1"

    fake_globals = {
        "__file__": str(init_path),
        "sysconfig": _FakeSysconfig,
        "platform": _FakePlatform,
        "os": __import__("os"),
        "sys": __import__("sys"),
    }
    # Strip the original `import platform` / `import sysconfig` so our
    # fakes survive.
    # Only strip the top-level imports we're shadowing via fake_globals;
    # keep the patched code's inline `import sysconfig as _qp_sysconfig`.
    src_no_imports = "\n".join(
        ln for ln in src.splitlines()
        if not ln.startswith("import platform")
        and not ln.startswith("import sys")
        and not ln.startswith("import os")
    )
    saved_path = list(__import__("sys").path)
    try:
        exec(src_no_imports, fake_globals)
        # The Windows-amd64 branch should have prepended windows-x86_64.
        first = fake_globals["sys"].path[0]
        assert first.endswith("windows-x86_64")
    finally:
        __import__("sys").path[:] = saved_path


def test_patched_init_picks_arm64_for_native_arm64_python(tmp_path):
    init_path = _make_fake_sdk(tmp_path)
    patch_sdk(tmp_path)
    src = init_path.read_text(encoding="utf-8").split("# trailing content")[0]

    class _FakeSysconfig:
        @staticmethod
        def get_platform():
            return "win-arm64"
    class _FakePlatform:
        @staticmethod
        def system():
            return "Windows"
        @staticmethod
        def machine():
            return "ARM64"
        @staticmethod
        def processor():
            return "ARMv8 (64-bit)"

    fake_globals = {
        "__file__": str(init_path),
        "sysconfig": _FakeSysconfig,
        "platform": _FakePlatform,
        "os": __import__("os"),
        "sys": __import__("sys"),
    }
    # Only strip the top-level imports we're shadowing via fake_globals;
    # keep the patched code's inline `import sysconfig as _qp_sysconfig`.
    src_no_imports = "\n".join(
        ln for ln in src.splitlines()
        if not ln.startswith("import platform")
        and not ln.startswith("import sys")
        and not ln.startswith("import os")
    )
    saved_path = list(__import__("sys").path)
    try:
        exec(src_no_imports, fake_globals)
        first = fake_globals["sys"].path[0]
        assert first.endswith("windows-arm64ec")
    finally:
        __import__("sys").path[:] = saved_path


def test_patch_creates_backup(tmp_path):
    init_path = _make_fake_sdk(tmp_path)
    original = init_path.read_text(encoding="utf-8")
    patch_sdk(tmp_path)
    backup = init_path.with_suffix(init_path.suffix + ".quad-bak")
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == original


def test_patch_handles_missing_sdk_root(tmp_path):
    summary = patch_sdk(tmp_path)
    assert summary["not_found"]
    assert not summary["patched"]
