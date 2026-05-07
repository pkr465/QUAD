"""Tests for the QAIRT SDK auto-discovery + install helper."""

from __future__ import annotations

import io
import os
import platform
import tarfile
import zipfile
from pathlib import Path

import pytest

from quad.sdk_manager import (
    QAIRT_PRODUCT_URL,
    SNPE_PRODUCT_URL,
    InstallResult,
    SDKInfo,
    apply_to_environment,
    auto_download_enabled,
    discover_sdks,
    install_archive,
    missing_sdk_message,
    resolve_sdk_root,
    startup_resolve_and_log,
    write_state_file,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all SDK env vars + auto-download flag for hermetic tests."""
    for var in (
        "QAIRT_SDK_ROOT",
        "QNN_SDK_ROOT",
        "SNPE_ROOT",
        "QUAD_SDK_AUTO_DOWNLOAD",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_fake_sdk(root: Path, flavor: str = "qairt", arch: str = "x86_64-linux-clang") -> Path:
    """Create a directory that looks like a QAIRT/SNPE SDK install."""
    bin_dir = root / "bin" / arch
    bin_dir.mkdir(parents=True)
    if flavor == "qairt":
        # Use platform-appropriate executable name
        exe_name = "qairt-converter.exe" if os.name == "nt" else "qairt-converter"
    else:
        exe_name = "snpe-net-run.exe" if os.name == "nt" else "snpe-net-run"
    (bin_dir / exe_name).write_text("#!/bin/sh\necho stub")
    (root / "lib").mkdir()
    return root


# ─── Discovery ────────────────────────────────────────────────────────────────


class TestDiscoverSDKs:
    def test_no_sdk_returns_empty(self, clean_env, tmp_path, monkeypatch) -> None:
        # Hermetic: scan only inside tmp_path
        monkeypatch.setattr(
            "quad.sdk_manager.DEFAULT_SCAN_PATHS", (str(tmp_path / "nope"),)
        )
        sdks = discover_sdks(project_root=tmp_path)
        assert sdks == []

    def test_finds_project_local_sdk(self, clean_env, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        sdk_dir = tmp_path / "sdks" / "qairt-2.45.0.260326"
        _make_fake_sdk(sdk_dir, flavor="qairt")
        sdks = discover_sdks(project_root=tmp_path)
        assert len(sdks) == 1
        assert sdks[0].flavor == "qairt"
        assert sdks[0].version == "2.45.0.260326"
        assert sdks[0].source.startswith("project:")
        assert sdks[0].has_qairt_converter is True

    def test_env_var_takes_priority(self, clean_env, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        env_sdk = tmp_path / "envsdk" / "qairt-2.43.0.250827"
        _make_fake_sdk(env_sdk, flavor="qairt")
        proj_sdk = tmp_path / "sdks" / "qairt-2.45.0.260326"
        _make_fake_sdk(proj_sdk, flavor="qairt")

        monkeypatch.setenv("QAIRT_SDK_ROOT", str(env_sdk))
        sdks = discover_sdks(project_root=tmp_path)
        assert sdks[0].source.startswith("env:")
        assert sdks[0].root == str(env_sdk.resolve())

    def test_dedups_same_path_via_multiple_routes(self, clean_env, tmp_path, monkeypatch) -> None:
        sdk = tmp_path / "sdks" / "qairt-2.45.0"
        _make_fake_sdk(sdk, flavor="qairt")
        # Point env var AND scan path at the same location
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(sdk))
        monkeypatch.setattr(
            "quad.sdk_manager.DEFAULT_SCAN_PATHS", (str(tmp_path / "sdks"),)
        )
        sdks = discover_sdks(project_root=tmp_path)
        assert len(sdks) == 1  # not 2

    def test_finds_snpe_flavor(self, clean_env, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        sdk_dir = tmp_path / "sdks" / "snpe-2.40.0"
        _make_fake_sdk(sdk_dir, flavor="snpe")
        sdks = discover_sdks(project_root=tmp_path)
        assert len(sdks) == 1
        assert sdks[0].flavor == "snpe"
        assert sdks[0].has_snpe_net_run is True
        assert sdks[0].has_qairt_converter is False

    def test_extra_paths_from_config(self, clean_env, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        cfg_sdk = tmp_path / "custom" / "qairt-2.45.0"
        _make_fake_sdk(cfg_sdk, flavor="qairt")
        sdks = discover_sdks(
            extra_paths=[str(tmp_path / "custom")], project_root=tmp_path
        )
        assert len(sdks) == 1
        assert sdks[0].source == "config"

    def test_resolve_returns_first(self, clean_env, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        info = resolve_sdk_root(project_root=tmp_path)
        assert info is None
        sdk_dir = tmp_path / "sdks" / "qairt-2.45.0"
        _make_fake_sdk(sdk_dir, flavor="qairt")
        info = resolve_sdk_root(project_root=tmp_path)
        assert info is not None
        assert info.flavor == "qairt"

    def test_ignores_non_sdk_directory(self, clean_env, tmp_path, monkeypatch) -> None:
        """A directory named 'qairt-X.Y' but missing bin/<arch>/ is not an SDK."""
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        bogus = tmp_path / "sdks" / "qairt-2.45.0"
        bogus.mkdir(parents=True)
        (bogus / "README.md").write_text("not actually an SDK")
        sdks = discover_sdks(project_root=tmp_path)
        assert sdks == []


# ─── Install (from archive) ───────────────────────────────────────────────────


def _build_zip(archive: Path, root_in_zip: str = "qairt-2.45.0.260326") -> None:
    """Build a zip that mimics a QAIRT release archive."""
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{root_in_zip}/bin/x86_64-linux-clang/qairt-converter", "stub")
        zf.writestr(f"{root_in_zip}/bin/x86_64-linux-clang/qairt-quantizer", "stub")
        zf.writestr(f"{root_in_zip}/lib/x86_64-linux-clang/libQnnHtp.so", "")
        zf.writestr(f"{root_in_zip}/README.md", "QAIRT 2.45")


def _build_zip_flat(archive: Path) -> None:
    """Build a zip without a top-level wrapper directory."""
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("bin/x86_64-linux-clang/qairt-converter", "stub")
        zf.writestr("lib/x86_64-linux-clang/libQnnHtp.so", "")


def _build_tgz(archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tf:
        for path, content in [
            ("qairt-2.45.0/bin/x86_64-linux-clang/qairt-converter", b"stub"),
            ("qairt-2.45.0/lib/foo.so", b""),
        ]:
            data = io.BytesIO(content)
            ti = tarfile.TarInfo(name=path)
            ti.size = len(content)
            tf.addfile(ti, data)


class TestInstallArchive:
    def test_install_zip(self, tmp_path) -> None:
        archive = tmp_path / "qairt-2.45.0.260326.zip"
        _build_zip(archive)
        result = install_archive(archive, project_root=tmp_path)
        assert isinstance(result, InstallResult)
        assert result.flavor == "qairt"
        assert result.version == "2.45.0.260326"
        assert Path(result.target_dir).is_dir()
        # Top-level dir should be hoisted away
        assert (Path(result.target_dir) / "bin").is_dir()

    def test_install_tgz(self, tmp_path) -> None:
        archive = tmp_path / "qairt-2.45.0.tar.gz"
        _build_tgz(archive)
        result = install_archive(archive, project_root=tmp_path)
        assert result.flavor == "qairt"
        assert (Path(result.target_dir) / "bin").is_dir()

    def test_install_flat_zip_no_wrapper(self, tmp_path) -> None:
        archive = tmp_path / "qairt-2.45.0.zip"
        _build_zip_flat(archive)
        result = install_archive(archive, project_root=tmp_path)
        assert (Path(result.target_dir) / "bin" / "x86_64-linux-clang" / "qairt-converter").exists()

    def test_install_missing_archive(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            install_archive(tmp_path / "nope.zip", project_root=tmp_path)

    def test_install_existing_target_no_overwrite(self, tmp_path) -> None:
        archive = tmp_path / "qairt-2.45.0.zip"
        _build_zip(archive)
        install_archive(archive, project_root=tmp_path)
        with pytest.raises(FileExistsError):
            install_archive(archive, project_root=tmp_path)

    def test_install_existing_target_with_overwrite(self, tmp_path) -> None:
        archive = tmp_path / "qairt-2.45.0.zip"
        _build_zip(archive)
        install_archive(archive, project_root=tmp_path)
        # Second call with overwrite should succeed
        result = install_archive(archive, project_root=tmp_path, overwrite=True)
        assert (Path(result.target_dir) / "bin").is_dir()

    def test_install_unrecognised_format(self, tmp_path) -> None:
        archive = tmp_path / "qairt-2.45.0.7z"
        archive.write_bytes(b"\x37\x7a")  # 7z magic, but we don't support it
        with pytest.raises(ValueError, match="Unsupported"):
            install_archive(archive, project_root=tmp_path)

    def test_install_rejects_zipslip(self, tmp_path) -> None:
        """Archive with '../etc/passwd' should be rejected, not extracted."""
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("../etc/passwd", "rooted")
        with pytest.raises(ValueError, match="unsafe path"):
            install_archive(archive, project_root=tmp_path)

    def test_install_rejects_non_sdk_archive(self, tmp_path) -> None:
        """Archive without bin/<arch>/qairt-converter should be rejected."""
        archive = tmp_path / "random.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("docs/README.md", "not an SDK")
        with pytest.raises(ValueError, match="do not look like"):
            install_archive(archive, project_root=tmp_path)

    def test_install_then_discover(self, clean_env, tmp_path, monkeypatch) -> None:
        """End-to-end: install + discover finds the same SDK."""
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        archive = tmp_path / "qairt-2.45.0.260326.zip"
        _build_zip(archive)
        install_archive(archive, project_root=tmp_path)
        sdks = discover_sdks(project_root=tmp_path)
        assert len(sdks) == 1
        assert sdks[0].version == "2.45.0.260326"


# ─── Environment / state ──────────────────────────────────────────────────────


class TestEnvironment:
    def test_apply_sets_env_vars(self, clean_env) -> None:
        info = SDKInfo(
            root="/tmp/qairt",
            version="2.45.0",
            flavor="qairt",
            source="test",
            bin_dir="/tmp/qairt/bin/linux",
        )
        apply_to_environment(info)
        assert os.environ["QAIRT_SDK_ROOT"] == "/tmp/qairt"
        assert os.environ["QNN_SDK_ROOT"] == "/tmp/qairt"
        assert os.environ["SNPE_ROOT"] == "/tmp/qairt"

    def test_apply_does_not_override_existing(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/already/set")
        info = SDKInfo(root="/new/path", version="x", flavor="qairt", source="t")
        apply_to_environment(info)
        # setdefault semantics — existing wins
        assert os.environ["QAIRT_SDK_ROOT"] == "/already/set"

    def test_apply_prepends_bin_to_path(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("PATH", "/usr/bin")
        info = SDKInfo(
            root="/x", version="x", flavor="qairt", source="t", bin_dir="/x/bin/linux"
        )
        apply_to_environment(info)
        sep = ";" if os.name == "nt" else ":"
        assert os.environ["PATH"].startswith(f"/x/bin/linux{sep}")

    def test_write_state_file(self, tmp_path) -> None:
        info = SDKInfo(root="/x", version="2.45.0", flavor="qairt", source="t")
        path = write_state_file(info, project_root=tmp_path)
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert data["resolved"]["version"] == "2.45.0"

    def test_write_state_file_missing_sdk(self, tmp_path) -> None:
        path = write_state_file(None, project_root=tmp_path)
        import json

        data = json.loads(path.read_text())
        assert data["resolved"] is None
        assert data["qairt_url"] == QAIRT_PRODUCT_URL


# ─── Messages / startup hook ──────────────────────────────────────────────────


class TestMessaging:
    def test_missing_sdk_message_includes_urls(self) -> None:
        msg = missing_sdk_message()
        assert QAIRT_PRODUCT_URL in msg
        assert SNPE_PRODUCT_URL in msg
        assert "quad sdk install" in msg

    def test_missing_sdk_message_includes_reason(self) -> None:
        msg = missing_sdk_message("custom reason here")
        assert "custom reason here" in msg

    def test_auto_download_default_off(self, clean_env) -> None:
        assert auto_download_enabled() is False

    def test_auto_download_env_toggle(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv("QUAD_SDK_AUTO_DOWNLOAD", "1")
        assert auto_download_enabled() is True
        monkeypatch.setenv("QUAD_SDK_AUTO_DOWNLOAD", "yes")
        assert auto_download_enabled() is True
        monkeypatch.setenv("QUAD_SDK_AUTO_DOWNLOAD", "no")
        assert auto_download_enabled() is False


class TestStartupHook:
    def test_startup_no_sdk_returns_none(self, clean_env, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        info = startup_resolve_and_log(project_root=tmp_path)
        assert info is None
        # State file should still be written
        assert (tmp_path / ".quad" / "sdk.json").exists()

    def test_startup_with_sdk_sets_env(self, clean_env, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("quad.sdk_manager.DEFAULT_SCAN_PATHS", ())
        sdk_dir = tmp_path / "sdks" / "qairt-2.45.0"
        _make_fake_sdk(sdk_dir, flavor="qairt")
        info = startup_resolve_and_log(project_root=tmp_path)
        assert info is not None
        assert info.flavor == "qairt"
        # Env var should now be set
        assert os.environ["QAIRT_SDK_ROOT"] == str(sdk_dir.resolve())
