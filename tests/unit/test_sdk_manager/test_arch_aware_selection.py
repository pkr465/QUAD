"""Tests for the arch-aware bin_dir selection and flavor detection
(Sprint 1 P0-2, P0-3, P1-2 fixes).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quad.sdk_manager import (
    _looks_like_sdk_root,
    _populate_sdkinfo,
    _select_bin_dir,
    _version_from_dir_name,
    apply_to_environment,
    host_arch_label,
    list_all_bin_dirs,
    rank_bin_subdir,
    SDKInfo,
)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


@pytest.fixture
def fake_qairt(tmp_path: Path) -> Path:
    """A QAIRT 2.46-shaped install: converters in arm64x/x86_64,
    runtime tools in aarch64-windows-msvc."""
    root = tmp_path / "qairt-2.46.0.260424"
    bin_root = root / "bin"
    # Native ARM64 runtime tools
    _touch(bin_root / "aarch64-windows-msvc" / "snpe-net-run.exe")
    _touch(bin_root / "aarch64-windows-msvc" / "qnn-platform-validator.exe")
    # x86_64: full kit (converters + runtime)
    _touch(bin_root / "x86_64-windows-msvc" / "qairt-converter")
    _touch(bin_root / "x86_64-windows-msvc" / "snpe-net-run.exe")
    # arm64x: converters only
    _touch(bin_root / "arm64x-windows-msvc" / "qairt-converter")
    return root


class TestVersionRegex:
    def test_canonical_qairt_dirname(self) -> None:
        assert _version_from_dir_name("qairt-2.45.0.260326") == ("qairt", "2.45.0.260326")

    def test_underscore_variant(self) -> None:
        assert _version_from_dir_name("qairt_2.45") == ("qairt", "2.45")

    def test_snpe_dirname(self) -> None:
        assert _version_from_dir_name("snpe-2.45.0") == ("snpe", "2.45.0")

    def test_v_prefix_archive_name(self) -> None:
        # Qualcomm developer-portal naming
        assert _version_from_dir_name("v2.46.0.260424") == (None, "2.46.0.260424")

    def test_bare_version_dirname(self) -> None:
        # Inner directory after extracting an archive without rewrap
        assert _version_from_dir_name("2.46.0.260424") == (None, "2.46.0.260424")

    def test_garbage_returns_none(self) -> None:
        assert _version_from_dir_name("not-an-sdk") == (None, None)


class TestLooksLikeSdkRoot:
    def test_qairt_wins_when_both_present(self, fake_qairt: Path) -> None:
        # A QAIRT install contains both markers; qairt must win.
        assert _looks_like_sdk_root(fake_qairt) == "qairt"

    def test_snpe_when_only_snpe_present(self, tmp_path: Path) -> None:
        _touch(tmp_path / "bin" / "aarch64-android" / "snpe-net-run")
        assert _looks_like_sdk_root(tmp_path) == "snpe"

    def test_returns_none_for_non_sdk(self, tmp_path: Path) -> None:
        # Has bin/ but no recognisable tools
        (tmp_path / "bin").mkdir()
        assert _looks_like_sdk_root(tmp_path) is None

    def test_returns_none_when_no_bin(self, tmp_path: Path) -> None:
        assert _looks_like_sdk_root(tmp_path) is None


class TestRankBinSubdir:
    def test_arm64_windows_prefers_aarch64(self) -> None:
        a = rank_bin_subdir("aarch64-windows-msvc",
                            host_platform="win32", host_arch="arm64")
        x = rank_bin_subdir("x86_64-windows-msvc",
                            host_platform="win32", host_arch="arm64")
        ax = rank_bin_subdir("arm64x-windows-msvc",
                             host_platform="win32", host_arch="arm64")
        assert a > ax > x > 0

    def test_x86_64_windows_prefers_x86_64(self) -> None:
        x = rank_bin_subdir("x86_64-windows-msvc",
                            host_platform="win32", host_arch="x86_64")
        a = rank_bin_subdir("aarch64-windows-msvc",
                            host_platform="win32", host_arch="x86_64")
        assert x > a > 0

    def test_unknown_returns_zero(self) -> None:
        assert rank_bin_subdir("not-a-known-arch") == 0


class TestSelectBinDir:
    def test_picks_native_arm64(self, fake_qairt: Path, monkeypatch) -> None:
        # Force the host to look like ARM64 Windows regardless of test runner
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("quad.sdk_manager.host_arch_label", lambda: "arm64")
        chosen = _select_bin_dir(fake_qairt / "bin")
        assert chosen.endswith("aarch64-windows-msvc"), chosen

    def test_falls_back_when_native_missing(self, tmp_path: Path,
                                            monkeypatch) -> None:
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("quad.sdk_manager.host_arch_label", lambda: "arm64")
        # Only x86_64 available
        _touch(tmp_path / "bin" / "x86_64-windows-msvc" / "qairt-converter")
        chosen = _select_bin_dir(tmp_path / "bin")
        assert chosen.endswith("x86_64-windows-msvc"), chosen


class TestPopulateSdkInfo:
    def test_qairt_install_classified_correctly(self, fake_qairt: Path,
                                                 monkeypatch) -> None:
        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setattr("quad.sdk_manager.host_arch_label", lambda: "arm64")
        info = _populate_sdkinfo(fake_qairt, source="test")
        assert info.flavor == "qairt"
        assert info.version == "2.46.0.260424"
        assert info.has_qairt_converter
        assert info.has_snpe_net_run
        assert info.bin_dir.endswith("aarch64-windows-msvc")


class TestApplyToEnvironment:
    def test_includes_all_bin_subdirs(self, fake_qairt: Path,
                                      monkeypatch) -> None:
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.delenv("QAIRT_SDK_ROOT", raising=False)
        monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
        monkeypatch.delenv("SNPE_ROOT", raising=False)
        info = SDKInfo(
            root=str(fake_qairt),
            version="2.46.0.260424",
            flavor="qairt",
            source="test",
            bin_dir=str(fake_qairt / "bin" / "aarch64-windows-msvc"),
        )
        apply_to_environment(info)
        # All three live bin subdirs must be on PATH
        import os
        path = os.environ["PATH"]
        for sub in ("aarch64-windows-msvc", "arm64x-windows-msvc",
                    "x86_64-windows-msvc"):
            assert sub in path, f"missing {sub} in PATH"


class TestParsePlatformValidator:
    def test_parses_supported_backends(self) -> None:
        from quad.adapters.qairt_adapter import _parse_platform_validator
        sample = (
            "Backend: cpu        is supported\n"
            "Backend: gpu        is supported\n"
            "Backend: dsp        is supported (skel: v75)\n"
            "Chipset: SM8750\n"
        )
        out = _parse_platform_validator(sample)
        assert "cpu" in out["runtimes"]
        assert "gpu" in out["runtimes"]
        assert "npu" in out["runtimes"]
        assert out["chipset"] == "SM8750"

    def test_returns_empty_on_no_match(self) -> None:
        from quad.adapters.qairt_adapter import _parse_platform_validator
        out = _parse_platform_validator("totally unrelated text")
        assert out["runtimes"] == []
        assert out["chipset"] is None


class TestParseLatency:
    def test_total_inference_time_match(self) -> None:
        from quad.adapters.qairt_adapter import QAIRTAdapter
        a = object.__new__(QAIRTAdapter)
        ms = QAIRTAdapter._parse_latency(a, "Total Inference Time: 12.34 ms")
        assert ms == pytest.approx(12.34)

    def test_no_match_returns_zero_not_lie(self) -> None:
        # The previous implementation returned 5.0 ms on parser failure,
        # silently lying about what was measured. Now we return 0.0 so
        # callers can detect "unknown".
        from quad.adapters.qairt_adapter import QAIRTAdapter
        a = object.__new__(QAIRTAdapter)
        ms = QAIRTAdapter._parse_latency(a, "no timing in here")
        assert ms == 0.0
