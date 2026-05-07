"""Tests for Windows ARM64X support."""

from __future__ import annotations

import pytest

from quad.adapters.dsp_env import get_windows_arch_folder
from quad.platforms.windows import WindowsPlatform, WINDOWS_ARCH_FOLDERS


class TestWindowsArchFolders:
    def test_arm64x_maps_to_arm64x_folder(self) -> None:
        """SC8380XP ARM64X native binary uses arm64x-windows-msvc."""
        assert get_windows_arch_folder("arm64x") == "arm64x-windows-msvc"

    def test_arm64ec_maps_to_aarch64(self) -> None:
        """ARM64EC uses same SNPE libs as ARM64."""
        assert get_windows_arch_folder("arm64ec") == "aarch64-windows-msvc"

    def test_arm64_maps_to_aarch64(self) -> None:
        assert get_windows_arch_folder("arm64") == "aarch64-windows-msvc"
        assert get_windows_arch_folder("aarch64") == "aarch64-windows-msvc"

    def test_x64_maps_to_x86_64(self) -> None:
        """x86_64 app on ARM64X: link x86_64 SNPE.lib (ARM64X dll at runtime)."""
        assert get_windows_arch_folder("x64") == "x86_64-windows-msvc"
        assert get_windows_arch_folder("x86_64") == "x86_64-windows-msvc"
        assert get_windows_arch_folder("AMD64") == "x86_64-windows-msvc"

    def test_unknown_defaults_to_x86_64(self) -> None:
        assert get_windows_arch_folder("unknown") == "x86_64-windows-msvc"

    def test_case_insensitive(self) -> None:
        assert get_windows_arch_folder("ARM64X") == "arm64x-windows-msvc"
        assert get_windows_arch_folder("X64") == "x86_64-windows-msvc"


class TestWindowsPlatformARM64X:
    def test_all_expected_arch_folders_present(self) -> None:
        """Ensure ARM64X is in the known folder map."""
        assert "arm64x" in WINDOWS_ARCH_FOLDERS
        assert WINDOWS_ARCH_FOLDERS["arm64x"] == "arm64x-windows-msvc"
        assert "arm64ec" in WINDOWS_ARCH_FOLDERS

    def test_sdk_arch_detects_arm64x_env(self, monkeypatch) -> None:
        monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
        monkeypatch.setenv("PROCESSOR_ARCHITEW6432", "ARM64")
        p = WindowsPlatform()
        # WOW64 scenario: x86_64 app on ARM64X → native is ARM64
        assert p.is_x86_on_arm64x() is True

    def test_sdk_arch_native_arm64x(self, monkeypatch) -> None:
        monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "ARM64")
        monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)
        p = WindowsPlatform()
        info = p.detect_device()
        assert "aarch64" in info.arch or "arm64" in info.arch.lower()

    def test_no_wow64_not_x86_on_arm64x(self, monkeypatch) -> None:
        monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
        monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)
        p = WindowsPlatform()
        assert p.is_x86_on_arm64x() is False


class TestARM64XCMakeTemplate:
    def test_cmake_supports_arm64x_platform(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("CMakeLists.txt.j2")
        rendered = t.render(model_path="model.dlc")
        assert "ARM64X" in rendered
        assert "arm64x-windows-msvc" in rendered

    def test_cmake_arm64ec_documented(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("CMakeLists.txt.j2")
        rendered = t.render(model_path="model.dlc")
        assert "ARM64EC" in rendered
        assert "arm64ec" in rendered.lower() or "Emulation" in rendered

    def test_cmake_scenario2_documented(self) -> None:
        """x86_64 app on ARM64X device documented."""
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("CMakeLists.txt.j2")
        rendered = t.render(model_path="model.dlc")
        assert "scenario 2" in rendered.lower() or "x86_64 app" in rendered.lower() or "x64" in rendered
