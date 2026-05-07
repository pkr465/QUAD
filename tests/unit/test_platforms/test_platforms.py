"""Tests for platform implementations."""

from __future__ import annotations

import pytest

from quad.platforms import get_platform, AndroidPlatform, LinuxPlatform, WindowsPlatform
from quad.platforms.base import DeviceInfo


class TestGetPlatform:
    def test_windows(self) -> None:
        p = get_platform("windows")
        assert isinstance(p, WindowsPlatform)

    def test_linux(self) -> None:
        p = get_platform("linux")
        assert isinstance(p, LinuxPlatform)

    def test_android(self) -> None:
        p = get_platform("android")
        assert isinstance(p, AndroidPlatform)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown platform"):
            get_platform("tizen")


class TestWindowsPlatform:
    def test_is_always_available(self) -> None:
        p = WindowsPlatform()
        assert p.is_available() is True

    def test_detect_returns_device_info(self) -> None:
        p = WindowsPlatform()
        info = p.detect_device()
        assert isinstance(info, DeviceInfo)
        assert info.platform == "windows"
        assert info.is_connected is True

    def test_run_command_local(self) -> None:
        p = WindowsPlatform()
        # echo works on both Windows and macOS/Linux (for testing host)
        rc, stdout, stderr = p.run_command(["echo", "hello"], timeout=5)
        assert rc == 0
        assert "hello" in stdout

    def test_sdk_arch_x64(self, monkeypatch) -> None:
        monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "AMD64")
        p = WindowsPlatform()
        assert p.get_sdk_arch() == "x86_64-windows-msvc"

    def test_sdk_arch_arm64(self, monkeypatch) -> None:
        monkeypatch.setenv("PROCESSOR_ARCHITECTURE", "ARM64")
        p = WindowsPlatform()
        assert p.get_sdk_arch() == "aarch64-windows-msvc"


class TestLinuxPlatform:
    def test_local_mode_available(self) -> None:
        # No SSH host → local mode → always available
        p = LinuxPlatform(ssh_host="")
        assert p.is_available() is True
        assert p._local is True

    def test_remote_mode_unavailable_no_host(self) -> None:
        p = LinuxPlatform(ssh_host="192.168.99.99")
        assert p._local is False
        # Connection will fail (no real device) — should return False not raise
        result = p.is_available()
        assert isinstance(result, bool)

    def test_local_run_command(self) -> None:
        p = LinuxPlatform(ssh_host="")
        rc, stdout, stderr = p.run_command(["echo", "test"], timeout=5)
        assert rc == 0

    def test_detect_device_local(self) -> None:
        p = LinuxPlatform(ssh_host="")
        info = p.detect_device()
        assert info.platform == "linux"
        assert info.is_connected is True
        assert info.arch in ("x86_64", "aarch64")


class TestAndroidPlatform:
    def test_no_adb_returns_false(self) -> None:
        # If ADB isn't installed or no device, is_available returns False (not raise)
        p = AndroidPlatform(adb_path="adb-nonexistent-tool")
        assert p.is_available() is False

    def test_detect_device_not_connected(self) -> None:
        p = AndroidPlatform(adb_path="adb-nonexistent-tool")
        info = p.detect_device()
        assert info.is_connected is False
        assert info.platform == "android"
        assert info.arch == "aarch64"

    def test_sdk_arch(self) -> None:
        p = AndroidPlatform()
        assert p.get_sdk_arch() == "aarch64-android"
