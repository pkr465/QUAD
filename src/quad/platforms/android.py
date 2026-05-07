"""Android platform — ADB-based communication with Android devices."""

from __future__ import annotations

import os
import subprocess

from quad.platforms.base import DeviceInfo, Platform


class AndroidPlatform(Platform):
    """Platform for Android devices (Snapdragon 8 Elite, etc.).

    Uses ADB (Android Debug Bridge) for device communication,
    file transfer, and command execution.
    """

    def __init__(self, device_serial: str = "", adb_path: str = "adb"):
        self.device_serial = device_serial or os.environ.get("ANDROID_SERIAL", "")
        self.adb_path = adb_path

    def _adb(self, *args: str) -> list[str]:
        """Build ADB command with optional device serial."""
        cmd = [self.adb_path]
        if self.device_serial:
            cmd += ["-s", self.device_serial]
        return cmd + list(args)

    def detect_device(self) -> DeviceInfo:
        """Detect Android device properties via ADB getprop."""
        sdk_path = os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("SNPE_ROOT", "")

        if not self.is_available():
            return DeviceInfo(
                platform="android", arch="aarch64",
                os_name="Android (not connected)", sdk_path=sdk_path,
                is_connected=False,
            )

        rc, chipset, _ = self.run_command(
            [self.adb_path, "shell", "getprop", "ro.board.platform"], timeout=10
        )
        rc2, android_ver, _ = self.run_command(
            [self.adb_path, "shell", "getprop", "ro.build.version.release"], timeout=10
        )

        return DeviceInfo(
            platform="android",
            arch="aarch64",  # All modern Android devices are ARM64
            os_name=f"Android {android_ver.strip()} ({chipset.strip()})",
            sdk_path=sdk_path,
            is_connected=True,
        )

    def run_command(self, cmd: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        """Run command on device via ADB shell."""
        # If cmd starts with adb, run directly; otherwise wrap in adb shell
        if cmd and cmd[0] in (self.adb_path, "adb"):
            full_cmd = cmd
        else:
            full_cmd = self._adb("shell") + cmd
        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Timed out after {timeout}s"
        except FileNotFoundError:
            return -1, "", f"ADB not found: {self.adb_path}"

    def push_file(self, local_path: str, remote_path: str) -> None:
        """Push file to device via ADB."""
        subprocess.run(
            self._adb("push", local_path, remote_path), check=True
        )

    def pull_file(self, remote_path: str, local_path: str) -> None:
        """Pull file from device via ADB."""
        subprocess.run(
            self._adb("pull", remote_path, local_path), check=True
        )

    def is_available(self) -> bool:
        """Check if an Android device is connected via ADB."""
        try:
            result = subprocess.run(
                [self.adb_path, "devices"], capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split("\n")
            # Filter for connected devices (not 'offline' or 'unauthorized')
            connected = [
                l for l in lines[1:]
                if l.strip() and "device" in l and "offline" not in l
            ]
            if self.device_serial:
                return any(self.device_serial in l for l in connected)
            return len(connected) > 0
        except Exception:
            return False

    def get_sdk_arch(self) -> str:
        """Return SDK architecture folder name for Android."""
        return "aarch64-android"

    def get_npu_status(self) -> dict:
        """Check NPU availability on the Android device."""
        rc, out, _ = self.run_command(
            ["cat", "/sys/class/npu/npu0/status"], timeout=5
        )
        return {"available": rc == 0, "status": out.strip()}
