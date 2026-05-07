"""Windows platform — local execution on Windows on Snapdragon.

ARM64X Support (SC8380XP):
  ARM64X is a binary format that works with both ARM64/ARM64EC apps AND x86_64 apps.
  SDK architecture folder: arm64x-windows-msvc

  Two supported scenarios (from SNPE Windows ARM64X docs):
  1. ARM64/ARM64EC/ARM64X app → link ARM64X SNPE.lib, run ARM64X SNPE.dll
  2. x86_64 app → link x86_64 SNPE.lib, run ARM64X SNPE.dll

  Tools, APIs, tutorials are the SAME as ARM64.
  Use PROCESSOR_ARCHITEW6432 env var to detect x86_64 app running on ARM64X.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path

from quad.platforms.base import DeviceInfo, Platform


# Known Windows SDK architecture folder names
WINDOWS_ARCH_FOLDERS = {
    "x64":     "x86_64-windows-msvc",
    "x86_64":  "x86_64-windows-msvc",
    "amd64":   "x86_64-windows-msvc",
    "arm64":   "aarch64-windows-msvc",
    "aarch64": "aarch64-windows-msvc",
    "arm64x":  "arm64x-windows-msvc",   # SC8380XP — new binary format
    "arm64ec": "aarch64-windows-msvc",  # ARM64EC uses same libs as ARM64
}

# SC8380XP is the Qualcomm chip that supports ARM64X
SC8380XP_CHIPSETS = ("SC8380XP", "Snapdragon X Elite")


class WindowsPlatform(Platform):
    """Platform for Windows on Snapdragon (Qualcomm X Elite / SC8380XP) devices.

    Runs commands locally (host and target are the same machine).
    Uses environment variables for architecture detection.

    ARM64X note:
      On ARM64X (SC8380XP), PROCESSOR_ARCHITECTURE may report x86_64
      when running an x86_64 app. Use PROCESSOR_ARCHITEW6432 to detect
      the underlying native architecture.
    """

    def detect_device(self) -> DeviceInfo:
        """Detect hardware via environment and WMI (when available)."""
        sdk_path = (
            os.environ.get("QAIRT_SDK_ROOT")
            or os.environ.get("QNN_SDK_ROOT")
            or ""
        )

        # Detect architecture — PROCESSOR_ARCHITECTURE can be misleading on ARM64X
        # when an x86_64 app runs on an ARM64X device (WOW64 scenario)
        native_arch = os.environ.get("PROCESSOR_ARCHITEW6432", "").lower()  # WOW64 native
        proc_arch = os.environ.get("PROCESSOR_ARCHITECTURE", "").lower()

        # PROCESSOR_ARCHITEW6432 is only set in WOW64 (x86 on ARM64X)
        # If set: native is ARM64X/ARM64, running x86_64 app
        if native_arch:
            arch = "arm64x" if "arm64" in native_arch else native_arch
        else:
            arch = proc_arch if proc_arch else "x86_64"
            if arch == "arm64":
                arch = "aarch64"

        return DeviceInfo(
            platform="windows",
            arch=arch,
            os_name=f"Windows (arch={arch})",
            sdk_path=sdk_path,
            is_connected=True,
        )

    def run_command(self, cmd: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd[0]}"

    def push_file(self, local_path: str, remote_path: str) -> None:
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, remote_path)

    def pull_file(self, remote_path: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)

    def is_available(self) -> bool:
        return True

    def get_sdk_arch(self) -> str:
        """Return the SDK architecture folder name for this device.

        ARM64X scenario 1 (native ARM64X app): arm64x-windows-msvc
        ARM64X scenario 2 (x86_64 app on ARM64X): x86_64-windows-msvc
          → Can still use ARM64X SNPE.dll at runtime
        ARM64: aarch64-windows-msvc
        x86_64: x86_64-windows-msvc
        """
        info = self.detect_device()
        arch = info.arch.lower()
        return WINDOWS_ARCH_FOLDERS.get(arch, "x86_64-windows-msvc")

    def is_arm64x(self) -> bool:
        """True if running on ARM64X (SC8380XP) hardware."""
        info = self.detect_device()
        return "arm64x" in info.arch.lower()

    def is_x86_on_arm64x(self) -> bool:
        """True if running an x86_64 app on ARM64X (WOW64 scenario).

        In this case: link x86_64 SNPE.lib but ARM64X SNPE.dll handles execution.
        """
        native = os.environ.get("PROCESSOR_ARCHITEW6432", "")
        proc = os.environ.get("PROCESSOR_ARCHITECTURE", "")
        return bool(native) and proc.lower() in ("amd64", "x86_64", "x86")
