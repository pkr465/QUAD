"""Linux platform — SSH-based communication with remote Linux device (e.g. Arduino UNO Q)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from quad.platforms.base import DeviceInfo, Platform


class LinuxPlatform(Platform):
    """Platform for Linux target devices (QCS2210, ARM embedded, etc.).

    Communicates via SSH for remote command execution and SCP for
    file transfers. Supports local execution when host == target.
    """

    def __init__(
        self,
        ssh_host: str = "",
        ssh_user: str = "root",
        ssh_key: str = "",
    ):
        self.ssh_host = ssh_host or os.environ.get("TARGET_IP", "")
        self.ssh_user = ssh_user or os.environ.get("TARGET_USER", "root")
        self.ssh_key = ssh_key
        self._local = not bool(self.ssh_host)

    def detect_device(self) -> DeviceInfo:
        """Detect target Linux device via /proc/cpuinfo (remote or local)."""
        sdk_path = os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("SNPE_ROOT", "")

        if self._local:
            # Read local system info
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                arch = "aarch64" if "aarch64" in os.uname().machine else "x86_64"
            except Exception:
                arch = "x86_64"
                cpuinfo = ""
            return DeviceInfo(
                platform="linux",
                arch=arch,
                os_name="Linux (local)",
                sdk_path=sdk_path,
                is_connected=True,
            )

        # Remote device
        rc, stdout, _ = self.run_command(["uname", "-m"], timeout=5)
        arch = stdout.strip() if rc == 0 else "aarch64"
        if arch == "aarch64":
            arch = "aarch64"

        rc2, os_out, _ = self.run_command(
            ["sh", "-c", "grep PRETTY_NAME /etc/os-release | cut -d= -f2"], timeout=5
        )
        os_name = os_out.strip().strip('"') if rc2 == 0 else "Linux"

        return DeviceInfo(
            platform="linux",
            arch=arch,
            os_name=os_name,
            sdk_path=sdk_path,
            is_connected=(rc == 0),
        )

    def _ssh_prefix(self) -> list[str]:
        """Build SSH command prefix."""
        if self._local:
            return []
        cmd = ["ssh"]
        if self.ssh_key:
            cmd += ["-i", self.ssh_key]
        cmd += ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
        cmd += [f"{self.ssh_user}@{self.ssh_host}"]
        return cmd

    def run_command(self, cmd: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        """Run command on target (local or via SSH)."""
        full_cmd = self._ssh_prefix() + (cmd if self._local else [" ".join(cmd)])
        try:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Timed out after {timeout}s"
        except FileNotFoundError as e:
            return -1, "", str(e)

    def push_file(self, local_path: str, remote_path: str) -> None:
        """Transfer file to target via SCP or local copy."""
        if self._local:
            import shutil
            Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, remote_path)
            return
        cmd = ["scp"]
        if self.ssh_key:
            cmd += ["-i", self.ssh_key]
        cmd += ["-o", "StrictHostKeyChecking=no"]
        cmd += [local_path, f"{self.ssh_user}@{self.ssh_host}:{remote_path}"]
        subprocess.run(cmd, check=True)

    def pull_file(self, remote_path: str, local_path: str) -> None:
        """Retrieve file from target via SCP or local copy."""
        if self._local:
            import shutil
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_path, local_path)
            return
        cmd = ["scp"]
        if self.ssh_key:
            cmd += ["-i", self.ssh_key]
        cmd += ["-o", "StrictHostKeyChecking=no"]
        cmd += [f"{self.ssh_user}@{self.ssh_host}:{remote_path}", local_path]
        subprocess.run(cmd, check=True)

    def is_available(self) -> bool:
        """Check SSH connectivity."""
        if self._local:
            return True
        rc, _, _ = self.run_command(["echo", "ok"], timeout=5)
        return rc == 0

    def get_sdk_arch(self) -> str:
        """Return SDK architecture folder name for this device."""
        info = self.detect_device()
        if "aarch64" in info.arch:
            # Detect GCC version for correct folder
            rc, out, _ = self.run_command(
                ["sh", "-c", "gcc --version | head -1 | grep -oP '\\d+\\.\\d+' | head -1"],
                timeout=5
            )
            gcc_ver = out.strip()
            if gcc_ver.startswith("11"):
                return "aarch64-oe-linux-gcc11.2"
            elif gcc_ver.startswith("9.4"):
                return "aarch64-ubuntu-gcc9.4"
            elif gcc_ver.startswith("9"):
                return "aarch64-oe-linux-gcc9.3"
            return "aarch64-oe-linux-gcc11.2"
        return "x86_64-linux-clang"
