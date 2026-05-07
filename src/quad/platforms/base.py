"""Abstract base class for QUAD platform implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DeviceInfo:
    """Information about a connected target device."""
    platform: str
    arch: str          # e.g. "x86_64", "aarch64"
    os_name: str       # e.g. "Windows 11", "Ubuntu 22.04"
    sdk_path: str = ""
    is_connected: bool = True


class Platform(ABC):
    """Base class for platform-specific device communication.

    Provides a unified interface for detecting hardware, running
    commands, and deploying files across Windows, Linux, and Android.
    """

    @abstractmethod
    def detect_device(self) -> DeviceInfo:
        """Detect the target device and return its info."""
        ...

    @abstractmethod
    def run_command(self, cmd: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        """Run a command on the target device.

        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        ...

    @abstractmethod
    def push_file(self, local_path: str, remote_path: str) -> None:
        """Transfer a file to the target device."""
        ...

    @abstractmethod
    def pull_file(self, remote_path: str, local_path: str) -> None:
        """Retrieve a file from the target device."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this platform's target device is reachable."""
        ...
