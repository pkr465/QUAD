"""QUAD Device — hardware discovery and abstraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DeviceType = Literal["cpu", "gpu", "npu", "auto"]

# Known device profiles (mock data — replaced by real detection in real mode)
_KNOWN_DEVICES = {
    "npu": {
        "name": "Hexagon NPU",
        "type": "npu",
        "tops": 45.0,
        "cores": 4,
        "memory_mb": 8192,
        "power_typical_mw": 2000,
    },
    "gpu": {
        "name": "Adreno X1-85",
        "type": "gpu",
        "tflops": 4.6,
        "cores": 1024,
        "memory_mb": 8192,
        "power_typical_mw": 3500,
    },
    "cpu": {
        "name": "Oryon ARM64",
        "type": "cpu",
        "cores": 12,
        "freq_ghz": 3.8,
        "memory_mb": 32768,
        "power_typical_mw": 5000,
    },
}

# Priority order for auto device selection
_DEVICE_PRIORITY = ["npu", "gpu", "cpu"]


@dataclass
class Device:
    """Represents a compute device (CPU, GPU, or NPU).

    Usage:
        device = Device("npu")        # Specific device
        device = Device("auto")       # Best available (NPU > GPU > CPU)
        device = Device("npu:0")      # Specific core
    """

    type: DeviceType
    index: int = 0
    name: str = ""
    tops: float = 0.0
    tflops: float = 0.0
    cores: int = 0
    memory_mb: int = 0
    power_typical_mw: float = 0.0
    _available: bool = True

    def __init__(self, device_type: str = "auto", **kwargs):
        # Parse "npu:0" format
        if ":" in device_type:
            parts = device_type.split(":")
            device_type = parts[0]
            self.index = int(parts[1])
        else:
            self.index = 0

        # Resolve "auto" to best available
        if device_type == "auto":
            device_type = _resolve_auto_device()

        self.type = device_type  # type: ignore

        # Load device properties from known profiles
        props = _KNOWN_DEVICES.get(device_type, _KNOWN_DEVICES["cpu"])
        self.name = props.get("name", device_type)
        self.tops = props.get("tops", 0.0)
        self.tflops = props.get("tflops", 0.0)
        self.cores = props.get("cores", 0)
        self.memory_mb = props.get("memory_mb", 0)
        self.power_typical_mw = props.get("power_typical_mw", 0.0)
        self._available = True

        # Apply any overrides
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @property
    def is_available(self) -> bool:
        """Whether this device is currently available."""
        return self._available

    @property
    def is_npu(self) -> bool:
        return self.type == "npu"

    @property
    def is_gpu(self) -> bool:
        return self.type == "gpu"

    @property
    def is_cpu(self) -> bool:
        return self.type == "cpu"

    def __repr__(self) -> str:
        if self.type == "npu":
            return f"Device(type='{self.type}', name='{self.name}', tops={self.tops})"
        elif self.type == "gpu":
            return f"Device(type='{self.type}', name='{self.name}', tflops={self.tflops})"
        return f"Device(type='{self.type}', name='{self.name}', cores={self.cores})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        return False

    def __hash__(self) -> int:
        return hash((self.type, self.index))


def _resolve_auto_device() -> str:
    """Select best available device (NPU > GPU > CPU)."""
    for device_type in _DEVICE_PRIORITY:
        if device_type in _KNOWN_DEVICES:
            return device_type
    return "cpu"


def list_devices() -> list[Device]:
    """Enumerate all available compute devices.

    Returns:
        List of Device objects for all detected compute units.
    """
    devices = []
    for dtype in _DEVICE_PRIORITY:
        if dtype in _KNOWN_DEVICES:
            devices.append(Device(dtype))
    return devices


def is_available(device_type: str) -> bool:
    """Check if a specific device type is available.

    Args:
        device_type: "npu", "gpu", or "cpu"
    """
    return device_type in _KNOWN_DEVICES
