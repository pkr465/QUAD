"""QUAD Device — hardware discovery and abstraction.

In addition to the legacy in-memory profiles for known Qualcomm
chipsets, ``list_devices()`` now performs a real local-host probe
(see ``quad.runtime.host_probe``) to populate fields with the actual
hardware on the running machine. This closes GAP_ANALYSIS T3.6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from quad.runtime.host_probe import hostinfo_to_device_profiles, probe_host

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "gpu", "npu", "auto"]

# Legacy fallback profiles — used when the host probe doesn't detect
# a particular compute unit. Values match what we previously hardcoded
# (so behaviour is unchanged on systems where the probe finds nothing,
# e.g. the CI sandbox).
_FALLBACK_DEVICES: dict[str, dict[str, Any]] = {
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

_KNOWN_DEVICES: dict[str, dict[str, Any]] = dict(_FALLBACK_DEVICES)

# Priority order for auto device selection
_DEVICE_PRIORITY = ["npu", "gpu", "cpu"]


def _refresh_known_devices(force: bool = False) -> None:
    """Refresh ``_KNOWN_DEVICES`` from a real host probe.

    This is called lazily by ``list_devices()`` and ``Device.__init__``
    so we never block module import on a subprocess. The first call
    overlays the host probe's findings on top of the fallback profiles
    — if the probe found a real CPU we use the real CPU; if it didn't,
    we keep the Oryon fallback.

    Args:
        force: re-probe even if a previous call populated _KNOWN_DEVICES.
            Useful for tests that monkeypatch the probe.
    """
    global _KNOWN_DEVICES
    if not force and getattr(_refresh_known_devices, "_done", False):
        return
    try:
        info = probe_host()
        probed = hostinfo_to_device_profiles(info)
        merged = dict(_FALLBACK_DEVICES)
        merged.update(probed)
        _KNOWN_DEVICES = merged
        logger.debug(
            "device_profiles_refreshed",
            extra={"source": info.source, "probed_keys": list(probed.keys())},
        )
    except Exception as e:
        logger.debug("host probe failed; using fallback profiles: %s", e)
    setattr(_refresh_known_devices, "_done", True)


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


def list_devices(refresh: bool = False) -> list[Device]:
    """Enumerate all available compute devices on the local machine.

    The first call probes the local hardware via
    ``quad.runtime.host_probe.probe_host`` to populate device fields
    with real CPU / GPU / NPU info. Subsequent calls reuse the
    cached probe (set ``refresh=True`` to re-probe).

    Args:
        refresh: re-run the host probe even if a previous call cached
            its result.

    Returns:
        List of Device objects for all detected compute units.
    """
    _refresh_known_devices(force=refresh)
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
    _refresh_known_devices()
    return device_type in _KNOWN_DEVICES
