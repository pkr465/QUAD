"""QUAD Runtime — unified Device, Tensor, Model, and Stream APIs.

This is the core programming interface for QUAD platform.
Usage:
    import quad

    device = quad.Device("npu")
    model = quad.load("model.onnx", device=device)
    output = model(input_tensor)
"""

from quad.runtime.device import Device, list_devices, is_available
from quad.runtime.tensor import Tensor
from quad.runtime.model import Model, load
from quad.runtime.stream import Stream, Event
from quad.runtime.memory import MemoryPool
from quad.runtime.power import PowerMonitor, PowerMode, estimate_battery_life

__all__ = [
    "Device",
    "Event",
    "MemoryPool",
    "Model",
    "PowerMode",
    "PowerMonitor",
    "Stream",
    "Tensor",
    "estimate_battery_life",
    "is_available",
    "list_devices",
    "load",
]
