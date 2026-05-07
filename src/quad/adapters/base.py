"""Abstract base class for SDK adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quad.models.conversion import ConversionRequest, ConversionResult
from quad.models.device import DeviceProfile
from quad.models.profiling import ProfileRequest, ProfilingReport


class SDKAdapter(ABC):
    """Base interface for all Qualcomm SDK adapters.

    Each adapter wraps a specific SDK (QNN, SNPE, Hexagon, etc.)
    behind a unified async interface. The MockAdapter provides
    simulated responses for development without hardware.
    """

    @abstractmethod
    async def detect_hardware(self, platform: str) -> DeviceProfile:
        """Detect hardware capabilities for the given platform."""
        ...

    @abstractmethod
    async def convert_model(self, request: ConversionRequest) -> ConversionResult:
        """Convert a model to the adapter's target format."""
        ...

    @abstractmethod
    async def profile(self, request: ProfileRequest) -> ProfilingReport:
        """Profile a model workload and return structured metrics."""
        ...

    @abstractmethod
    async def get_supported_ops(self) -> list[str]:
        """Return list of supported ONNX operator names."""
        ...

    @abstractmethod
    async def execute_inference(self, model_path: str, input_data: Any) -> Any:
        """Execute inference on the given model with input data."""
        ...
