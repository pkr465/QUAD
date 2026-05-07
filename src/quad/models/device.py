"""Pydantic model for device hardware profile."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DeviceProfile(BaseModel):
    """Hardware profile returned by hardware_detect tool."""

    chipset: str = Field(description="Full chipset name, e.g. 'Snapdragon X Elite X1E-80-100'")
    platform: Literal["windows", "linux", "android"]
    cpu_cores: int = Field(ge=1)
    cpu_arch: str = Field(description="e.g. 'Oryon ARM64', 'Kryo ARM64'")
    cpu_freq_ghz: float = Field(gt=0)
    gpu_model: str = Field(description="e.g. 'Adreno X1-85'")
    gpu_tflops: float = Field(ge=0)
    npu_model: str = Field(description="e.g. 'Hexagon NPU'")
    npu_tops: float = Field(ge=0)
    ram_gb: float = Field(gt=0)
    sdk_path: str | None = None
    sdk_version: str | None = None
    available_runtimes: list[Literal["cpu", "gpu", "npu"]] = Field(default_factory=lambda: ["cpu"])
