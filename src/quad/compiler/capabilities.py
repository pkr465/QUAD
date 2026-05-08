"""Compute capabilities — chipset abstraction for portability."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComputeCapability:
    """Hardware compute capability descriptor (e.g. ``qnpu_v3`` for Hexagon v73).

    Defines what operations a target supports and at what performance level.
    The capability tag is the matching key used by ``QBin`` at load time
    to pick a target-specific pre-compiled binary.
    """
    name: str           # e.g. "qnpu_v3"
    chipset: str        # e.g. "Snapdragon X Elite"
    npu_version: str    # e.g. "hexagon_v73"
    npu_tops: float
    gpu_name: str
    gpu_tflops: float
    supports_int4: bool = False
    supports_int8: bool = True
    supports_fp16: bool = True
    supports_fp32: bool = True
    vtcm_kb: int = 2048     # Vector Tightly Coupled Memory
    hvx_width: int = 128    # HVX vector width in bytes


# Registry of known compute capabilities
_CAPABILITIES: dict[str, ComputeCapability] = {
    "qnpu_v3": ComputeCapability(
        name="qnpu_v3",
        chipset="Snapdragon X Elite",
        npu_version="hexagon_v73",
        npu_tops=45.0,
        gpu_name="Adreno X1-85",
        gpu_tflops=4.6,
        supports_int4=True,
        supports_int8=True,
        vtcm_kb=4096,
        hvx_width=128,
    ),
    "qnpu_v3_mobile": ComputeCapability(
        name="qnpu_v3_mobile",
        chipset="Snapdragon 8 Elite",
        npu_version="hexagon_v73",
        npu_tops=48.0,
        gpu_name="Adreno 830",
        gpu_tflops=5.0,
        supports_int4=True,
        supports_int8=True,
        vtcm_kb=4096,
        hvx_width=128,
    ),
    "qnpu_v2": ComputeCapability(
        name="qnpu_v2",
        chipset="Snapdragon 8 Gen 3",
        npu_version="hexagon_v69",
        npu_tops=36.0,
        gpu_name="Adreno 750",
        gpu_tflops=3.8,
        supports_int4=False,
        supports_int8=True,
        vtcm_kb=2048,
        hvx_width=128,
    ),
    "qdsp_v66": ComputeCapability(
        name="qdsp_v66",
        chipset="QCS2210",
        npu_version="hexagon_v66",
        npu_tops=1.0,
        gpu_name="Adreno 504",
        gpu_tflops=0.4,
        supports_int4=False,
        supports_int8=True,
        vtcm_kb=256,
        hvx_width=128,
    ),
}


def get_capability(name: str) -> ComputeCapability:
    """Get compute capability by name."""
    if name not in _CAPABILITIES:
        raise ValueError(
            f"Unknown compute capability '{name}'. "
            f"Available: {list(_CAPABILITIES.keys())}"
        )
    return _CAPABILITIES[name]


def list_capabilities() -> list[ComputeCapability]:
    """List all known compute capabilities."""
    return list(_CAPABILITIES.values())


def get_capability_for_chipset(chipset: str) -> ComputeCapability | None:
    """Find compute capability by chipset name (partial match)."""
    chipset_lower = chipset.lower()
    for cap in _CAPABILITIES.values():
        if chipset_lower in cap.chipset.lower():
            return cap
    return None
