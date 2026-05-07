"""QUAD PSNPE — Parallel SNPE for bulk heterogeneous inference.

PSNPE (Parallel SNPE) runs multiple SNPE instances concurrently across
heterogeneous hardware (HVX, HMX) to maximise bulk inference throughput.

Usage:
    from quad.psnpe import PSNPEManager, BuildConfig, RuntimeConfig, ExecutionMode

    mgr = PSNPEManager()
    mgr.build(BuildConfig(
        container_path="model.dlc",
        runtime_configs=[RuntimeConfig(runtime="dsp", num_instances=4)],
        output_buffer_names=["output:0"],
    ))
    results = mgr.execute_sync([{"input:0": raw_bytes}])
"""

from quad.psnpe.config import (
    BuildConfig,
    ExecutionMode,
    ModelConfig,
    PSNPEConfig,
    RuntimeConfig,
)
from quad.psnpe.manager import PSNPEManager

__all__ = [
    "BuildConfig",
    "ExecutionMode",
    "ModelConfig",
    "PSNPEConfig",
    "PSNPEManager",
    "RuntimeConfig",
]
