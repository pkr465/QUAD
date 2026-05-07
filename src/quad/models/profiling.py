"""Pydantic models for workload profiling."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from quad.models.device import DeviceProfile


class LatencyStats(BaseModel):
    """Latency distribution statistics."""

    mean_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)
    min_ms: float = Field(ge=0)
    max_ms: float = Field(ge=0)


class LayerProfile(BaseModel):
    """Per-layer profiling data (microsecond-based, all backends)."""

    name: str
    op_type: str
    runtime: Literal["cpu", "gpu", "npu"]
    latency_ms: float = Field(ge=0)
    memory_mb: float = Field(ge=0)


class LintingLayerProfile(BaseModel):
    """Per-op profiling data from HTP linting mode (cycle-based).

    Cycle counts cannot be directly converted to microseconds due to
    parallelized HTP execution. Use for relative comparisons only.
    """

    name: str
    index: int
    total_cycles: int = Field(ge=0)
    wait_cycles: int = Field(default=0, ge=0)
    overlap_cycles: int = Field(default=0, ge=0)
    overlap_wait_cycles: int = Field(default=0, ge=0)
    overlap_ratio: float = Field(ge=0, le=1)
    cycle_fraction: float = Field(ge=0, le=1)
    resources: list[str] = Field(default_factory=list)  # ["HVX", "HMX", "DMA"]
    is_bottleneck: bool = False
    optimization_hint: Optional[str] = None


class ProfileRequest(BaseModel):
    """Input for profile_workload tool."""

    model_path: str
    platform: Literal["windows", "linux", "android"] = "windows"
    runtime: Literal["cpu", "gpu", "npu", "auto"] = "auto"
    duration_s: int = Field(default=10, ge=1)
    iterations: int = Field(default=100, ge=1)

    # Profiling level — controls depth and output format
    # basic/detailed: all backends, microsecond latency
    # linting: HTP-only, cycle counts per op with overlap analysis
    # qhas: HTP-only, full QNN HTP Analysis Summary + chrometrace
    profiling_level: Literal["basic", "detailed", "linting", "qhas"] = "detailed"

    # HTP-specific options (used when profiling_level is linting or qhas)
    htp_soc: str = "sm8750"               # SoC target for graph-prepare step (QHAS)
    sdk_root: Optional[str] = None         # SDK root for QHAS reader library path

    # Advanced snpe-net-run options
    enable_init_cache: bool = False
    pd_type: Literal["unsigned", "signed"] = "unsigned"
    enable_cpu_fxp: bool = False
    input_dimensions: Optional[dict[str, list[int]]] = None  # {name: [N,H,W,C]}


class ProfilingReport(BaseModel):
    """Output from profile_workload tool."""

    latency: LatencyStats
    throughput_fps: float = Field(ge=0)
    power_mw: float = Field(ge=0)
    memory_peak_mb: float = Field(ge=0)
    memory_avg_mb: float = Field(ge=0)
    utilization: dict[str, float] = Field(
        default_factory=dict, description="Runtime utilization percentages"
    )
    layers: list[LayerProfile] = Field(default_factory=list)
    device: DeviceProfile
    runtime_used: str
    duration_s: float = Field(ge=0)
    profiling_level: str = "detailed"

    # Linting-specific fields (populated when profiling_level == "linting")
    linting_layers: list[LintingLayerProfile] = Field(default_factory=list)
    linting_total_cycles: int = 0
    linting_bottleneck_count: int = 0
    linting_optimization_hints: list[str] = Field(default_factory=list)

    # QHAS-specific fields (populated when profiling_level == "qhas")
    qhas_chrometrace_path: Optional[str] = None
    qhas_htp_json_path: Optional[str] = None
