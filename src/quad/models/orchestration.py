"""Pydantic models for workload orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OrchestrationRequest(BaseModel):
    """Input for orchestrate_workload tool."""

    model_path: str
    power_mode: Literal["performance", "balanced", "efficiency"] = "balanced"


class AllocationMap(BaseModel):
    """Output from orchestrate_workload tool — layer-to-runtime mapping."""

    allocation: dict[str, Literal["cpu", "gpu", "npu"]] = Field(
        description="Mapping of layer_name → assigned runtime"
    )
    projected_latency_ms: float = Field(ge=0)
    projected_power_mw: float = Field(ge=0)
    projected_memory_mb: float = Field(ge=0)
    power_mode: Literal["performance", "balanced", "efficiency"]
    fallback_layers: list[str] = Field(
        default_factory=list, description="Layers forced to CPU due to incompatibility"
    )
    npu_utilization_pct: float = Field(ge=0, le=100)
    gpu_utilization_pct: float = Field(ge=0, le=100)
    cpu_utilization_pct: float = Field(ge=0, le=100)
