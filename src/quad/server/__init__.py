"""QUAD MCP Server — entry point with adapter-backed tools."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from fastmcp import FastMCP

from quad.adapters.factory import AdapterFactory
from quad.config import load_config
from quad.logging import setup_logging
from quad.sdk_manager import startup_resolve_and_log
from quad.tools.convert_model import convert_model_impl
from quad.tools.generate_code import generate_code_impl
from quad.tools.hardware_detect import hardware_detect_impl
from quad.tools.orchestrate_workload import orchestrate_workload_impl
from quad.tools.profile_workload import profile_workload_impl

# Load config and initialize
_config = load_config()

# SDK auto-discovery — runs before the AdapterFactory so the env vars
# the factory inspects (QAIRT_SDK_ROOT etc.) are populated by whatever
# we found in env / quad.toml / ./sdks / vendor defaults.
_resolved_sdk = startup_resolve_and_log()

_factory = AdapterFactory(_config)

setup_logging(_config.log_level, _config.log_format)

mcp = FastMCP(name="QUAD — Qualcomm Unified Agent for Developers")


@mcp.tool
async def hardware_detect(platform: Literal["windows", "linux", "android"]) -> dict[str, Any]:
    """Detect Qualcomm chipset and available compute units (CPU/GPU/NPU) on the target platform.

    Returns a DeviceProfile with chipset info, CPU/GPU/NPU specs, available memory,
    and detected SDK installation path.
    """
    return await hardware_detect_impl(platform, _factory)


@mcp.tool
async def convert_model(
    source_format: Literal["onnx", "pytorch", "tensorflow", "tflite"],
    model_path: str,
    target_sdk: Literal["qnn", "snpe"] = "qnn",
    quantization: Literal["fp32", "int8", "int4"] = "fp32",
) -> dict[str, Any]:
    """Convert ML model from source framework to QNN/SNPE format with optional quantization.

    Supports ONNX, PyTorch, TensorFlow, and TFLite inputs.
    Outputs QNN context binary (.bin) or SNPE DLC (.dlc) format.
    Quantization options: fp32 (no quantization), int8, int4.
    """
    return await convert_model_impl(source_format, model_path, target_sdk, quantization, _factory)


@mcp.tool
async def profile_workload(
    model_path: str,
    platform: Literal["windows", "linux", "android"] = "windows",
    runtime: Literal["cpu", "gpu", "npu", "auto"] = "auto",
    duration_s: int = 10,
) -> dict[str, Any]:
    """Run platform-appropriate profiler and return structured performance/power metrics.

    Returns latency statistics (mean, p50, p95, p99), throughput (FPS),
    power consumption (mW), memory usage (MB), and per-layer profiling data.
    """
    return await profile_workload_impl(model_path, platform, runtime, duration_s, _factory)


@mcp.tool
async def orchestrate_workload(
    model_path: str,
    power_mode: Literal["performance", "balanced", "efficiency"] = "balanced",
) -> dict[str, Any]:
    """Allocate inference graph nodes across CPU/GPU/NPU based on profiling and power mode.

    Power modes:
    - performance: maximize NPU usage for lowest latency
    - balanced: NPU for heavy layers, CPU/GPU for lighter ones
    - efficiency: minimize power consumption, CPU-first allocation
    """
    return await orchestrate_workload_impl(model_path, power_mode, _factory)


@mcp.tool
async def generate_code(
    platform: Literal["windows", "linux", "android"],
    sdk: Literal["qnn", "snpe"] = "qnn",
    language: Literal["cpp", "python", "java", "kotlin", "arduino_sketch"] = "python",
    model_path: str = "model.bin",
) -> dict[str, Any]:
    """Generate platform-specific inference code using templates.

    Produces compilable source code, build instructions, and dependency lists
    for the specified platform/language/SDK combination.
    """
    template_dir = str(Path(__file__).parents[2] / "templates")
    # Fall back to local templates dir if the package path doesn't exist
    if not Path(template_dir).exists():
        template_dir = _config.template_dir
    return await generate_code_impl(platform, sdk, language, model_path, template_dir)


def cli() -> None:
    """CLI entry point for quad-server."""
    mcp.run()


if __name__ == "__main__":
    cli()
