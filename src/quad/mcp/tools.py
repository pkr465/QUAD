"""MCP-tool wrappers — thin enrichment over :mod:`quad.core.operations`.

These are the functions that ``quad.mcp.server`` registers via the
``@mcp.tool`` decorator. They:

  1. Call into :mod:`quad.core.operations` for the pure data payload
  2. Add ``ui`` / ``tips`` / ``suggestions`` keys via
     :mod:`quad.mcp.enrichment`
  3. Return the enriched dict

Backward compatibility: the ``_impl`` aliases are re-exported by
:mod:`quad.tools.<name>` so existing imports still work.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from quad.adapters.factory import AdapterFactory
from quad.core.operations import (
    convert_model as _core_convert_model,
    generate_code as _core_generate_code,
    hardware_detect as _core_hardware_detect,
    orchestrate_workload as _core_orchestrate_workload,
    profile_workload as _core_profile_workload,
)
from quad.mcp.enrichment import (
    enrich_convert_model,
    enrich_generate_code,
    enrich_hardware_detect,
    enrich_orchestrate_workload,
    enrich_profile_workload,
)


async def hardware_detect_impl(
    platform: Literal["windows", "linux", "android"],
    factory: AdapterFactory,
    *,
    enrich: bool = True,
) -> dict[str, Any]:
    payload = await _core_hardware_detect(platform, factory)
    return enrich_hardware_detect(payload) if enrich else payload


async def convert_model_impl(
    source_format: Literal["onnx", "pytorch", "tensorflow", "tflite"],
    model_path: str,
    target_sdk: Literal["qnn", "snpe"],
    quantization: Literal["fp32", "int8", "int4"],
    factory: AdapterFactory,
    input_name: Optional[str] = None,
    input_dimensions: Optional[str] = None,
    output_nodes: Optional[list[str]] = None,
    float_bitwidth: int = 32,
    quantization_overrides: Optional[str] = None,
    allow_unconsumed_nodes: bool = False,
    input_layout: Literal["nhwc", "nchw", "auto"] = "auto",
    channel_order: Literal["rgb", "bgr", "auto"] = "auto",
    mean_values: Optional[list[float]] = None,
    *,
    enrich: bool = True,
) -> dict[str, Any]:
    payload = await _core_convert_model(
        source_format=source_format,
        model_path=model_path,
        target_sdk=target_sdk,
        quantization=quantization,
        factory=factory,
        input_name=input_name,
        input_dimensions=input_dimensions,
        output_nodes=output_nodes,
        float_bitwidth=float_bitwidth,
        quantization_overrides=quantization_overrides,
        allow_unconsumed_nodes=allow_unconsumed_nodes,
        input_layout=input_layout,
        channel_order=channel_order,
        mean_values=mean_values,
    )
    return enrich_convert_model(payload) if enrich else payload


async def profile_workload_impl(
    model_path: str,
    platform: Literal["windows", "linux", "android"],
    runtime: Literal["cpu", "gpu", "npu", "auto"],
    duration_s: int,
    factory: AdapterFactory,
    profiling_level: Literal["basic", "detailed", "linting", "qhas"] = "detailed",
    htp_soc: str = "sm8750",
    sdk_root: str | None = None,
    *,
    enrich: bool = True,
) -> dict[str, Any]:
    payload = await _core_profile_workload(
        model_path=model_path,
        platform=platform,
        runtime=runtime,
        duration_s=duration_s,
        factory=factory,
        profiling_level=profiling_level,
        htp_soc=htp_soc,
        sdk_root=sdk_root,
    )
    return enrich_profile_workload(payload, profiling_level=profiling_level) if enrich else payload


async def orchestrate_workload_impl(
    model_path: str,
    power_mode: Literal["performance", "balanced", "efficiency"],
    factory: AdapterFactory,
    *,
    enrich: bool = True,
) -> dict[str, Any]:
    payload = await _core_orchestrate_workload(
        model_path=model_path,
        power_mode=power_mode,
        factory=factory,
    )
    return enrich_orchestrate_workload(payload) if enrich else payload


async def generate_code_impl(
    platform: Literal["windows", "linux", "android"],
    sdk: Literal["qnn", "snpe"],
    language: Literal["cpp", "python", "java", "kotlin", "arduino_sketch"],
    model_path: str,
    template_dir: str = "templates",
    *,
    enrich: bool = True,
) -> dict[str, Any]:
    payload = await _core_generate_code(
        platform=platform,
        sdk=sdk,
        language=language,
        model_path=model_path,
        template_dir=template_dir,
    )
    return enrich_generate_code(payload, platform=platform, language=language, sdk=sdk) if enrich else payload
