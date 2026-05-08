"""Pure model-conversion operation."""

from __future__ import annotations

from typing import Any, Literal, Optional

from quad.adapters.factory import AdapterFactory
from quad.models.conversion import ConversionRequest


async def convert_model(
    source_format: Literal["onnx", "pytorch", "tensorflow", "tflite"],
    model_path: str,
    target_sdk: Literal["qnn", "snpe"],
    quantization: Literal["fp32", "int8", "int4"],
    factory: AdapterFactory,
    *,
    input_name: Optional[str] = None,
    input_dimensions: Optional[str] = None,
    output_nodes: Optional[list[str]] = None,
    float_bitwidth: int = 32,
    quantization_overrides: Optional[str] = None,
    allow_unconsumed_nodes: bool = False,
    input_layout: Literal["nhwc", "nchw", "auto"] = "auto",
    channel_order: Literal["rgb", "bgr", "auto"] = "auto",
    mean_values: Optional[list[float]] = None,
) -> dict[str, Any]:
    """Convert a model. Returns ConversionResult dict (no MCP enrichment)."""
    adapter = factory.get_adapter(target_sdk)
    request = ConversionRequest(
        source_format=source_format,
        model_path=model_path,
        target_sdk=target_sdk,
        quantization=quantization,
        input_name=input_name,
        input_dimensions=input_dimensions,
        output_nodes=output_nodes or [],
        float_bitwidth=float_bitwidth,
        quantization_overrides=quantization_overrides,
        allow_unconsumed_nodes=allow_unconsumed_nodes,
        input_layout=input_layout,
        channel_order=channel_order,
        mean_values=mean_values,
    )
    result = await adapter.convert_model(request)
    return result.model_dump()
