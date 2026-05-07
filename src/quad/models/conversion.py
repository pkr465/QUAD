"""Pydantic models for model conversion."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ConversionRequest(BaseModel):
    """Input for convert_model tool."""

    source_format: Literal["onnx", "pytorch", "tensorflow", "tflite"]
    model_path: str
    target_sdk: Literal["qnn", "snpe"] = "qnn"
    quantization: Literal["fp32", "int8", "int4"] = "fp32"

    # Input specification
    input_name: Optional[str] = None          # Input tensor name (TF/PyTorch required)
    input_dimensions: Optional[str] = None    # Comma-separated dims e.g. "1,3,224,224"

    # Output specification
    output_nodes: list[str] = Field(default_factory=list)  # TF required

    # Precision
    float_bitwidth: int = 32               # 32 (default) or 16

    # Quantization
    quantization_overrides: Optional[str] = None  # Path to JSON overrides file

    # TF-specific
    allow_unconsumed_nodes: bool = False   # For TF models like MobilenetSSD

    # Image format (from SNPE Input Image Formatting docs)
    # SNPE requires NHWC; PyTorch models provide NCHW — set input_layout to "nchw"
    # to signal that NCHW→NHWC transposition is needed before inference.
    input_layout: Literal["nhwc", "nchw", "auto"] = "auto"
    # Channel order must match training. Caffe/legacy models use BGR; most modern use RGB.
    channel_order: Literal["rgb", "bgr", "auto"] = "auto"
    # Per-channel mean values to subtract (in channel_order order).
    # e.g. AlexNet BGR mean: [104.0, 117.0, 123.0]
    mean_values: Optional[list[float]] = None


class ConversionResult(BaseModel):
    """Output from convert_model tool."""

    output_path: str
    model_size_mb: float = Field(ge=0)
    original_size_mb: float = Field(ge=0)
    compression_ratio: float = Field(ge=0)
    supported_ops_pct: float = Field(ge=0, le=100)
    unsupported_ops: list[str] = Field(default_factory=list)
    quantization_applied: str
    conversion_time_s: float = Field(ge=0)
    target_sdk: str
    warnings: list[str] = Field(default_factory=list)

    # Conversion notes — surfaces MODEL_TIPS and format-specific gotchas
    # e.g. "MobilenetSSD requires allow_unconsumed_nodes=True"
    conversion_notes: list[str] = Field(default_factory=list)
    # Image format guidance for inference — NHWC layout, channel order, mean values
    image_format_notes: list[str] = Field(default_factory=list)
