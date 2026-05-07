"""QUAD Compiler — Portable IR and compilation pipeline."""

from quad.compiler.ir import QuadIR, IRNode, IRGraph
from quad.compiler.frontend_onnx import compile_onnx
from quad.compiler.qbin import QBin
from quad.compiler.capabilities import ComputeCapability, get_capability
from quad.compiler.pipeline import compile_model
from quad.compiler.model_conversion import (
    ConversionConfig,
    QAIRTConversionConfig,
    InputSpec,
    SourceFramework,
    build_quantize_cli_args,
    build_graph_prepare_cli_args,
    build_mobilenet_ssd_conversion_args,
    build_deeplabv3_conversion_args,
    is_cache_compatible,
    convert_nchw_to_nhwc,
    convert_channel_order,
    prepare_batch_input,
    generate_image_format_notes,
    ONNX_CONVERSION_NOTES,
    TFLITE_CONVERSION_NOTES,
    PYTORCH_CONVERSION_NOTES,
    QUANTIZATION_NOTES,
    OFFLINE_CACHE_NOTES,
    QAIRT_CONVERTER_NOTES,
    IMAGE_FORMAT_NOTES,
    MODEL_TIPS,
)

__all__ = [
    "ComputeCapability",
    "ConversionConfig",
    "IRGraph",
    "IRNode",
    "IMAGE_FORMAT_NOTES",
    "InputSpec",
    "MODEL_TIPS",
    "OFFLINE_CACHE_NOTES",
    "ONNX_CONVERSION_NOTES",
    "PYTORCH_CONVERSION_NOTES",
    "QAIRTConversionConfig",
    "QAIRT_CONVERTER_NOTES",
    "QUANTIZATION_NOTES",
    "QBin",
    "QuadIR",
    "SourceFramework",
    "TFLITE_CONVERSION_NOTES",
    "build_deeplabv3_conversion_args",
    "build_graph_prepare_cli_args",
    "build_mobilenet_ssd_conversion_args",
    "build_quantize_cli_args",
    "compile_model",
    "compile_onnx",
    "convert_channel_order",
    "convert_nchw_to_nhwc",
    "get_capability",
    "generate_image_format_notes",
    "is_cache_compatible",
    "prepare_batch_input",
]
