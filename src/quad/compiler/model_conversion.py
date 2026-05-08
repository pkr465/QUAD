"""Model conversion utilities — enhanced converter support for all frameworks.

Based on SNPE Model Conversion documentation covering:
- TensorFlow: frozen .pb, checkpoint+meta, SavedModel directory
- TFLite: .tflite files
- PyTorch: .pt/.pth files (via ONNX export)
- ONNX: .onnx files (primary format)

Two converter paths:
1. Unified: qairt-converter --input_network model.onnx (QAIRT 2.x+, recommended)
2. Legacy per-framework: snpe-tensorflow-to-dlc, snpe-onnx-to-dlc, etc.

Key flags:
  --input_network    Path to source model file
  --input_dim        Input tensor name + dimensions: "name" "N,H,W,C"
  --out_node         Output node name(s) — required for TensorFlow
  --output_path      Output .dlc path (legacy) / auto-named (unified)
  --udo_config_paths UDO config for custom ops

TensorFlow Graph Compatibility Rules:
  - All nodes belonging to a layer must be in a UNIQUE TensorFlow scope
  - A node can only belong to a single layer
  - Supported TF layers: BatchNorm, Conv2d, Concat, Deconv, ElementWise,
    FullyConnected, LRN, Pool, Relu/Sigmoid/Tanh/Elu, Softmax, PReLU,
    Slice, Reshape
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Source Framework Types
# ══════════════════════════════════════════════════════════════════════════════

class SourceFramework(str, Enum):
    """Supported source model frameworks."""
    ONNX = "onnx"               # .onnx files (primary, best supported)
    TENSORFLOW = "tensorflow"   # Frozen .pb, checkpoint+meta, SavedModel
    TFLITE = "tflite"           # .tflite files
    PYTORCH = "pytorch"         # .pt/.pth (converted via ONNX or direct)


class TensorFlowInputFormat(str, Enum):
    """TensorFlow model input formats."""
    FROZEN_PB = "frozen_pb"     # Single .pb file (most common)
    CHECKPOINT = "checkpoint"   # Checkpoint + graph meta files
    SAVED_MODEL = "saved_model" # SavedModel directory (TF 2.x)


# ══════════════════════════════════════════════════════════════════════════════
# Converter Selection
# ══════════════════════════════════════════════════════════════════════════════

# Legacy per-framework converters (still used for UDO and some edge cases)
LEGACY_CONVERTERS: dict[str, str] = {
    "onnx": "snpe-onnx-to-dlc",
    "tensorflow": "snpe-tensorflow-to-dlc",
    "tflite": "snpe-tflite-to-dlc",
    "pytorch": "snpe-pytorch-to-dlc",
}

# Unified converter (QAIRT 2.x+ — recommended for new projects)
UNIFIED_CONVERTER = "qairt-converter"


# ══════════════════════════════════════════════════════════════════════════════
# Enhanced Conversion Request
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class InputSpec:
    """Specification for a model input tensor.

    Used with --input_dim flag: --input_dim "name" "N,H,W,C"
    """
    name: str                   # Input tensor name (e.g. "input", "Placeholder")
    dimensions: tuple[int, ...]  # Shape (e.g. (1, 299, 299, 3))

    @property
    def dim_string(self) -> str:
        """Format for --input_dim flag: "1,299,299,3" """
        return ",".join(str(d) for d in self.dimensions)

    @property
    def cli_args(self) -> list[str]:
        """CLI arguments for legacy converter: --input_dim name "dims" """
        return ["--input_dim", self.name, self.dim_string]


@dataclass
class ConversionConfig:
    """Enhanced model conversion configuration.

    Supports all flags for both unified (qairt-converter) and legacy converters.
    """
    # Required
    model_path: str                  # Path to source model (.onnx, .pb, .tflite, .pt)
    source_framework: SourceFramework = SourceFramework.ONNX

    # Optional — input/output specification
    input_specs: list[InputSpec] = field(default_factory=list)  # --input_dim per input
    output_nodes: list[str] = field(default_factory=list)       # --out_node (TF required)
    output_path: str = ""            # --output_path (auto-generated if empty)

    # Quantization
    quantization: str = "fp32"       # fp32, int8, int4 (int8/4 → separate qairt-quantizer step)
    float_bitwidth: int = 32         # --float_bitwidth 16 or 32

    # UDO support
    udo_config_paths: list[str] = field(default_factory=list)  # --udo_config_paths

    # TensorFlow specific
    tf_input_format: TensorFlowInputFormat = TensorFlowInputFormat.FROZEN_PB

    # Additional flags
    allow_unconsumed_nodes: bool = False  # --allow_unconsumed_nodes (legacy TF, e.g. MobilenetSSD)
    input_type: str = ""                  # --input_type name type (legacy only)
    input_color_encoding: str = ""        # --input_color_encoding (maps to QAIRT flag when set)

    # Advanced
    use_unified_converter: bool = True  # True = qairt-converter, False = legacy
    dry_run: bool = False               # --dry_run (validate without converting)
    extra_args: list[str] = field(default_factory=list)  # Additional CLI args

    @property
    def output_dlc_path(self) -> str:
        """Determine output .dlc path (auto-generate from model name if not specified).

        Always returns a POSIX-style path so the result is identical on
        Windows and Linux (the path is consumed by SDK tools that
        ultimately run on POSIX targets).
        """
        if self.output_path:
            return self.output_path
        return Path(self.model_path).with_suffix(".dlc").as_posix()

    def build_cli_args(self) -> list[str]:
        """Build the complete CLI argument list for the converter.

        Returns the full argument list (excluding the tool binary itself).
        """
        args: list[str] = []

        if self.use_unified_converter:
            # Unified converter (qairt-converter)
            args += ["--input_network", self.model_path]

            # Float bitwidth (FP16 conversion)
            if self.float_bitwidth == 16:
                args += ["--float_bitwidth", "16"]

        else:
            # Legacy per-framework converter
            if self.source_framework == SourceFramework.TENSORFLOW:
                args += ["--input_network", self.model_path]

                # TF requires --input_dim and --out_node
                for inp in self.input_specs:
                    args += inp.cli_args

                for node in self.output_nodes:
                    args += ["--out_node", node]

                args += ["--output_path", self.output_dlc_path]

            elif self.source_framework == SourceFramework.ONNX:
                args += ["--input_network", self.model_path]
                args += ["--output_path", self.output_dlc_path]

            elif self.source_framework == SourceFramework.TFLITE:
                args += ["--input_network", self.model_path]
                args += ["--output_path", self.output_dlc_path]

            elif self.source_framework == SourceFramework.PYTORCH:
                args += ["--input_network", self.model_path]
                for inp in self.input_specs:
                    args += inp.cli_args  # --input_dim name dims
                args += ["--output_path", self.output_dlc_path]

        # Common flags
        if self.udo_config_paths:
            for udo_path in self.udo_config_paths:
                args += ["--udo_config_paths", udo_path]

        # Legacy-only extra flags
        if not self.use_unified_converter:
            if self.allow_unconsumed_nodes:
                args.append("--allow_unconsumed_nodes")
            if self.input_type:
                # --input_type applies to each input spec or as a global type
                for inp in self.input_specs:
                    args += ["--input_type", inp.name, self.input_type]
            if self.input_color_encoding:
                args += ["--input_color_encoding", self.input_color_encoding]

        if self.dry_run:
            args.append("--dry_run")

        args.extend(self.extra_args)
        return args

    @property
    def converter_tool(self) -> str:
        """Get the converter tool name."""
        if self.use_unified_converter:
            return UNIFIED_CONVERTER
        return LEGACY_CONVERTERS.get(self.source_framework.value, UNIFIED_CONVERTER)

    def validate(self) -> list[str]:
        """Validate the conversion config, return list of errors (empty = valid)."""
        errors: list[str] = []

        if not self.model_path:
            errors.append("model_path is required")

        # TensorFlow requires output nodes with legacy converter
        if (self.source_framework == SourceFramework.TENSORFLOW
            and not self.use_unified_converter
            and not self.output_nodes):
            errors.append(
                "TensorFlow conversion requires --out_node. "
                "Specify output_nodes=['NodeName'] or use unified converter."
            )

        # PyTorch legacy converter requires --input_dim
        if (self.source_framework == SourceFramework.PYTORCH
                and not self.use_unified_converter
                and not self.input_specs):
            errors.append(
                "PyTorch legacy converter requires --input_dim. "
                "Add input_specs=[InputSpec('input', (1,3,224,224))]"
            )

        # Check file extension matches framework
        ext = Path(self.model_path).suffix.lower()
        expected_exts = {
            SourceFramework.ONNX: [".onnx"],
            SourceFramework.TENSORFLOW: [".pb", ".meta", ""],  # "" for SavedModel dir
            SourceFramework.TFLITE: [".tflite"],
            SourceFramework.PYTORCH: [".pt", ".pth"],
        }
        if ext and ext not in expected_exts.get(self.source_framework, []):
            errors.append(
                f"File extension '{ext}' doesn't match framework "
                f"'{self.source_framework.value}'. "
                f"Expected: {expected_exts.get(self.source_framework)}"
            )

        return errors


# ══════════════════════════════════════════════════════════════════════════════
# TensorFlow → SNPE Layer Mapping Reference
# ══════════════════════════════════════════════════════════════════════════════

# Documents how TensorFlow graph nodes map to SNPE layers
# Key rules:
#   - All nodes belonging to a layer must be in a UNIQUE TensorFlow scope
#   - A node can only belong to a single layer

TF_LAYER_MAPPINGS: dict[str, dict[str, Any]] = {
    "BatchNormalization": {
        "tf_ops": ["FusedBatchNorm", "FusedBatchNormV3", "batch_normalization"],
        "snpe_layer": "Batchnorm",
        "note": "Must be in unique scope",
    },
    "Conv2d": {
        "tf_ops": ["Conv2D", "conv2d"],
        "snpe_layer": "Conv2d",
        "note": "Supports both NHWC and NCHW (converts to NHWC internally)",
    },
    "Concatenation": {
        "tf_ops": ["Concat", "ConcatV2"],
        "snpe_layer": "Concat",
        "note": "Axis parameter determines concatenation dimension",
    },
    "Deconvolution": {
        "tf_ops": ["Conv2DTranspose", "conv2d_transpose"],
        "snpe_layer": "TransposeConv2d",
        "note": "Also known as transposed convolution",
    },
    "ElementWise": {
        "tf_ops": ["Add", "AddV2", "Mul", "Maximum", "Sum"],
        "snpe_layer": "ElementWiseAdd/Mul/Max",
        "note": "Broadcasting supported",
    },
    "FullyConnected": {
        "tf_ops": ["MatMul", "dense", "tensordot"],
        "snpe_layer": "FullyConnected",
        "note": "Weight matrix must be constant/static",
    },
    "LRN": {
        "tf_ops": ["LRN", "local_response_normalization"],
        "snpe_layer": "Lrn",
        "note": "Local Response Normalization",
    },
    "PoolingAvg": {
        "tf_ops": ["AvgPool", "average_pooling2d"],
        "snpe_layer": "PoolAvg2d",
        "note": "Supports padding modes SAME and VALID",
    },
    "PoolingMax": {
        "tf_ops": ["MaxPool", "max_pooling2d"],
        "snpe_layer": "PoolMax2d",
        "note": "Supports padding modes SAME and VALID",
    },
    "Relu": {
        "tf_ops": ["Relu", "relu"],
        "snpe_layer": "Relu",
        "note": "Also supports Relu6, LeakyRelu",
    },
    "Sigmoid": {
        "tf_ops": ["Sigmoid", "sigmoid"],
        "snpe_layer": "Sigmoid",
        "note": "",
    },
    "Tanh": {
        "tf_ops": ["Tanh", "tanh"],
        "snpe_layer": "Tanh",
        "note": "",
    },
    "Elu": {
        "tf_ops": ["Elu", "elu"],
        "snpe_layer": "Elu",
        "note": "",
    },
    "Softmax": {
        "tf_ops": ["Softmax", "softmax"],
        "snpe_layer": "Softmax",
        "note": "",
    },
    "PReLU": {
        "tf_ops": ["PReLU"],
        "snpe_layer": "Prelu",
        "note": "TFLearn PReLU reference",
    },
    "Slice": {
        "tf_ops": ["Slice", "strided_slice"],
        "snpe_layer": "StridedSlice",
        "note": "",
    },
    "Reshape": {
        "tf_ops": ["Reshape", "reshape"],
        "snpe_layer": "Reshape",
        "note": "",
    },
}


def get_tf_layer_mapping(tf_op_name: str) -> dict[str, Any] | None:
    """Look up SNPE layer mapping for a TensorFlow operation.

    Args:
        tf_op_name: TensorFlow operation type (e.g. "Conv2D", "MatMul")

    Returns:
        Mapping dict with snpe_layer, note, etc. or None if not found.
    """
    for layer_name, mapping in TF_LAYER_MAPPINGS.items():
        if tf_op_name in mapping["tf_ops"]:
            return {"layer": layer_name, **mapping}
    return None


def get_all_supported_tf_ops() -> list[str]:
    """Get all TensorFlow operations that can be converted to SNPE layers."""
    ops: list[str] = []
    for mapping in TF_LAYER_MAPPINGS.values():
        ops.extend(mapping["tf_ops"])
    return sorted(set(ops))

# ══════════════════════════════════════════════════════════════════════════════
# ONNX Model Conversion Notes
# ══════════════════════════════════════════════════════════════════════════════
#
# From SNPE ONNX Model Conversion documentation:
#
# 1. SYMBOLIC TENSOR SHAPES NOT SUPPORTED:
#    Neither snpe-onnx-to-dlc nor the runtime support symbolic shape variables.
#    Use Network Resizing (setInputDimensions / --input_dimensions) instead.
#
# 2. DATA TYPES FROM ONNX MODEL ARE IGNORED:
#    SNPE determines data types based on runtime needs and builder parameters.
#    The data types specified in the ONNX model will usually be ignored.
#    (This means you don't need to worry about ONNX model precision — SNPE
#    handles FP32/FP16/INT8 based on quantization settings, not model spec.)
#
# 3. ONNX FUNCTIONS ARE ALWAYS INLINED:
#    If the model contains ONNX functions, the converter always inlines
#    function nodes (expands them into their constituent operations).
#
# 4. SUPPORTED OPS:
#    See Supported ONNX Ops (130+ operators, up to Opset 24).
#    Check with: from quad.utils.layer_support import get_supported_ops
#
# 5. WINDOWS POWERSHELL NOTE:
#    Run converter via python in activated venv:
#    (venv-3.10) > python snpe-onnx-to-dlc <options>

ONNX_CONVERSION_NOTES = {
    "symbolic_shapes": (
        "ONNX symbolic tensor shape variables are NOT supported by the converter or runtime. "
        "Use Network Resizing (quad.load(..., input_dimensions={...})) to set dynamic shapes."
    ),
    "data_types_ignored": (
        "Data types specified in the ONNX model are usually IGNORED. "
        "SNPE determines types based on runtime needs and builder parameters. "
        "Use quantization (INT8/INT4) via qairt-quantizer for fixed-point execution."
    ),
    "functions_inlined": (
        "ONNX functions are always inlined during conversion. "
        "Function nodes are expanded into their constituent operations."
    ),
    "windows_powershell": (
        "On Windows PowerShell, run converter via python in activated venv: "
        "(venv-3.10) > python snpe-onnx-to-dlc <options>"
    ),
}


def get_onnx_conversion_warnings(model_path: str) -> list[str]:
    """Check an ONNX model for potential conversion issues.

    Returns warnings about:
    - Symbolic shapes (must use network resizing instead)
    - Data types that will be overridden
    - Functions that will be inlined

    In mock mode (no onnx package): returns general guidance.
    In real mode: inspects the actual model file.
    """
    warnings: list[str] = []

    try:
        import onnx
        model = onnx.load(model_path)
        graph = model.graph

        # Check for symbolic dimensions
        for inp in graph.input:
            if inp.type.tensor_type.shape:
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_param:  # Symbolic (e.g. "batch_size", "N")
                        warnings.append(
                            f"Input '{inp.name}' has symbolic dimension '{dim.dim_param}'. "
                            f"SNPE does not support symbolic shapes. Use input_dimensions "
                            f"parameter to set concrete dimensions at load time."
                        )

        # Check for ONNX functions
        if model.functions:
            warnings.append(
                f"Model contains {len(model.functions)} ONNX function(s). "
                f"These will be inlined (expanded) during conversion."
            )

    except ImportError:
        # onnx package not available — return general guidance
        warnings.append(ONNX_CONVERSION_NOTES["symbolic_shapes"])
    except Exception:
        pass  # File doesn't exist or isn't valid ONNX — skip checks

    return warnings

# ══════════════════════════════════════════════════════════════════════════════
# TFLite Model Conversion Notes
# ══════════════════════════════════════════════════════════════════════════════
#
# From SNPE TFLite Model Conversion documentation:
#
# Tool: snpe-tflite-to-dlc --input_network model.tflite --input_dim input "1,299,299,3"
#
# 1. ONLY FLOAT INPUT DATA TYPES SUPPORTED:
#    SNPE and TFLite converter currently only support float input data types.
#    Quantized TFLite models need to have float inputs (quantization is internal).
#
# 2. MLIR-BASED TFLITE CONVERTER ISSUES:
#    Some older versions of MLIR-based TFLite converter produce models that
#    fail to load. If you encounter loading failures, try:
#    - Upgrading TensorFlow/TFLite version
#    - Using the non-MLIR converter (legacy)
#
# 3. SOURCE:
#    TF model → TFLite: https://www.tensorflow.org/lite/convert#python_api_
#    TFHub models: https://tfhub.dev/
#
# 4. --input_dim IS SUPPORTED (same syntax as TensorFlow converter):
#    --input_dim "input_name" "N,H,W,C"

TFLITE_CONVERSION_NOTES = {
    "float_inputs_only": (
        "SNPE and TFLite converter currently only support FLOAT input data types. "
        "Quantized TFLite models must still have float inputs (internal quantization only)."
    ),
    "mlir_issues": (
        "Some older versions of MLIR-based TFLite converter can lead to failure "
        "loading the model. Upgrade TensorFlow or use the non-MLIR converter if "
        "you encounter loading issues."
    ),
    "input_dim_supported": (
        "TFLite converter supports --input_dim to specify input tensor dimensions: "
        "snpe-tflite-to-dlc --input_network model.tflite --input_dim input \"1,299,299,3\""
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Model Conversion Notes
# ══════════════════════════════════════════════════════════════════════════════
#
# From SNPE PyTorch Model Conversion documentation:
#
# Tool: snpe-pytorch-to-dlc --input_network model.pt --input_dim input "1,3,224,224"
#
# 1. TORCHSCRIPT FORMAT REQUIRED:
#    PyTorch models must be converted to TorchScript (.pt) before SNPE conversion.
#    Use torch.jit.trace() to convert:
#      script_model = torch.jit.trace(model, input_data)
#      script_model.save("model.pt")
#
# 2. ONLY FLOAT INPUT DATA TYPES SUPPORTED:
#    Same as TFLite — only float inputs accepted by the converter.
#
# 3. --input_dim REQUIRED:
#    Must specify input dimensions: --input_dim "input" "N,C,H,W"
#    Note PyTorch uses NCHW layout (not NHWC like TensorFlow).
#
# 4. TRACING vs SCRIPTING:
#    torch.jit.trace: works for models with fixed control flow
#    torch.jit.script: needed for dynamic control flow (if/for)
#    SNPE converter works with both, but trace is more reliable.
#
# 5. CONVERSION CODE EXAMPLE:
#    import torch
#    import torchvision.models as models
#    model = models.resnet18()
#    input_data = torch.randn(1, 3, 224, 224)
#    script_model = torch.jit.trace(model, input_data)
#    script_model.save("resnet18.pt")

PYTORCH_CONVERSION_NOTES = {
    "torchscript_deprecated": (
        "WARNING: TorchScript (torch.jit.trace/torch.jit.script) is DEPRECATED. "
        "Use torch.export instead: https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html "
        "However, snpe-pytorch-to-dlc still requires TorchScript .pt files. "
        "For new projects: export to ONNX via torch.onnx.export() and use snpe-onnx-to-dlc or qairt-converter."
    ),
    "torchscript_legacy": (
        "Legacy path (still supported by snpe-pytorch-to-dlc): "
        "script_model = torch.jit.trace(model, sample_input); script_model.save('model.pt') "
        "NOTE: This path is deprecated. Prefer ONNX export for new projects."
    ),
    "float_inputs_only": (
        "SNPE PyTorch converter currently only supports FLOAT input data types."
    ),
    "input_dim_required": (
        "Must specify --input_dim with PyTorch converter. "
        "Note: PyTorch uses NCHW layout (not NHWC like TensorFlow). "
        "Example: --input_dim input \"1,3,224,224\""
    ),
    "nchw_layout": (
        "PyTorch models use NCHW layout (Batch, Channels, Height, Width). "
        "SNPE internally converts to NHWC. Specify dimensions in NCHW order."
    ),
    "trace_vs_script": (
        "torch.jit.trace: fixed control flow (recommended, more reliable with SNPE). "
        "torch.jit.script: dynamic control flow (if/for). Both work but trace preferred."
    ),
}


def generate_onnx_export_code(
    model_import: str = "torchvision.models.resnet18(pretrained=True)",
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    output_path: str = "model.onnx",
) -> str:
    """Generate Python code to export a PyTorch model to ONNX format.

    RECOMMENDED: Use ONNX export instead of TorchScript (which is deprecated).
    The exported ONNX model can then be converted with:
      qairt-converter --input_network model.onnx
    or:
      snpe-onnx-to-dlc --input_network model.onnx

    See: https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html
    """
    shape_str = ", ".join(str(d) for d in input_shape)
    return f"""import torch
import torch.onnx

# Load/create the model
model = {model_import}
model.eval()

# Create sample input (NCHW format for PyTorch)
dummy_input = torch.randn({shape_str})

# Export to ONNX (RECOMMENDED over TorchScript)
torch.onnx.export(
    model,
    dummy_input,
    "{output_path}",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={{"input": {{0: "batch_size"}}, "output": {{0: "batch_size"}}}},
)
print(f"Exported ONNX model to: {output_path}")
print("Convert with: qairt-converter --input_network {output_path}")
"""


def generate_torchscript_export_code(
    model_import: str = "torchvision.models.resnet18()",
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    output_path: str = "model.pt",
) -> str:
    """Generate Python code to convert a PyTorch model to TorchScript.

    Returns a complete Python script string that can be executed to produce
    the .pt file required by snpe-pytorch-to-dlc.

    Args:
        model_import: Python expression to create the model
        input_shape: Input tensor shape in NCHW format
        output_path: Output .pt file path
    """
    shape_str = ", ".join(str(d) for d in input_shape)
    return f"""import torch

# Load/create the model
model = {model_import}
model.eval()  # Set to evaluation mode

# Create sample input for tracing (NCHW format)
input_data = torch.randn({shape_str})

# Convert to TorchScript via tracing
script_model = torch.jit.trace(model, input_data)

# Save the TorchScript model
script_model.save("{output_path}")
print(f"Saved TorchScript model to: {output_path}")
"""

# ══════════════════════════════════════════════════════════════════════════════
# Quantization Notes
# ══════════════════════════════════════════════════════════════════════════════
#
# From SNPE "Quantizing a Model" documentation:
#
# Tool: snpe-dlc-quantize --input_dlc model.dlc --input_list images.txt --output_dlc q.dlc
# (also: snpe-dlc-quant and qairt-quantizer — same role, slightly different flags)
#
# 1. BATCH DIMENSION MUST BE 1 DURING CONVERSION:
#    snpe-dlc-quantize requires batch_size=1 in the DLC.
#    Use Network Resizing at inference time to change batch size.
#
# 2. INPUT DATA REQUIREMENTS:
#    - 5-10 examples: sufficient for quick experiments
#    - 50-100 examples: recommended for robust quantization
#    - Must be REPRESENTATIVE (cover all output classes/modalities)
#    - Must NOT be from the training set
#    - Should include all input data modalities the model will encounter
#
# 3. QUANTIZATION GUARANTEES:
#    Only ops in the "Supported Network Layers" table are GUARANTEED to
#    quantize successfully. Other ops have no guarantee.
#
# 4. WORKFLOW:
#    Step 1: convert  → snpe-onnx-to-dlc   → model.dlc       (float)
#    Step 2: quantize → snpe-dlc-quantize  → model_q.dlc     (int8)
#    Step 3: prepare  → snpe-dlc-graph-prepare (optional, HTP offline cache)

QUANTIZATION_NOTES = {
    "batch_must_be_1": (
        "The DLC batch dimension MUST be set to 1 during model conversion. "
        "Resize at inference time using Network Resizing (--input_dimensions) "
        "if a different batch size is needed."
    ),
    "input_data_quick": (
        "5-10 representative input examples sufficient for quick experiments."
    ),
    "input_data_robust": (
        "50-100 representative examples recommended for robust quantization. "
        "Must cover all output classes/modalities. Must NOT come from training set."
    ),
    "op_guarantees": (
        "Only ops listed in 'Supported Network Layers' are GUARANTEED to quantize. "
        "Other ops (e.g. custom/rare ops) have no quantization guarantee. "
        "Check quad.utils.layer_support for the supported ops list."
    ),
    "workflow": (
        "Quantization workflow: "
        "1. Convert: snpe-onnx-to-dlc → model.dlc (float) "
        "2. Quantize: snpe-dlc-quantize → model_quantized.dlc (int8) "
        "3. (Optional) Cache: snpe-dlc-graph-prepare → model_cached.dlc (HTP)"
    ),
}

# Recommended calibration data sizes
QUANTIZATION_CALIBRATION_MIN = 5      # Minimum for quick experiments
QUANTIZATION_CALIBRATION_RECOMMENDED = 50   # Recommended lower bound
QUANTIZATION_CALIBRATION_ROBUST = 100       # For production / max accuracy


def build_quantize_cli_args(
    input_dlc: str,
    input_list: str,
    output_dlc: str = "",
    weights_bitwidth: int = 8,
    act_bitwidth: int = 8,
    use_per_channel: bool = False,
    algorithms: list[str] | None = None,
    htp_socs: str = "",
    udo_package_path: str = "",
    use_unified: bool = True,
) -> list[str]:
    """Build CLI args for snpe-dlc-quantize / snpe-dlc-quant / qairt-quantizer.

    Args:
        input_dlc: Path to non-quantized DLC (batch must be 1).
        input_list: Path to representative input data list (5-100 files).
        output_dlc: Output path (auto: input_name_quantized.dlc).
        weights_bitwidth: 4, 8 (default), or 16.
        act_bitwidth: 8 (default) or 16.
        use_per_channel: Per-channel quantization for conv/deconv/FC.
        algorithms: e.g. ["cle"] for cross-layer equalization.
        htp_socs: e.g. "sm8650" — pack HTP offline cache in DLC.
        udo_package_path: Registration lib for UDO packages.
        use_unified: True → qairt-quantizer flags; False → snpe-dlc-quantize.

    Returns:
        Complete CLI argument list (excluding tool binary).
    """
    args: list[str] = []

    if use_unified:
        # qairt-quantizer flags
        args += ["--input_dlc", input_dlc]
        args += ["--input_list", input_list]
        if output_dlc:
            args += ["--output_dlc", output_dlc]
        args += ["--weights_bitwidth", str(weights_bitwidth)]
        args += ["--act_bitwidth", str(act_bitwidth)]
        if use_per_channel:
            args.append("--use_per_channel_quantization")
        for algo in (algorithms or []):
            args += ["--apply_algorithms", algo]
    else:
        # snpe-dlc-quantize flags
        args += ["--input_dlc", input_dlc]
        args += ["--input_list", input_list]
        if output_dlc:
            args += ["--output_dlc", output_dlc]
        args += ["--weights_bitwidth", str(weights_bitwidth)]
        args += ["--act_bitwidth", str(act_bitwidth)]
        if use_per_channel:
            args.append("--use_per_channel_quantization")
        for algo in (algorithms or []):
            args += ["--algorithms", algo]
        if htp_socs:
            args += ["--enable_htp", "--htp_socs", htp_socs]

    if udo_package_path:
        args += ["--udo_package_path", udo_package_path]

    return args

# ══════════════════════════════════════════════════════════════════════════════
# Offline Graph Caching for DSP/HTP
# ══════════════════════════════════════════════════════════════════════════════
#
# From SNPE "Offline Graph Caching for DSP Runtime on HTP" documentation.
#
# Workflow (4 steps on HOST x86-64, result deployed to device):
#   1. Convert:  snpe-onnx-to-dlc → model.dlc           (float)
#   2. Quantize: snpe-dlc-quant   → model_q.dlc         (int8)
#   3. Prepare:  snpe-dlc-graph-prepare → model_cache.dlc (int8 + HTP cache)
#   4. Execute:  snpe-net-run on device                   (fast init)
#
# snpe-dlc-graph-prepare is PREFERRED over snpe-dlc-quantize --enable_htp
# (the --enable_htp path in snpe-dlc-quantize is DEPRECATED).
#
# KEY RULES:
#
# 1. OUTPUT MATCHING: cache and runtime MUST specify the same graph outputs.
#    If they differ, cache is rejected → online prepare (slow init).
#    Specify outputs: --set_output_layers OR --set_output_tensors
#    In input_list: "#layername" (layer) or "%tensorname" (tensor)
#
# 2. CACHE SoC COMPATIBILITY:
#    - Newer arch → will NOT run on older arch
#      (v69 cache ≠ run on v68 device)
#    - v68/v69 cache will NOT run on v73 device
#    - Same arch: compatible if prepared VTCM ≤ running SoC VTCM
#
# 3. INPUT DIMENSIONS: cache is tied to the dimensions it was prepared with.
#    Resized caches only work if runtime uses the SAME dimensions.
#
# 4. OPTIMIZATION LEVELS (--optimization_level):
#    Level 2 (default): faster prepare, good performance
#    Level 3:           slower prepare, usually better perf, sometimes worse
#    Level 3 may produce larger cache → possible degradation in init time.
#
# 5. FLOAT MODEL: snpe-dlc-graph-prepare also works with float DLC
#    to generate cache for HTP FP16 runtime (no quantization needed).

OFFLINE_CACHE_NOTES = {
    "workflow": (
        "4-step HTP offline caching workflow (all on host x86-64): "
        "1. snpe-onnx-to-dlc → float DLC  "
        "2. snpe-dlc-quant → quantized DLC  "
        "3. snpe-dlc-graph-prepare --htp_socs sm8750 → DLC with HTP cache  "
        "4. Deploy + snpe-net-run on device (fast init)"
    ),
    "output_matching": (
        "Cache and runtime MUST use the SAME graph outputs. "
        "Mismatch → cache rejected → slow online prepare. "
        "Use --set_output_layers or --set_output_tensors to pin outputs."
    ),
    "soc_compatibility": (
        "Compatibility rules: "
        "(a) Newer DSP arch cache will NOT run on older arch devices. "
        "(b) v68/v69 cache will NOT run on v73 devices. "
        "(c) Same arch: OK if prepared VTCM <= running SoC VTCM."
    ),
    "input_dimensions": (
        "Cache records are tied to the input dimensions used during preparation. "
        "Resized networks must use the SAME dimensions at inference time. "
        "Use --input_name and --input_dimensions to prepare for non-default sizes."
    ),
    "optimization_level": (
        "Level 2 (default): faster prepare. "
        "Level 3: usually better runtime perf but slower prepare + larger cache. "
        "Level 3 may INCREASE init time due to larger cache size."
    ),
    "graph_prepare_preferred": (
        "snpe-dlc-graph-prepare is PREFERRED. "
        "snpe-dlc-quantize --enable_htp is DEPRECATED."
    ),
}

# Cache SoC compatibility matrix (from docs empirical rules)
# Key: (prepared_arch, running_arch) → compatible
_CACHE_COMPAT = {
    ("v68", "v68"): True,
    ("v68", "v69"): False,   # v68 → v69 OK (newer can run older cache)... actually docs say newer arch CANNOT run on OLDER arch
    ("v68", "v73"): False,   # explicitly stated: v68/v69 cache won't run on v73
    ("v69", "v68"): False,   # newer arch cache on older arch = NOT compatible
    ("v69", "v69"): True,
    ("v69", "v73"): False,   # explicitly stated
    ("v73", "v68"): False,   # v73 on v68 not compatible (v73 is newer)
    ("v73", "v69"): False,
    ("v73", "v73"): True,
    ("v75", "v73"): False,
    ("v75", "v75"): True,
    ("v79", "v75"): False,
    ("v79", "v79"): True,
}


def is_cache_compatible(prepared_arch: str, running_arch: str) -> bool | None:
    """Check if an HTP cache prepared for one arch can run on another.

    Implements the compatibility rules from SNPE offline caching docs:
    - Newer arch cache cannot run on older arch device
    - v68/v69 cache will not run on v73 device
    - Same arch: compatible (assuming VTCM fits)

    Args:
        prepared_arch: Architecture used for cache preparation (e.g. "v73")
        running_arch: Architecture of the target device (e.g. "v75")

    Returns:
        True = compatible, False = incompatible, None = unknown combination
    """
    # Normalize
    prep = prepared_arch.lower().replace("hexagon-", "")
    run = running_arch.lower().replace("hexagon-", "")

    if prep == run:
        return True

    # Check explicit rules
    result = _CACHE_COMPAT.get((prep, run))
    if result is not None:
        return result

    # General rule: newer arch cache CANNOT run on older arch device
    arch_order = ["v65", "v66", "v68", "v69", "v73", "v75", "v79", "v81"]
    try:
        prep_idx = arch_order.index(prep)
        run_idx = arch_order.index(run)
        # Cache prepared for NEWER arch → won't run on OLDER device
        if prep_idx > run_idx:
            return False
        # Cache prepared for OLDER arch → MAY run on NEWER device (same features)
        return True
    except ValueError:
        return None  # Unknown arch


def build_graph_prepare_cli_args(
    input_dlc: str,
    output_dlc: str = "",
    htp_socs: str = "sm8750",
    optimization_level: int = 2,
    set_output_layers: list[str] | None = None,
    set_output_tensors: list[str] | None = None,
    input_name: str = "",
    input_dimensions: str = "",
    vtcm_override: int = 0,
) -> list[str]:
    """Build CLI args for snpe-dlc-graph-prepare.

    Generates HTP offline cache for the DLC. Must be run on x86-64 Linux host.

    Args:
        input_dlc: Path to quantized DLC (or float DLC for FP16 cache).
        output_dlc: Output path (default: input_name_cached.dlc).
        htp_socs: Target SoC(s) e.g. "sm8750" or "sm8350,sm8450,sm8550".
        optimization_level: 2 (default) or 3 (better perf, slower prepare).
        set_output_layers: Layer names whose ALL outputs = graph outputs.
        set_output_tensors: Specific tensor names = graph outputs.
            IMPORTANT: Must match runtime output specification exactly.
        input_name: Input name for resized networks.
        input_dimensions: New dims e.g. "1,224,224,3" (paired with input_name).
        vtcm_override: Override VTCM size in MB (0 = use SoC max).
    """
    args = [f"--input_dlc={input_dlc}"]

    if output_dlc:
        args.append(f"--output_dlc={output_dlc}")

    args.append(f"--htp_socs={htp_socs}")
    args.append(f"--optimization_level={optimization_level}")

    for layer in (set_output_layers or []):
        args.append(f"--set_output_layers={layer}")

    for tensor in (set_output_tensors or []):
        args.append(f"--set_output_tensors={tensor}")

    if input_name and input_dimensions:
        args.append(f"--input_name={input_name}")
        args.append(f"--input_dimensions={input_dimensions}")

    if vtcm_override:
        args.append(f"--vtcm_override={vtcm_override}")

    return args

# ══════════════════════════════════════════════════════════════════════════════
# QAIRT Converter — Differences from Legacy Converters
# ══════════════════════════════════════════════════════════════════════════════
#
# qairt-converter is the UNIFIED converter for all frameworks.
# Key differences vs snpe-onnx-to-dlc / snpe-tensorflow-to-dlc / etc.:
#
# 1. AUTO-DETECTS FRAMEWORK from file extension (.onnx, .pb, .tflite, .pt)
#
# 2. LAYOUTS PRESERVED BY DEFAULT (BREAKING CHANGE):
#    Legacy: converts input to NHWC (spatial-first) by default
#    QAIRT:  preserves source layout (NCHW stays NCHW for ONNX/PyTorch)
#    → Models converted with qairt-converter WILL differ from legacy-converted
#
# 3. HTP IS DEFAULT BACKEND (affects optimization behaviors):
#    Legacy: backend left empty
#    QAIRT:  HTP set as default → HTP-specific optimizations apply
#    → e.g., IntBiasUpdates applied to FullyConnected when backend=HTP
#
# 4. QUANTIZER IS SEPARATE (qairt-quantizer is standalone):
#    Legacy: qnn-<fw>-converter invokes quantizer when --input_list is passed
#    QAIRT:  qairt-quantizer is a separate step (like snpe-dlc-quant)
#
# 5. RENAMED FLAGS:
#    --input_encoding → --input_color_encoding (BREAKING)
#    --define_symbol  → --onnx_define_symbol   (BREAKING)
#    --show_unconsumed_nodes → --tf_show_unconsumed_nodes
#    --signature_name → --tflite_signature_name
#    --input_dim      → --source_model_input_shape (TF/PyTorch)
#    --out_node       → --out_tensor_node
#
# 6. ONNX OUTPUT ORDER PRESERVED (legacy may reorder)
#    Legacy: use --preserve_onnx_output_order to keep order
#    QAIRT:  preserved by default
#
# 7. OUTPUT FORMAT: DLC only (.cpp/.bin/.json not supported)
#    For .cpp/.bin/.json output: use legacy qnn-<framework>-converter
#
# 8. DISCONNECTED INPUTS PRESERVED by default:
#    Legacy: may remove during constant folding
#    QAIRT:  all source inputs retained (use --remove_unused_inputs to remove)

QAIRT_CONVERTER_NOTES = {
    "layout_preserved": (
        "BREAKING: QAIRT converter PRESERVES source layout by default. "
        "Legacy converter converts inputs to NHWC. For ONNX/PyTorch models, "
        "the converted graph will DIFFER from legacy-converted graphs."
    ),
    "htp_default_backend": (
        "HTP is the default backend (legacy leaves it empty). "
        "This enables HTP-specific optimizations like IntBiasUpdates for FullyConnected."
    ),
    "quantizer_separate": (
        "Quantizer is a SEPARATE step with qairt-quantizer. "
        "Unlike legacy converters which invoke quantizer when --input_list is passed."
    ),
    "renamed_flags": {
        "--input_encoding": "--input_color_encoding",
        "--define_symbol": "--onnx_define_symbol",
        "--show_unconsumed_nodes": "--tf_show_unconsumed_nodes",
        "--signature_name": "--tflite_signature_name",
        "--input_dim": "--source_model_input_shape",
        "--out_node": "--out_tensor_node",
    },
    "output_order": (
        "ONNX output order is PRESERVED by default. "
        "Legacy: use --preserve_onnx_output_order to maintain order."
    ),
    "output_format": (
        "DLC only. Legacy .cpp/.bin/.json format is NOT supported. "
        "Use qnn-<framework>-converter for .cpp/.bin/.json output."
    ),
    "disconnected_inputs": (
        "All source inputs are PRESERVED by default (even disconnected). "
        "Use --remove_unused_inputs to remove disconnected nodes."
    ),
    "tf_required_args": (
        "TensorFlow REQUIRES: --source_model_input_shape AND --out_tensor_node. "
        "Example: qairt-converter --input_network model.pb "
        "--source_model_input_shape input 1,299,299,3 "
        "--out_tensor_node InceptionV3/Predictions/Reshape_1"
    ),
}


class ExportFormat(str):
    """Export format options for qairt-converter."""
    DLC_DEFAULT = "DLC_DEFAULT"          # Float→Float or Quant→Quant (preserves precision)
    DLC_STRIP_QUANT = "DLC_STRIP_QUANT"  # Strips quantization → float (for CPU/GPU runtimes)
    # Note: DLC_STRIP_QUANT may result in accuracy loss


@dataclass
class QAIRTConversionConfig:
    """Enhanced configuration for qairt-converter (unified converter).

    Use this instead of ConversionConfig for new projects.
    qairt-converter auto-detects framework from file extension.
    """
    model_path: str                      # Auto-detects framework from extension
    output_path: str = ""                # Optional; auto-named from input
    float_bitwidth: int = 32             # 32 (default) or 16
    float_bias_bitwidth: int = 0         # 0 (auto), 16, or 32
    export_format: str = ExportFormat.DLC_DEFAULT  # or DLC_STRIP_QUANT

    # I/O specification (required for TF; optional for ONNX/TFLite)
    input_shapes: list[InputSpec] = field(default_factory=list)    # --source_model_input_shape
    output_tensor_nodes: list[str] = field(default_factory=list)   # --out_tensor_node (TF required)

    # Layout customization
    io_config_yaml: str = ""             # --config path/to/io_config.yaml
    dump_config_template: str = ""       # --dump_config_template output/io_config.yaml

    # Quantization
    quantization_overrides: str = ""     # --quantization_overrides path/to/overrides.json
    remove_unused_inputs: bool = False   # --remove_unused_inputs

    # ONNX-specific (prefixed with --onnx_)
    onnx_skip_simplification: bool = False
    onnx_override_batch: int = 0
    onnx_define_symbols: dict[str, int] = field(default_factory=dict)  # name→value

    # TF-specific (prefixed with --tf_)
    tf_saved_model_tag: str = ""
    tf_saved_model_signature_key: str = ""

    # TFLite-specific
    tflite_signature_name: str = ""

    # Target backend (HTP is default)
    target_backend: str = "HTP"          # CPU, GPU, DSP, HTP, HTA, LPAI
    target_soc_model: str = ""           # e.g. "SM8750"

    # Legacy-compatibility fields (not used by qairt-converter itself)
    allow_unconsumed_nodes: bool = False   # Legacy flag only; QAIRT handles this automatically
    input_type: str = ""                   # Legacy flag only; kept for completeness

    # Other
    dry_run: bool = False
    enable_framework_trace: bool = False
    extra_args: list[str] = field(default_factory=list)

    @property
    def converter_tool(self) -> str:
        return UNIFIED_CONVERTER  # Always qairt-converter

    def build_cli_args(self) -> list[str]:
        """Build complete CLI arg list for qairt-converter."""
        args = ["--input_network", self.model_path]

        if self.output_path:
            args += ["--output_path", self.output_path]

        # Float bitwidth
        if self.float_bitwidth != 32:
            args += ["--float_bitwidth", str(self.float_bitwidth)]
        if self.float_bias_bitwidth:
            args += ["--float_bias_bitwidth", str(self.float_bias_bitwidth)]

        # Export format (strip quant for float runtimes)
        if self.export_format != ExportFormat.DLC_DEFAULT:
            args += ["--export_format", self.export_format]

        # Input shapes (--source_model_input_shape name dims)
        for spec in self.input_shapes:
            args += ["--source_model_input_shape", spec.name, spec.dim_string]

        # Output tensor nodes (TF required)
        for node in self.output_tensor_nodes:
            args += ["--out_tensor_node", node]

        # YAML config / template
        if self.io_config_yaml:
            args += ["--config", self.io_config_yaml]
        if self.dump_config_template:
            args += ["--dump_config_template", self.dump_config_template]

        # Quantization overrides
        if self.quantization_overrides:
            args += ["--quantization_overrides", self.quantization_overrides]

        if self.remove_unused_inputs:
            args.append("--remove_unused_inputs")

        # ONNX-specific
        if self.onnx_skip_simplification:
            args.append("--onnx_skip_simplification")
        if self.onnx_override_batch:
            args += ["--onnx_override_batch", str(self.onnx_override_batch)]
        for sym, val in self.onnx_define_symbols.items():
            args += ["--onnx_define_symbol", sym, str(val)]

        # TF-specific
        if self.tf_saved_model_tag:
            args += ["--tf_saved_model_tag", self.tf_saved_model_tag]
        if self.tf_saved_model_signature_key:
            args += ["--tf_saved_model_signature_key", self.tf_saved_model_signature_key]

        # TFLite-specific
        if self.tflite_signature_name:
            args += ["--tflite_signature_name", self.tflite_signature_name]

        # Backend
        if self.target_backend and self.target_backend != "HTP":
            args += ["--target_backend", self.target_backend]
        if self.target_soc_model:
            args += ["--target_soc_model", self.target_soc_model]

        if self.dry_run:
            args.append("--dry_run")
        if self.enable_framework_trace:
            args.append("--enable_framework_trace")

        args.extend(self.extra_args)
        return args

    def validate(self) -> list[str]:
        """Validate the QAIRT conversion config."""
        errors: list[str] = []

        if not self.model_path:
            errors.append("model_path is required")

        # TF requires input shapes and output nodes
        if self.model_path.endswith(".pb") and not self.input_shapes:
            errors.append(
                "TensorFlow (.pb) requires --source_model_input_shape. "
                "Add input_shapes=[InputSpec('name', (1,299,299,3))]"
            )
        if self.model_path.endswith(".pb") and not self.output_tensor_nodes:
            errors.append(
                "TensorFlow (.pb) requires --out_tensor_node. "
                "Add output_tensor_nodes=['NodeName']"
            )

        if self.float_bitwidth not in (16, 32):
            errors.append("float_bitwidth must be 16 or 32")

        return errors


def get_legacy_to_qairt_flag_mapping() -> dict[str, str]:
    """Return mapping of legacy converter flags to qairt-converter equivalents."""
    return QAIRT_CONVERTER_NOTES["renamed_flags"]

# ══════════════════════════════════════════════════════════════════════════════
# QAIRT Quantizer — Differences from Legacy (snpe-dlc-quant)
# ══════════════════════════════════════════════════════════════════════════════
#
# qairt-quantizer is the UNIFIED quantizer replacing snpe-dlc-quant / snpe-dlc-quantize.
#
# KEY DIFFERENCES vs legacy quantizers:
#
# 1. FILLS IN GAPS: qairt-quantizer generates calibration encodings only for
#    tensors that are MISSING encodings (not already set by QAT/overrides in
#    the converter step). snpe-dlc-quant quantizes all tensors.
#
# 2. HTP IS DEFAULT BACKEND (same as qairt-converter):
#    IntBiasUpdates optimization applied to FullyConnected for HTP.
#
# 3. NO-OPS (applied in converter stage now):
#    --ignore_quantization_overrides  → no-op (overrides applied in converter)
#    --enable_float_fallback          → no-op (fallback handled by converter)
#    Float fallback: missing tensors automatically fall back to float in converter.
#
# 4. WORKFLOW WITH GAPS:
#    Option A: qairt-converter (with overrides) → qairt-quantizer (fill gaps)
#    Option B: qairt-converter --export_format=DLC_STRIP_QUANT → qairt-quantizer
#              (strips all encodings, full re-calibration)
#    Option C: qairt-quantizer --input_list + --ignore_quantization_overrides
#              (ignore converter encodings, full re-calibration)
#
# 5. AIMET SUPPORT:
#    --use_aimet_quantizer: use AIMET instead of default quantizer
#    --apply_algorithms adaround: AdaRound (with or without YAML config)
#    --apply_algorithms amp: AMP (requires YAML config with candidates/eval)
#    Setup: source {SNPE_ROOT}/bin/aimet_env_setup.sh --env_path <venv> --aimet_sdk_tar <tar>
#    Min AIMET version: 1.33.0
#
# NOTE: --enable_float_fallback and --input_list are MUTUALLY EXCLUSIVE.
#       One is MANDATORY.

QAIRT_QUANTIZER_NOTES = {
    "fills_gaps": (
        "qairt-quantizer fills in MISSING encodings only. "
        "Tensors already quantized by QAT/overrides in converter are not re-calibrated."
    ),
    "noop_flags": (
        "--ignore_quantization_overrides and --enable_float_fallback are NO-OPS. "
        "These behaviors are now handled automatically by qairt-converter."
    ),
    "htp_default": (
        "HTP is the default backend. IntBiasUpdates optimization applied for FullyConnected."
    ),
    "float_fallback_alternative": (
        "For full re-calibration, use one of: "
        "(A) qairt-converter --export_format=DLC_STRIP_QUANT → qairt-quantizer --input_list, "
        "(B) qairt-quantizer --input_list + --ignore_quantization_overrides"
    ),
    "mutually_exclusive": (
        "--enable_float_fallback and --input_list are MUTUALLY EXCLUSIVE. One is mandatory."
    ),
    "aimet_setup": (
        "AIMET setup: source {SNPE_ROOT}/bin/aimet_env_setup.sh "
        "--env_path <venv_path> --aimet_sdk_tar <aimetpro-release-x.x.x.torch-xxx.tar.gz>. "
        "Min version: AIMET-1.33.0. Set AIMET_ENV_PYTHON to <venv>/bin/python."
    ),
}


def build_qairt_quantizer_args(
    input_dlc: str,
    output_dlc: str = "",
    input_list: str = "",
    weights_bitwidth: int = 8,
    act_bitwidth: int = 8,
    bias_bitwidth: int = 8,
    use_per_channel: bool = False,
    algorithms: list[str] | None = None,
    use_aimet: bool = False,
    aimet_config: str = "",
    ignore_quantization_overrides: bool = False,
    target_backend: str = "HTP",
    target_soc_model: str = "",
) -> list[str]:
    """Build CLI args for qairt-quantizer.

    Args:
        input_dlc: Non-quantized DLC (batch must be 1).
        output_dlc: Output path (auto: input_quantized.dlc).
        input_list: Calibration data list (50-100 representative samples).
            Mutually exclusive with enable_float_fallback.
        weights_bitwidth: 4, 8 (default), or 16.
        act_bitwidth: 8 (default) or 16.
        bias_bitwidth: 8 (default) or 32.
        use_per_channel: Per-channel quantization for conv/deconv/FC.
        algorithms: e.g. ["cle"] for cross-layer equalization,
                   ["adaround"] or ["amp"] for AIMET algorithms.
        use_aimet: Use AIMET quantizer instead of default.
            Requires aimet_env_setup.sh to be run first.
        aimet_config: Path to YAML config for AMP or AdaRound.
        ignore_quantization_overrides: Ignore converter encodings (full re-calibration).
        target_backend: CPU, GPU, DSP, HTP (default), HTA, LPAI.
        target_soc_model: e.g. "SM8750".
    """
    args = ["--input_dlc", input_dlc]

    if output_dlc:
        args += ["--output_dlc", output_dlc]

    if input_list:
        args += ["--input_list", input_list]

    args += ["--weights_bitwidth", str(weights_bitwidth)]
    args += ["--act_bitwidth", str(act_bitwidth)]
    args += ["--bias_bitwidth", str(bias_bitwidth)]

    if use_per_channel:
        args.append("--use_per_channel_quantization")

    for algo in (algorithms or []):
        args += ["--apply_algorithms", algo]

    if use_aimet:
        args.append("--use_aimet_quantizer")
    if aimet_config:
        args += ["--config", aimet_config]

    if ignore_quantization_overrides:
        args.append("--ignore_quantization_overrides")

    if target_backend and target_backend != "HTP":
        args += ["--target_backend", target_backend]
    if target_soc_model:
        args += ["--target_soc_model", target_soc_model]

    return args


def generate_aimet_amp_yaml(
    dataset_name: str = "calibration",
    dataloader_callback: str = "path/to/dataloader_fn",
    candidates: list | None = None,
    allowed_accuracy_drop: float = 0.02,
    eval_callback: str = "path/to/eval_fn",
) -> str:
    """Generate YAML config template for AIMET AMP algorithm.

    AMP = Automatic Mixed Precision — selects optimal bitwidth per layer.
    """
    if candidates is None:
        candidates = "[[[8, 'int'], [16, 'int']], [[16, 'float'], [16, 'float']]]"
    return f"""aimet_quantizer:
   datasets:
       {dataset_name}:
           dataloader_callback: '{dataloader_callback}'
           dataloader_kwargs: {{}}

   amp:
       dataset: {dataset_name}
       candidates: {candidates}
       allowed_accuracy_drop: {allowed_accuracy_drop}
       eval_callback_for_phase2: '{eval_callback}'
"""


def generate_aimet_adaround_yaml(
    dataset_name: str = "calibration",
    dataloader_callback: str = "path/to/dataloader_fn",
    num_batches: int = 1,
) -> str:
    """Generate YAML config template for AIMET AdaRound algorithm.

    AdaRound = Adaptive Rounding — learns optimal rounding for weights.
    Can also run without YAML by passing "adaround" to --apply_algorithms.
    """
    return f"""aimet_quantizer:
    datasets:
        {dataset_name}:
            dataloader_callback: '{dataloader_callback}'
            dataloader_kwargs: {{}}

    adaround:
        dataset: {dataset_name}
        num_batches: {num_batches}
"""

# ══════════════════════════════════════════════════════════════════════════════
# Model Tips — MobilenetSSD Example
# ══════════════════════════════════════════════════════════════════════════════
#
# From SNPE "Model Tips: Using MobilenetSSD" documentation.
# This is the canonical worked example for multi-output object detection models.
#
# CONVERSION COMMAND:
# snpe-tensorflow-to-dlc
#   --input_network frozen_inference_graph.pb
#   --input_dim Preprocessor/sub 1,300,300,3
#   --out_node detection_classes
#   --out_node detection_boxes
#   --out_node detection_scores
#   --output_path mobilenet_ssd.dlc
#   --allow_unconsumed_nodes
#
# OUTPUT LAYER NAMES:
#   Postprocessor/BatchMultiClassNonMaxSuppression
#   add
#
# OUTPUT BUFFER NAMES (with index offsets noted):
#   detection_classes:0    (+1 index offset from standard class indices)
#   Postprocessor/BatchMultiClassNonMaxSuppression_classes  (0 index offset)
#   Postprocessor/BatchMultiClassNonMaxSuppression_boxes
#   Postprocessor/BatchMultiClassNonMaxSuppression_scores
#
# RUNTIME LIMITATIONS:
# 1. Batch dimension > 1 is NOT supported.
# 2. DetectionOutput layer runs on CPU ONLY.
#    For GPU/DSP runtime: enable CPU fallback via runtime order:
#      snpe-net-run --runtime_order dsp,cpu
#      API: Snpe_SNPEBuilder_SetRuntimeProcessorOrder()
# 3. DetectionOutput performance: top_k has EXPONENTIAL impact.
#    top_k=100 << top_k=1000 (much faster with lower top_k).
#    Smaller confidence_threshold → more boxes → slower.
# 4. Input resizing NOT possible for MobilenetSSD.
#    PriorBox layer folding during conversion prevents resize
#    via Snpe_SNPEBuilder_SetInputDimensions().

MODEL_TIPS = {
    "deeplabv3": {
        "description": "DeepLabv3 semantic segmentation (MobileNet-v2 backbone, Pascal VOC 2012)",
        "input_name": "sub_7",
        "input_shape": (1, 513, 513, 3),
        "output_nodes": ["ArgMax"],
        "output_layers": ["ArgMax"],
        "output_buffers": {
            "segmentation_map": "ArgMax:0.raw",  # 513x513x1 integer class map
        },
        "model_url": "http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz",
        "limitations": [
            "Some ops CPU only — enable CPU fallback for GPU/DSP: --runtime_order dsp,cpu",
            "SNPE does NOT support model-internal preprocessing — must preprocess externally",
            "Output contains padding from preprocessing — must crop and resize back",
        ],
        "preprocessing": [
            # Step 1: Calculate resize ratio so longest side = 513
            "resize_ratio = 513.0 / max(width, height)",
            # Step 2: Resize with antialiasing (longer dim = 513, shorter < 513)
            "target_size = (int(resize_ratio * width), int(resize_ratio * height)); image = image.resize(target_size, Image.LANCZOS)",
            # Step 3: Pad shorter dimension to 513x513 with mean value 128
            "pad_h = 513 - target_size[1]; pad_w = 513 - target_size[0]; image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=128)",
            # Step 4: Convert to float32
            "image = image.astype(np.float32)",
            # Step 5: Multiply by normalization factor
            "image = image * 0.00784313771874",
            # Step 6: Subtract 1.0 (maps [0,255] → [-1, 1])
            "image = image - 1.0",
        ],
        "postprocessing": [
            # Step 1: Crop padding (restore original aspect ratio)
            "crop_h = int(resize_ratio * orig_height); crop_w = int(resize_ratio * orig_width); seg_map = seg_map[:crop_h, :crop_w]",
            # Step 2: Resize back to original image dimensions
            "seg_map = cv2.resize(seg_map, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)",
        ],
        "output_description": "513x513x1 integer segmentation map; each value is a class index (0=background)",
    },
    "mobilenet_ssd": {
        "description": "MobilenetSSD quantization-aware object detection model",
        "input_name": "Preprocessor/sub",
        "input_shape": (1, 300, 300, 3),
        "output_nodes": ["detection_classes", "detection_boxes", "detection_scores"],
        "allow_unconsumed_nodes": True,
        "output_layers": [
            "Postprocessor/BatchMultiClassNonMaxSuppression",
            "add",
        ],
        "output_buffers": {
            "classes_offset": "detection_classes:0",        # +1 index offset
            "classes_no_offset": "Postprocessor/BatchMultiClassNonMaxSuppression_classes",
            "boxes": "Postprocessor/BatchMultiClassNonMaxSuppression_boxes",
            "scores": "Postprocessor/BatchMultiClassNonMaxSuppression_scores",
        },
        "limitations": [
            "Batch > 1 NOT supported",
            "DetectionOutput layer: CPU only — enable CPU fallback for GPU/DSP",
            "top_k parameter has EXPONENTIAL impact on latency",
            "Input resizing NOT possible (PriorBox layer folding prevents it)",
        ],
        "performance_tips": [
            "Use top_k=100-200 for real-time inference (vs top_k=1000)",
            "Increase confidence_threshold to reduce boxes and improve speed",
            "Enable CPU fallback when using GPU/DSP: --runtime_order dsp,cpu",
        ],
        "model_url": "http://download.tensorflow.org/models/object_detection/"
                     "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz",
        "tested_tf_version": "1.11",
    }
}


def build_mobilenet_ssd_conversion_args(
    frozen_graph_path: str,
    output_dlc_path: str = "mobilenet_ssd.dlc",
    use_qairt: bool = False,
) -> list[str]:
    """Build conversion CLI args for MobilenetSSD.

    Args:
        frozen_graph_path: Path to exported frozen_inference_graph.pb
        output_dlc_path: Output .dlc path
        use_qairt: True = use qairt-converter (new), False = legacy snpe-tensorflow-to-dlc

    Returns:
        Complete CLI argument list (excluding tool binary).
    """
    tip = MODEL_TIPS["mobilenet_ssd"]

    if use_qairt:
        # qairt-converter uses --source_model_input_shape and --out_tensor_node
        args = ["--input_network", frozen_graph_path]
        args += ["--source_model_input_shape", tip["input_name"],
                 ",".join(str(d) for d in tip["input_shape"])]
        for node in tip["output_nodes"]:
            args += ["--out_tensor_node", node]
        args += ["--output_path", output_dlc_path]
    else:
        # Legacy: --input_dim and --out_node
        args = ["--input_network", frozen_graph_path]
        args += ["--input_dim", tip["input_name"],
                 ",".join(str(d) for d in tip["input_shape"])]
        for node in tip["output_nodes"]:
            args += ["--out_node", node]
        args += ["--output_path", output_dlc_path]
        if tip["allow_unconsumed_nodes"]:
            args.append("--allow_unconsumed_nodes")

    return args


def generate_mobilenet_ssd_export_script(
    pipeline_config: str = "<path>/pipeline.config",
    trained_checkpoint: str = "<path>/model.ckpt",
    export_dir: str = "<path>/exported",
    tfmodels_dir: str = "~/tfmodels",
) -> str:
    """Generate the shell script for exporting MobilenetSSD from checkpoint.

    Returns the export_train.sh script content from the SNPE docs.
    """
    return f"""#!/bin/bash
# Export MobilenetSSD trained graph for SNPE conversion
# From SNPE Model Tips documentation

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH={pipeline_config}
TRAINED_CKPT_PREFIX={trained_checkpoint}
EXPORT_DIR={export_dir}

pushd {tfmodels_dir}/models/research
python3 object_detection/export_inference_graph.py \
    --input_type=${{INPUT_TYPE}} \
    --pipeline_config_path=${{PIPELINE_CONFIG_PATH}} \
    --trained_checkpoint_prefix=${{TRAINED_CKPT_PREFIX}} \
    --output_directory=${{EXPORT_DIR}}
popd

echo "Exported frozen graph to: ${{EXPORT_DIR}}/frozen_inference_graph.pb"
"""

def build_deeplabv3_conversion_args(
    frozen_graph_path: str,
    output_dlc_path: str = "deeplabv3.dlc",
    use_qairt: bool = False,
) -> list[str]:
    """Build conversion CLI args for DeepLabv3 semantic segmentation model.

    Args:
        frozen_graph_path: Path to frozen_inference_graph.pb
        output_dlc_path: Output .dlc path
        use_qairt: True = qairt-converter; False = legacy snpe-tensorflow-to-dlc
    """
    tip = MODEL_TIPS["deeplabv3"]

    if use_qairt:
        args = ["--input_network", frozen_graph_path]
        args += ["--source_model_input_shape", tip["input_name"],
                 ",".join(str(d) for d in tip["input_shape"])]
        for node in tip["output_nodes"]:
            args += ["--out_tensor_node", node]
        args += ["--output_path", output_dlc_path]
    else:
        args = ["--input_network", frozen_graph_path]
        args += ["--input_dim", tip["input_name"],
                 ",".join(str(d) for d in tip["input_shape"])]
        for node in tip["output_nodes"]:
            args += ["--out_node", node]
        args += ["--output_path", output_dlc_path]

    return args


# ══════════════════════════════════════════════════════════════════════════════
# Input Image Formatting
# ══════════════════════════════════════════════════════════════════════════════

IMAGE_FORMAT_NOTES: dict[str, Any] = {
    "snpe_layout": {
        "name": "NHWC (channel-last, channel fastest-changing)",
        "description": (
            "SNPE requires input images in NHWC format: "
            "(batch × height × width × channel) where channel is the fastest-changing dimension. "
            "This means pixel data is stored as interleaved channels: "
            "[R0, G0, B0, R1, G1, B1, ...] for RGB images."
        ),
        "format": "batch × height × width × channel",
        "channel_stride": "fastest-changing (innermost)",
    },
    "pytorch_layout": {
        "name": "NCHW (channel-first, planar)",
        "description": (
            "PyTorch uses NCHW: (batch × channel × height × width). "
            "All R values come first, then all G values, then all B values. "
            "Must be transposed to NHWC before use with SNPE."
        ),
        "format": "batch × channel × height × width",
        "conversion": "np.transpose(image, (0, 2, 3, 1))  # NCHW → NHWC",
    },
    "channel_order": {
        "description": (
            "SNPE does NOT reorder channels. The channel order must MATCH what the model "
            "was trained with. Most OpenCV and Caffe-trained models (e.g. bvlc_alexnet, "
            "bvlc_googlenet) expect BGR. Most TensorFlow and modern PyTorch models expect RGB."
        ),
        "bgr_models": ["bvlc_alexnet", "bvlc_googlenet", "VGG", "Caffe-trained models"],
        "rgb_models": ["TensorFlow models", "most modern PyTorch models"],
        "conversion": "image_bgr = image_rgb[:, :, ::-1]  # RGB → BGR (reverses last axis)",
    },
    "batch_handling": {
        "description": (
            "For batch > 1, individual image files must be manually concatenated into a "
            "single binary file. SNPE reads batch inputs as one contiguous memory block."
        ),
        "concatenation": "cat image0.raw image1.raw image2.raw > batch_input.raw",
        "python_concat": "np.concatenate([img0, img1, img2], axis=0).tofile('batch_input.raw')",
        "note": "Each image must already be in NHWC format and correct channel order before concatenation.",
    },
    "mnist_example": {
        "description": (
            "Even single-channel images require a 4D tensor. "
            "MNIST: PyTorch shape (1, 1, 28, 28) → SNPE shape (1, 28, 28, 1)."
        ),
        "pytorch_shape": "(batch=1, channel=1, height=28, width=28)",
        "snpe_shape": "(batch=1, height=28, width=28, channel=1)",
        "conversion": "np.transpose(mnist_image, (0, 2, 3, 1))  # NCHW → NHWC",
    },
    "alexnet_example": {
        "description": (
            "AlexNet (bvlc_alexnet): 227×227 input, BGR channel order. "
            "PyTorch NCHW (1,3,227,227) → SNPE NHWC (1,227,227,3), then BGR reorder."
        ),
        "steps": [
            "1. Load image as RGB (H×W×3)",
            "2. Resize to 227×227",
            "3. Subtract mean pixel values: mean=[104, 117, 123] (BGR order)",
            "4. Convert to float32",
            "5. Reverse channels: image[:,:,::-1] (RGB→BGR)",
            "6. Write to raw binary: image.astype(np.float32).tofile('input.raw')",
        ],
        "output": "1000-class probability tensor (1×1000) — argmax gives predicted class",
    },
    "output_batch": {
        "description": (
            "For batch > 1, the output tensors from all batch items are concatenated "
            "along the batch dimension. E.g. AlexNet batch=2: output shape (2, 1000)."
        ),
    },
}


def convert_nchw_to_nhwc(image: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Convert image from PyTorch NCHW format to SNPE NHWC format.

    Args:
        image: NumPy array of shape (batch, channel, height, width)

    Returns:
        NumPy array of shape (batch, height, width, channel)

    Example:
        >>> import numpy as np
        >>> img = np.zeros((1, 3, 227, 227), dtype=np.float32)
        >>> snpe_img = convert_nchw_to_nhwc(img)
        >>> snpe_img.shape
        (1, 227, 227, 3)
    """
    import numpy as np
    arr = np.asarray(image)
    if arr.ndim == 3:
        # Single image without batch dim: (C, H, W) → (H, W, C)
        return np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 4:
        # Batched: (N, C, H, W) → (N, H, W, C)
        return np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")


def convert_channel_order(
    image: "np.ndarray",  # type: ignore[name-defined]
    from_order: str = "rgb",
    to_order: str = "bgr",
) -> "np.ndarray":  # type: ignore[name-defined]
    """Reorder image channels (e.g., RGB ↔ BGR).

    Args:
        image: NumPy array in NHWC or HWC format (channel is last axis)
        from_order: Source channel order ('rgb' or 'bgr')
        to_order: Target channel order ('rgb' or 'bgr')

    Returns:
        Image with channels reordered (same shape)

    Example:
        >>> import numpy as np
        >>> img_rgb = np.zeros((1, 227, 227, 3), dtype=np.float32)
        >>> img_bgr = convert_channel_order(img_rgb, 'rgb', 'bgr')
    """
    import numpy as np
    arr = np.asarray(image)
    from_order = from_order.lower()
    to_order = to_order.lower()

    if from_order == to_order:
        return arr

    supported = ("rgb", "bgr")
    if from_order not in supported or to_order not in supported:
        raise ValueError(f"Supported orders: {supported}. Got: {from_order!r} → {to_order!r}")

    # Reversing last axis converts RGB↔BGR (both are 3-channel, reverse maps correctly)
    return arr[..., ::-1]


def prepare_batch_input(
    images: "list[np.ndarray]",  # type: ignore[name-defined]
    output_path: str,
    channel_order: str = "rgb",
    target_channel_order: str = "rgb",
) -> int:
    """Concatenate multiple NHWC images into a single raw binary batch file for SNPE.

    SNPE requires batch > 1 to be provided as a single concatenated binary file.
    Each image must already be in NHWC layout before calling this function.

    Args:
        images: List of NumPy arrays each with shape (1, H, W, C) or (H, W, C)
        output_path: Path to write the concatenated .raw file
        channel_order: Current channel order of input images ('rgb' or 'bgr')
        target_channel_order: Required channel order for the model ('rgb' or 'bgr')

    Returns:
        Total number of bytes written

    Example:
        >>> import numpy as np
        >>> imgs = [np.zeros((1, 227, 227, 3), dtype=np.float32) for _ in range(4)]
        >>> n_bytes = prepare_batch_input(imgs, 'batch_input.raw')
    """
    import numpy as np

    processed = []
    for img in images:
        arr = np.asarray(img, dtype=np.float32)
        # Ensure 4D
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        # Reorder channels if needed
        if channel_order.lower() != target_channel_order.lower():
            arr = convert_channel_order(arr, channel_order, target_channel_order)
        processed.append(arr)

    batch = np.concatenate(processed, axis=0)  # (N, H, W, C)
    data = batch.astype(np.float32)
    data.tofile(output_path)
    return data.nbytes


def generate_image_format_notes() -> str:
    """Return a human-readable summary of SNPE image format requirements."""
    lines = [
        "SNPE Input Image Formatting Requirements",
        "=" * 45,
        "",
        "1. LAYOUT: NHWC (channel-last)",
        "   SNPE: (batch × height × width × channel)  — channel fastest-changing",
        "   PyTorch: (batch × channel × height × width) — must transpose",
        "   Conversion: np.transpose(img, (0, 2, 3, 1))  # NCHW → NHWC",
        "",
        "2. CHANNEL ORDER: Must match training",
        "   BGR: bvlc_alexnet, bvlc_googlenet, Caffe-based models",
        "   RGB: TensorFlow models, most modern PyTorch models",
        "   Conversion: image[:,:,::-1]  # reverses last axis (RGB ↔ BGR)",
        "",
        "3. BATCH > 1: Manual concatenation required",
        "   Shell: cat img0.raw img1.raw > batch.raw",
        "   Python: np.concatenate([img0, img1], axis=0).tofile('batch.raw')",
        "",
        "4. SINGLE-CHANNEL (e.g. MNIST): 4D tensor still required",
        "   PyTorch: (1, 1, 28, 28)  → SNPE: (1, 28, 28, 1)",
        "",
        "5. AlexNet example (227×227, BGR):",
        "   a) Resize to 227×227",
        "   b) Subtract mean [104, 117, 123]",
        "   c) Convert to float32",
        "   d) BGR: image[:,:,::-1]",
        "   e) Save: image.tofile('alexnet_input.raw')",
    ]
    return "\n".join(lines)


def generate_deeplabv3_preprocess_code(input_var: str = "image") -> str:
    """Generate Python preprocessing code for DeepLabv3 input images.

    Steps (must be applied IN THIS ORDER):
    1. resize_ratio = 513.0 / max(width, height)
    2. Resize with LANCZOS antialiasing
    3. Pad shorter dim to 513x513x3 with mean value=128
    4. Convert to float32
    5. Multiply by 0.00784313771874
    6. Subtract 1.0  ->  final range [-1, 1]
    """
    lines = [
        "import numpy as np",
        "from PIL import Image",
        "",
        f"def preprocess_deeplabv3({input_var}):",
        "    height, width = " + input_var + ".shape[:2]",
        "    resize_ratio = 513.0 / max(width, height)",
        "    target_w = int(resize_ratio * width)",
        "    target_h = int(resize_ratio * height)",
        "    pil_img = Image.fromarray(" + input_var + ").resize((target_w, target_h), Image.LANCZOS)",
        "    resized = numpy.array(pil_img)",
        "    pad_h = 513 - target_h",
        "    pad_w = 513 - target_w",
        "    padded = numpy.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=128)",
        "    result = padded.astype(numpy.float32)",
        "    result *= 0.00784313771874",
        "    result -= 1.0",
        "    return result  # shape (513,513,3), dtype float32, range [-1,1]",
    ]
    return "\n".join(lines)

