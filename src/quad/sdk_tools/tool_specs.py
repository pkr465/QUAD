"""SNPE/QAIRT SDK Tool CLI Builders — complete flag reference for all tools.

Based on SNPE Tools documentation (80-63442-10 Rev AH, Apr 13 2026).

Covers shared flag groups, per-tool builders, and input encoding/layout enums.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Shared Enums
# ══════════════════════════════════════════════════════════════════════════════

class InputEncoding(str, Enum):
    """Input color/data encoding types supported by SNPE converters."""
    BGR = "bgr"             # Default for legacy models (Caffe, OpenCV)
    RGB = "rgb"
    RGBA = "rgba"
    ARGB32 = "argb32"
    NV21 = "nv21"           # YUV semi-planar (Android camera default)
    NV12 = "nv12"           # YUV semi-planar
    TIME_SERIES = "time_series"  # RNN/LSTM inputs
    OTHER = "other"         # Unknown or none of the above


class InputLayout(str, Enum):
    """Input tensor layout specifiers.

    N=Batch, C=Channels, D=Depth, H=Height, W=Width, F=Feature, T=Time,
    I=Input (weights), O=Output (weights)
    """
    NCDHW = "NCDHW"       # 5D: Batch×Channels×Depth×Height×Width
    NDHWC = "NDHWC"       # 5D: channel-last
    NCHW = "NCHW"         # 4D image: PyTorch default
    NHWC = "NHWC"         # 4D image: SNPE/TF default (channel fastest-changing)
    HWIO = "HWIO"         # Conv weights (TF style)
    OIHW = "OIHW"         # Conv weights (PyTorch style)
    NFC = "NFC"           # Conv1D inputs (batch×feature×channel)
    NCF = "NCF"           # Conv1D inputs (channel-first)
    NTF = "NTF"           # RNN/LSTM: batch×time×feature
    TNF = "TNF"           # RNN/LSTM: time×batch×feature
    NF = "NF"             # 2D: Dense/FC inputs
    NC = "NC"             # 2D: batch×channels
    F = "F"               # 1D: bias tensors
    NONTRIVIAL = "NONTRIVIAL"  # Everything else


class InputType(str, Enum):
    """DSP input quantization type — controls how DSP handles input data.

    IMAGE:   Input float [0,255]; DSP casts to uint8 (mean=0, max=255).
    DEFAULT: Pass float directly; DSP quantizes based on model params.
    OPAQUE:  Consumer layer requires float; bypass quantization.
    """
    IMAGE = "image"
    DEFAULT = "default"
    OPAQUE = "opaque"


class FloatBitwidth(int, Enum):
    """Float bitwidth for parameters/activations."""
    FP32 = 32
    FP16 = 16


class QuantBitwidth(int, Enum):
    """Quantization bitwidth."""
    INT4 = 4
    INT8 = 8
    INT16 = 16
    INT32 = 32


class MaskedSoftmaxMode(str, Enum):
    """MaskedSoftmax optimization modes."""
    COMPRESSED = "compressed"
    UNCOMPRESSED = "uncompressed"


class ExportFormat(str, Enum):
    """qairt-converter export format."""
    DLC_DEFAULT = "DLC_DEFAULT"       # Float graph from float source, or quant from encodings
    DLC_STRIP_QUANT = "DLC_STRIP_QUANT"  # Float graph, discard quant data


class TargetBackend(str, Enum):
    """qairt-converter / qairt-quantizer target backend."""
    CPU = "CPU"
    GPU = "GPU"
    DSP = "DSP"
    HTP = "HTP"    # Default
    HTA = "HTA"
    LPAI = "LPAI"


class PerfProfile(str, Enum):
    """snpe-net-run / snpe-parallel-run performance profile."""
    LOW_BALANCED = "low_balanced"
    BALANCED = "balanced"
    DEFAULT = "default"               # Same as balanced (deprecated)
    HIGH_PERFORMANCE = "high_performance"
    SUSTAINED_HIGH = "sustained_high_performance"
    BURST = "burst"
    LOW_POWER_SAVER = "low_power_saver"
    POWER_SAVER = "power_saver"
    HIGH_POWER_SAVER = "high_power_saver"
    EXTREME_POWER_SAVER = "extreme_power_saver"
    SYSTEM_SETTINGS = "system_settings"


class ProfilingLevelNet(str, Enum):
    """snpe-net-run profiling levels (differs from snpe-diagview levels)."""
    OFF = "off"
    BASIC = "basic"
    MODERATE = "moderate"
    DETAILED = "detailed"
    LINTING = "linting"


class CacheCompatibilityMode(str, Enum):
    """HTP cache compatibility check mode."""
    PERMISSIVE = "permissive"          # Compatible if it can run on device
    STRICT = "strict"                  # Compatible only if it fully utilizes HW capability
    ALWAYS_GENERATE_NEW = "always_generate_new"  # Always incompatible, regenerate


class GpuMode(str, Enum):
    """GPU compute mode."""
    DEFAULT = "default"    # float32 math, float16 storage
    FLOAT16 = "float16"    # float16 math, float16 storage


class PriorityHint(str, Enum):
    """Inference priority hint."""
    LOW = "low"
    NORMAL = "normal"
    NORMAL_HIGH = "normal_high"   # DSP only
    HIGH = "high"


class RuntimeOrder(str, Enum):
    """Runtime names for --runtime_order flag."""
    CPU = "cpu"
    GPU = "gpu"
    GPU_FLOAT16 = "gpu_float16"
    AIP = "aip"
    DSP = "dsp"


class QuantCalibration(str, Enum):
    """qairt-quantizer calibration methods."""
    MIN_MAX = "min-max"
    SQNR = "sqnr"
    ENTROPY = "entropy"
    MSE = "mse"
    PERCENTILE = "percentile"


class QuantSchema(str, Enum):
    """qairt-quantizer quantization schema."""
    ASYMMETRIC = "asymmetric"
    SYMMETRIC = "symmetric"
    UNSIGNED_SYMMETRIC = "unsignedsymmetric"


# ══════════════════════════════════════════════════════════════════════════════
# Shared Input Spec
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConverterInputSpec:
    """Full input tensor specification for converter tools.

    Covers all per-input flags: name/dim/dtype/encoding/layout/type.
    """
    name: str
    dimensions: Optional[tuple[int, ...]] = None        # --input_dim / --source_model_input_shape
    dtype: str = "float32"                               # --input_dtype
    encoding_in: InputEncoding = InputEncoding.BGR       # --input_encoding (in)
    encoding_out: Optional[InputEncoding] = None         # --input_encoding (optional out)
    layout: Optional[InputLayout] = None                 # --input_layout / --source_model_input_layout
    input_type: Optional[InputType] = None               # --input_type (DSP handling)

    @property
    def dim_string(self) -> str:
        """Comma-separated dimension string, e.g. '1,224,224,3'."""
        if self.dimensions is None:
            return ""
        return ",".join(str(d) for d in self.dimensions)

    def legacy_dim_args(self) -> list[str]:
        """--input_dim 'name' 'N,H,W,C' (legacy converters)."""
        if self.dimensions is None:
            return []
        return ["--input_dim", f"'{self.name}'", self.dim_string]

    def qairt_shape_args(self) -> list[str]:
        """--source_model_input_shape 'name' 'N,C,H,W' (qairt-converter)."""
        if self.dimensions is None:
            return []
        return ["--source_model_input_shape", f"'{self.name}'", self.dim_string]

    def encoding_args(self) -> list[str]:
        """--input_encoding 'name' encoding_in [encoding_out]."""
        args = ["--input_encoding", f'"{self.name}"', self.encoding_in.value]
        if self.encoding_out is not None:
            args.append(self.encoding_out.value)
        return args

    def layout_args(self) -> list[str]:
        """--input_layout 'name' LAYOUT."""
        if self.layout is None:
            return []
        return ["--input_layout", f'"{self.name}"', self.layout.value]

    def input_type_args(self) -> list[str]:
        """--input_type 'name' type."""
        if self.input_type is None:
            return []
        return ["--input_type", f'"{self.name}"', self.input_type.value]


# ══════════════════════════════════════════════════════════════════════════════
# snpe-onnx-to-dlc Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OnnxConverterArgs:
    """CLI arg builder for snpe-onnx-to-dlc.

    Required: input_network
    Optional: input_specs, output_path, out_nodes, float_bitwidth,
              dry_run, batch_override, symbol_overrides, maskedsoftmax,
              no_simplification, keep_disconnected_nodes, etc.
    """
    input_network: str
    output_path: Optional[str] = None
    input_specs: list[ConverterInputSpec] = field(default_factory=list)
    out_nodes: list[str] = field(default_factory=list)
    float_bitwidth: FloatBitwidth = FloatBitwidth.FP32
    float_bias_bitwidth: Optional[FloatBitwidth] = None
    model_version: Optional[str] = None
    batch_override: Optional[int] = None
    symbol_overrides: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False
    no_simplification: bool = False
    disable_batchnorm_folding: bool = False
    expand_lstm_op_structure: bool = False
    keep_disconnected_nodes: bool = False
    preserve_onnx_output_order: bool = False
    keep_quant_nodes: bool = False
    quantization_overrides: Optional[str] = None
    udo_config_paths: list[str] = field(default_factory=list)
    apply_masked_softmax: Optional[MaskedSoftmaxMode] = None
    copyright_file: Optional[str] = None
    enable_framework_trace: bool = False
    onnx_validate_models: bool = False
    onnx_summary: bool = False

    def build(self) -> list[str]:
        """Build the complete CLI argument list."""
        args = ["snpe-onnx-to-dlc", "--input_network", self.input_network]

        for spec in self.input_specs:
            args += spec.legacy_dim_args()
            args += spec.encoding_args()
            args += spec.layout_args()
            args += spec.input_type_args()

        for node in self.out_nodes:
            args += ["--out_node", node]

        if self.output_path:
            args += ["--output_path", self.output_path]
        if self.float_bitwidth != FloatBitwidth.FP32:
            args += ["--float_bitwidth", str(self.float_bitwidth.value)]
        if self.float_bias_bitwidth is not None:
            args += ["--float_bias_bitwidth", str(self.float_bias_bitwidth.value)]
        if self.model_version:
            args += ["--model_version", self.model_version]
        if self.batch_override is not None:
            args += ["--batch", str(self.batch_override)]
        for sym, val in self.symbol_overrides.items():
            args += ["--define_symbol", sym, str(val)]
        if self.dry_run:
            args += ["--dry_run", "info"]
        if self.no_simplification:
            args.append("--no_simplification")
        if self.disable_batchnorm_folding:
            args.append("--disable_batchnorm_folding")
        if self.expand_lstm_op_structure:
            args.append("--expand_lstm_op_structure")
        if self.keep_disconnected_nodes:
            args.append("--keep_disconnected_nodes")
        if self.preserve_onnx_output_order:
            args.append("--preserve_onnx_output_order")
        if self.keep_quant_nodes:
            args.append("--keep_quant_nodes")
        if self.quantization_overrides:
            args += ["--quantization_overrides", self.quantization_overrides]
        for p in self.udo_config_paths:
            args += ["--udo_config_paths", p]
        if self.apply_masked_softmax:
            args += ["--apply_masked_softmax", self.apply_masked_softmax.value]
        if self.copyright_file:
            args += ["--copyright_file", self.copyright_file]
        if self.enable_framework_trace:
            args.append("--enable_framework_trace")
        if self.onnx_validate_models:
            args.append("--onnx_validate_models")
        if self.onnx_summary:
            args.append("--onnx_summary")

        return args


# ══════════════════════════════════════════════════════════════════════════════
# qairt-converter Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QairtConverterArgs:
    """CLI arg builder for qairt-converter (unified, recommended).

    Auto-detects framework from input_network extension.
    Key differences vs legacy converters:
      - --source_model_input_shape (not --input_dim)
      - --out_tensor_node (not --out_node)
      - --source_model_input_layout vs --desired_input_layout (source vs target)
      - --desired_input_color_encoding (not --input_encoding)
      - --target_backend / --target_soc_model
      - --export_format (DLC_DEFAULT or DLC_STRIP_QUANT)
      - Framework-specific flags: --onnx_*, --tf_*, --tflite_*
    """
    input_network: str
    output_path: Optional[str] = None
    input_specs: list[ConverterInputSpec] = field(default_factory=list)
    out_tensor_nodes: list[str] = field(default_factory=list)
    float_bitwidth: FloatBitwidth = FloatBitwidth.FP32
    float_bias_bitwidth: Optional[FloatBitwidth] = None
    model_version: Optional[str] = None
    target_backend: Optional[TargetBackend] = None
    target_soc_model: Optional[str] = None        # e.g. "sm8750", "sm8650"
    export_format: ExportFormat = ExportFormat.DLC_DEFAULT
    dry_run: bool = False
    remove_unused_inputs: bool = False
    enable_framework_trace: bool = False
    preserve_io_datatype: list[str] = field(default_factory=list)  # tensor names
    quantization_overrides: Optional[str] = None
    # ONNX-specific
    onnx_override_batch: Optional[int] = None
    onnx_define_symbols: dict[str, int] = field(default_factory=dict)
    onnx_skip_simplification: bool = False
    onnx_validate_models: bool = False
    onnx_summary: bool = False
    # TensorFlow-specific
    tf_override_batch: Optional[int] = None
    tf_saved_model_tag: Optional[str] = None
    tf_saved_model_signature_key: Optional[str] = None
    tf_disable_optimization: bool = False
    tf_show_unconsumed_nodes: bool = False
    # TFLite-specific
    tflite_signature_name: Optional[str] = None
    # PyTorch-specific
    dump_exported_onnx: bool = False
    # LoRA
    lora_weight_list: Optional[str] = None
    quant_updatable_mode: Optional[str] = None  # none, adapter_only, all
    # UDO
    converter_op_package_lib: Optional[str] = None

    def build(self) -> list[str]:
        """Build the complete CLI argument list."""
        args = ["qairt-converter", "--input_network", self.input_network]

        for spec in self.input_specs:
            args += spec.qairt_shape_args()
            args += spec.encoding_args()
            # In qairt-converter, --source_model_input_layout and --desired_input_layout are separate
            if spec.layout is not None:
                args += ["--source_model_input_layout", f'"{spec.name}"', spec.layout.value]
            args += spec.input_type_args()

        for node in self.out_tensor_nodes:
            args += ["--out_tensor_node", node]

        if self.output_path:
            args += ["--output_path", self.output_path]
        if self.float_bitwidth != FloatBitwidth.FP32:
            args += ["--float_bitwidth", str(self.float_bitwidth.value)]
        if self.float_bias_bitwidth is not None:
            args += ["--float_bias_bitwidth", str(self.float_bias_bitwidth.value)]
        if self.model_version:
            args += ["--set_model_version", self.model_version]
        if self.target_backend:
            args += ["--target_backend", self.target_backend.value]
            if self.target_soc_model:
                args += ["--target_soc_model", self.target_soc_model]
        if self.export_format != ExportFormat.DLC_DEFAULT:
            args += ["--export_format", self.export_format.value]
        if self.dry_run:
            args.append("--dry_run")
        if self.remove_unused_inputs:
            args.append("--remove_unused_inputs")
        if self.enable_framework_trace:
            args.append("--enable_framework_trace")
        for tensor in self.preserve_io_datatype:
            args += ["--preserve_io_datatype", tensor]
        if self.quantization_overrides:
            args += ["--quantization_overrides", self.quantization_overrides]
        # ONNX
        if self.onnx_override_batch is not None:
            args += ["--onnx_override_batch", str(self.onnx_override_batch)]
        for sym, val in self.onnx_define_symbols.items():
            args += ["--onnx_define_symbol", sym, str(val)]
        if self.onnx_skip_simplification:
            args.append("--onnx_skip_simplification")
        if self.onnx_validate_models:
            args.append("--onnx_validate_models")
        if self.onnx_summary:
            args.append("--onnx_summary")
        # TF
        if self.tf_override_batch is not None:
            args += ["--tf_override_batch", str(self.tf_override_batch)]
        if self.tf_saved_model_tag:
            args += ["--tf_saved_model_tag", self.tf_saved_model_tag]
        if self.tf_saved_model_signature_key:
            args += ["--tf_saved_model_signature_key", self.tf_saved_model_signature_key]
        if self.tf_disable_optimization:
            args.append("--tf_disable_optimization")
        if self.tf_show_unconsumed_nodes:
            args.append("--tf_show_unconsumed_nodes")
        # TFLite
        if self.tflite_signature_name:
            args += ["--tflite_signature_name", self.tflite_signature_name]
        # PyTorch
        if self.dump_exported_onnx:
            args.append("--dump_exported_onnx")
        # LoRA
        if self.lora_weight_list:
            args += ["--lora_weight_list", self.lora_weight_list]
        if self.quant_updatable_mode:
            args += ["--quant_updatable_mode", self.quant_updatable_mode]
        if self.converter_op_package_lib:
            args += ["--converter_op_package_lib", self.converter_op_package_lib]

        return args


# ══════════════════════════════════════════════════════════════════════════════
# qairt-quantizer Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QairtQuantizerArgs:
    """CLI arg builder for qairt-quantizer.

    Converts non-quantized DLC to quantized DLC.
    Supports AIMET, per-channel, per-row, calibration methods.
    """
    input_dlc: str
    output_dlc: Optional[str] = None
    input_list: Optional[str] = None          # Required for quantization (calibration data)
    weights_bitwidth: QuantBitwidth = QuantBitwidth.INT8
    act_bitwidth: QuantBitwidth = QuantBitwidth.INT8
    bias_bitwidth: QuantBitwidth = QuantBitwidth.INT8
    float_bitwidth: FloatBitwidth = FloatBitwidth.FP32
    float_bias_bitwidth: Optional[int] = None   # 32 or 16
    use_per_channel_quantization: bool = False
    use_per_row_quantization: bool = False
    enable_per_row_quantized_bias: bool = False
    act_quantizer_calibration: QuantCalibration = QuantCalibration.MIN_MAX
    param_quantizer_calibration: QuantCalibration = QuantCalibration.MIN_MAX
    act_quantizer_schema: QuantSchema = QuantSchema.ASYMMETRIC
    param_quantizer_schema: QuantSchema = QuantSchema.ASYMMETRIC
    percentile_calibration_value: float = 99.99
    apply_algorithms: list[str] = field(default_factory=list)  # e.g. ["cle"]
    enable_float_fallback: bool = False
    ignore_quantization_overrides: bool = False
    use_aimet_quantizer: bool = False
    keep_weights_quantized: bool = False
    adjust_bias_encoding: bool = False
    use_native_input_files: bool = False
    use_native_output_files: bool = False
    preserve_io_datatype: list[str] = field(default_factory=list)
    target_backend: Optional[TargetBackend] = None
    target_soc_model: Optional[str] = None
    dump_encoding_json: bool = False
    export_stripped_dlc: bool = False
    op_package_lib: Optional[str] = None
    config_file: Optional[str] = None         # YAML config file

    def build(self) -> list[str]:
        """Build the complete CLI argument list."""
        args = ["qairt-quantizer", "--input_dlc", self.input_dlc]

        if self.output_dlc:
            args += ["--output_dlc", self.output_dlc]
        if self.input_list:
            args += ["--input_list", self.input_list]

        if self.weights_bitwidth != QuantBitwidth.INT8:
            args += ["--weights_bitwidth", str(self.weights_bitwidth.value)]
        if self.act_bitwidth != QuantBitwidth.INT8:
            args += ["--act_bitwidth", str(self.act_bitwidth.value)]
        if self.bias_bitwidth != QuantBitwidth.INT8:
            args += ["--bias_bitwidth", str(self.bias_bitwidth.value)]
        if self.float_bitwidth != FloatBitwidth.FP32:
            args += ["--float_bitwidth", str(self.float_bitwidth.value)]
        if self.float_bias_bitwidth is not None:
            args += ["--float_bias_bitwidth", str(self.float_bias_bitwidth)]

        if self.use_per_channel_quantization:
            args.append("--use_per_channel_quantization")
        if self.use_per_row_quantization:
            args.append("--use_per_row_quantization")
        if self.enable_per_row_quantized_bias:
            args.append("--enable_per_row_quantized_bias")

        if self.act_quantizer_calibration != QuantCalibration.MIN_MAX:
            args += ["--act_quantizer_calibration", self.act_quantizer_calibration.value]
        if self.param_quantizer_calibration != QuantCalibration.MIN_MAX:
            args += ["--param_quantizer_calibration", self.param_quantizer_calibration.value]
        if self.act_quantizer_schema != QuantSchema.ASYMMETRIC:
            args += ["--act_quantizer_schema", self.act_quantizer_schema.value]
        if self.param_quantizer_schema != QuantSchema.ASYMMETRIC:
            args += ["--param_quantizer_schema", self.param_quantizer_schema.value]
        if self.percentile_calibration_value != 99.99:
            args += ["--percentile_calibration_value", str(self.percentile_calibration_value)]

        for algo in self.apply_algorithms:
            args += ["--apply_algorithms", algo]

        if self.enable_float_fallback:
            args.append("--enable_float_fallback")
        if self.ignore_quantization_overrides:
            args.append("--ignore_quantization_overrides")
        if self.use_aimet_quantizer:
            args.append("--use_aimet_quantizer")
        if self.keep_weights_quantized:
            args.append("--keep_weights_quantized")
        if self.adjust_bias_encoding:
            args.append("--adjust_bias_encoding")
        if self.use_native_input_files:
            args.append("--use_native_input_files")
        if self.use_native_output_files:
            args.append("--use_native_output_files")

        for tensor in self.preserve_io_datatype:
            args += ["--preserve_io_datatype", tensor]

        if self.target_backend:
            args += ["--target_backend", self.target_backend.value]
            if self.target_soc_model:
                args += ["--target_soc_model", self.target_soc_model]

        if self.dump_encoding_json:
            args.append("--dump_encoding_json")
        if self.export_stripped_dlc:
            args.append("--export_stripped_dlc")
        if self.op_package_lib:
            args += ["--op_package_lib", self.op_package_lib]
        if self.config_file:
            args += ["--config", self.config_file]

        return args


# ══════════════════════════════════════════════════════════════════════════════
# snpe-net-run Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SnpeNetRunArgs:
    """CLI arg builder for snpe-net-run.

    Loads a DLC, runs inference on input tensors, saves raw output.
    Supports all buffer types, profiling, caching, network resizing, UDO.
    """
    container: str
    input_list: str
    # Runtime
    use_dsp: bool = False
    use_gpu: bool = False
    use_cpu: bool = False
    use_aip: bool = False
    runtime_order: list[RuntimeOrder] = field(default_factory=list)
    gpu_mode: Optional[GpuMode] = None
    # Performance
    perf_profile: PerfProfile = PerfProfile.DEFAULT
    perf_config_yaml: Optional[str] = None
    priority_hint: PriorityHint = PriorityHint.NORMAL
    # Profiling
    profiling_level: ProfilingLevelNet = ProfilingLevelNet.DETAILED
    # Output
    output_dir: str = "output"
    set_output_tensors: list[str] = field(default_factory=list)
    set_unconsumed_as_output: bool = False
    debug: bool = False
    # Buffer types
    userbuffer_float: bool = False
    userbuffer_tf8: bool = False
    userbuffer_auto: bool = False
    userbuffer_memorymapped: bool = False
    userbuffer_memorymapped_shared: bool = False
    use_native_input_files: bool = False
    use_native_output_files: bool = False
    static_min_max: bool = False
    # Network resizing
    input_name: Optional[str] = None
    input_dimensions: Optional[str] = None   # e.g. "1,224,224,3"
    # Caching
    enable_init_cache: bool = False
    cache_compatibility_mode: CacheCompatibilityMode = CacheCompatibilityMode.PERMISSIVE
    validate_cache: bool = False
    enable_htp_accelerated_init: bool = False
    # Platform
    platform_options: Optional[str] = None    # e.g. "unsignedPD:OFF"
    enable_cpu_fallback: bool = False
    enable_cpu_fxp: bool = False
    enable_cpu_qmx: bool = False
    # Duration / control
    duration: Optional[int] = None             # seconds
    keep_num_outputs: Optional[int] = None
    timeout: Optional[int] = None              # microseconds (HTP only)
    # Multi-graph
    graph_init: Optional[str] = None
    graph_execute: Optional[str] = None
    # Logging
    userlogs: Optional[str] = None
    model_name: Optional[str] = None
    # UDO
    udo_package_path: Optional[str] = None
    # Deferred alloc
    deferred_init: Optional[str] = None  # weights, spill-fill, all, none

    def build(self) -> list[str]:
        """Build the complete CLI argument list."""
        args = [
            "snpe-net-run",
            "--container", self.container,
            "--input_list", self.input_list,
        ]

        # Runtime selection
        if self.runtime_order:
            args += ["--runtime_order", ",".join(r.value for r in self.runtime_order)]
        else:
            if self.use_dsp:
                args.append("--use_dsp")
            elif self.use_gpu:
                args.append("--use_gpu")
            elif self.use_aip:
                args.append("--use_aip")
            elif self.use_cpu:
                args.append("--use_cpu")

        if self.gpu_mode and self.gpu_mode != GpuMode.DEFAULT:
            args += ["--gpu_mode", self.gpu_mode.value]
        if self.perf_config_yaml:
            args += ["--perf_config_yaml", self.perf_config_yaml]
        else:
            args += ["--perf_profile", self.perf_profile.value]
        if self.priority_hint != PriorityHint.NORMAL:
            args += ["--priority_hint", self.priority_hint.value]

        args += ["--profiling_level", self.profiling_level.value]
        args += ["--output_dir", self.output_dir]

        for tensor in self.set_output_tensors:
            args += ["--set_output_tensors", tensor]
        if self.set_unconsumed_as_output:
            args.append("--set_unconsumed_as_output")
        if self.debug:
            args.append("--debug")

        # Buffer modes
        if self.userbuffer_auto:
            args.append("--userbuffer_auto")
        elif self.userbuffer_tf8:
            args.append("--userbuffer_tf8")
        elif self.userbuffer_float:
            args.append("--userbuffer_float")
        if self.userbuffer_memorymapped_shared:
            args.append("--userbuffer_memorymapped_shared")
        elif self.userbuffer_memorymapped:
            args.append("--userbuffer_memorymapped")
        if self.use_native_input_files:
            args.append("--use_native_input_files")
        if self.use_native_output_files:
            args.append("--use_native_output_files")
        if self.static_min_max:
            args.append("--static_min_max")

        # Network resizing
        if self.input_name:
            args += [f"--input_name={self.input_name}"]
        if self.input_dimensions:
            args += [f"--input_dimensions={self.input_dimensions}"]

        # Caching
        if self.enable_init_cache:
            args.append("--enable_init_cache")
        if self.enable_htp_accelerated_init:
            args.append("--enable_htp_accelerated_init")
        if self.cache_compatibility_mode != CacheCompatibilityMode.PERMISSIVE:
            args += [f"--cache_compatibility_mode={self.cache_compatibility_mode.value}"]
        if self.validate_cache:
            args.append("--validate_cache")

        # Platform
        if self.platform_options:
            args += ["--platform_options", self.platform_options]
        if self.enable_cpu_fallback:
            args.append("--enable_cpu_fallback")
        if self.enable_cpu_fxp:
            args.append("--enable_cpu_fxp")
        if self.enable_cpu_qmx:
            args.append("--enable_cpu_qmx")

        # Duration
        if self.duration is not None:
            args += ["--duration", str(self.duration)]
        if self.keep_num_outputs is not None:
            args += ["--keep_num_outputs", str(self.keep_num_outputs)]
        if self.timeout is not None:
            args += [f"--timeout={self.timeout}"]

        # Multi-graph
        if self.graph_init:
            args += ["--graph_init", self.graph_init]
        if self.graph_execute:
            args += ["--graph_execute", self.graph_execute]

        # Logging
        if self.userlogs:
            args += ["--userlogs", self.userlogs]
        if self.model_name:
            args += ["--model_name", self.model_name]
        if self.udo_package_path:
            args += [f"--udo_package_path={self.udo_package_path}"]
        if self.deferred_init and self.deferred_init != "none":
            args += ["--deferred_init", self.deferred_init]

        return args


# ══════════════════════════════════════════════════════════════════════════════
# snpe-diagview Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiagviewArgs:
    """CLI arg builder for snpe-diagview."""
    input_log: str
    output_csv: Optional[str] = None
    chrometrace: Optional[str] = None    # Output prefix for chrometrace JSON
    csv_format_version: int = 1          # 1 (default) or 2

    def build(self) -> list[str]:
        args = ["snpe-diagview", "--input_log", self.input_log]
        if self.output_csv:
            args += ["--output", self.output_csv]
            if self.csv_format_version != 1:
                args += ["--csv_format_version", str(self.csv_format_version)]
        if self.chrometrace:
            args += ["--chrometrace", self.chrometrace]
        return args


# ══════════════════════════════════════════════════════════════════════════════
# Diagview Timing Layer Descriptions (from timing diagram)
# ══════════════════════════════════════════════════════════════════════════════

DIAGVIEW_TIMING_LAYERS: dict[str, str] = {
    "Total Inference Time": (
        "Full end-to-end time from SNPE API call to result. "
        "Spans from App through SNPE → HTP Stub → RPC → HTP Core and back."
    ),
    "Forward Propagate Time": (
        "Time from Backend API Boundary to HTP Core completion. "
        "Excludes SNPE overhead before backend API boundary."
    ),
    "RPC Execute Time": (
        "Time from RPC Boundary into HTP and back. "
        "Includes RPC call overhead + HTP execution time."
    ),
    "SNPE Acc Time": (
        "Time within the HTP boundary (from HTP Boundary to HTP Core). "
        "SNPE accelerator scheduling and execution."
    ),
    "Acc Time": (
        "Pure HTP Core execution time. "
        "The innermost measurement — actual NPU compute time."
    ),
}

DIAGVIEW_NOTES = {
    "averaging": (
        "If input_list has multiple inputs, all timing is AVERAGED over the entire input set."
    ),
    "log_files": (
        "snpe-net-run generates SNPEDiag_0.log, SNPEDiag_1.log, ..., SNPEDiag_n.log "
        "where n corresponds to the nth iteration."
    ),
    "backend_splitting": (
        "Note: Some backends (DSP/GPU) may split one op into multiple ops or fuse multiple ops. "
        "A detailed profiling log may therefore show mismatching op counts vs layer mapping."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# snpe-dlc-info / qairt-dlc-info Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DlcInfoArgs:
    """CLI arg builder for snpe-dlc-info and qairt-dlc-info."""
    input_dlc: str
    save: Optional[str] = None           # Save to CSV
    display_memory: bool = False
    display_all_encodings: bool = False
    dump_framework_trace: bool = False
    tool: str = "snpe-dlc-info"         # or "qairt-dlc-info"

    def build(self) -> list[str]:
        args = [self.tool, "--input_dlc", self.input_dlc]
        if self.save:
            args += ["--save", self.save]
        if self.display_memory:
            args.append("--memory" if self.tool == "snpe-dlc-info" else "--display_memory")
        if self.display_all_encodings:
            args.append("--display_all_encodings")
        if self.dump_framework_trace:
            args.append("--dump_framework_trace")
        return args


# ══════════════════════════════════════════════════════════════════════════════
# snpe-dlc-diff / qairt-dlc-diff Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DlcDiffArgs:
    """CLI arg builder for snpe-dlc-diff and qairt-dlc-diff."""
    input_dlc_one: str
    input_dlc_two: str
    compare_copyrights: bool = False
    compare_layers: bool = True
    compare_parameters: bool = True
    compare_dimensions: bool = True
    compare_weights: bool = False
    compare_outputs: bool = False
    diff_by_id: bool = False
    compare_hta: bool = False            # snpe-dlc-diff only
    save: Optional[str] = None
    tool: str = "snpe-dlc-diff"         # or "qairt-dlc-diff"

    def build(self) -> list[str]:
        # Flag names differ slightly between snpe and qairt versions
        is_qairt = self.tool == "qairt-dlc-diff"

        args = [self.tool, "-i1", self.input_dlc_one, "-i2", self.input_dlc_two]
        if self.compare_copyrights:
            args.append("-c" if not is_qairt else "--compare_copyrights")
        if self.compare_layers:
            args.append("-l" if not is_qairt else "--compare_layers")
        if self.compare_parameters:
            args.append("-p" if not is_qairt else "--compare_parameters")
        if self.compare_dimensions:
            args.append("-d" if not is_qairt else "--compare_dimensions")
        if self.compare_weights:
            args.append("-w" if not is_qairt else "--compare_weights")
        if self.compare_outputs:
            args.append("-o" if not is_qairt else "--compare_outputs")
        if self.diff_by_id:
            args.append("-i" if not is_qairt else "--enable_diff_by_id")
        if self.compare_hta and not is_qairt:
            args.append("-x")
        if self.save:
            args += ["-s", self.save] if not is_qairt else ["--save", self.save]
        return args
