"""SNPE Accuracy Debugger (Experimental) — Layer-level accuracy analysis tool.

Based on SNPE Accuracy Debugger documentation (80-63442-10 Rev AH, Apr 13 2026).

The Accuracy Debugger finds inaccuracies in neural networks at the layer level.
It compares golden outputs (from ML frameworks: TF, ONNX, TFLite) with SNPE
inference engine outputs, and provides visualization and analysis tools.

Supported models: ONNX, TFLite, TensorFlow 1.x

Six features (modes), each invoked with its corresponding --option:

1. --framework_runner   Run model through ML framework; produce golden outputs
2. --inference_engine   Run model through SNPE; produce inference outputs
3. --verification       Compare golden vs inference using verifiers
4. --compare_encodings  Compare SNPE DLC encodings with AIMET encodings
5. --tensor_inspection  Compare target vs reference tensors with plots/stats
6. --quant_checker      Analyze quantization quality (weights/biases/activations)

If no mode is specified, runs framework_runner → inference_engine → verification
sequentially (E2E mode).

Output directories:
  working_dir/framework_runner/YYYY-MM-DD_HH:mm:ss/ (+ latest → symlink)
  working_dir/inference_engine/ (+ latest → symlink)
  working_dir/verification/ (+ latest → symlink)
  working_dir/compare_encodings/ (+ latest → symlink)

Note: On Windows x86, only CPU runtime is currently tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

TOOL_NAME = "snpe-accuracy-debugger"
DEFAULT_WORKING_DIR = "working_directory"

# Output subdirectory names
OUTPUT_DIRS = {
    "framework_runner": "framework_runner",
    "inference_engine": "inference_engine",
    "verification": "verification",
    "compare_encodings": "compare_encodings",
    "tensor_inspection": "tensor_inspection",
    "quant_checker": "quant_checker",
    "wrapper": "wrapper",
}

# Generated files of interest
KEY_OUTPUT_FILES = {
    "tensor_mapping": "tensor_mapping.json",
    "graph_struct": "{model_name}_graph_struct.json",
    "snpe_graph_struct": "snpe_model_graph_struct.json",
    "inference_engine_options": "inference_engine_options.json",
    "framework_runner_options": "framework_runner_options.json",
}


# ══════════════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════════════

class AccDebuggerMode(str, Enum):
    """Accuracy Debugger operating modes."""
    FRAMEWORK_RUNNER = "framework_runner"
    INFERENCE_ENGINE = "inference_engine"
    VERIFICATION = "verification"
    COMPARE_ENCODINGS = "compare_encodings"
    TENSOR_INSPECTION = "tensor_inspection"
    QUANT_CHECKER = "quant_checker"


class AccDebuggerFramework(str, Enum):
    """Supported ML frameworks."""
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TFLITE = "tflite"


class AccDebuggerRuntime(str, Enum):
    """SNPE inference runtimes."""
    CPU = "cpu"
    GPU = "gpu"
    DSP = "dsp"
    DSPV68 = "dspv68"
    DSPV69 = "dspv69"
    DSPV73 = "dspv73"
    DSPV75 = "dspv75"


class AccDebuggerArchitecture(str, Enum):
    """Target device architectures."""
    AARCH64_ANDROID = "aarch64-android"
    AARCH64_ANDROID_CLANG60 = "aarch64-android-clang6.0"
    AARCH64_ANDROID_CLANG80 = "aarch64-android-clang8.0"
    X86_64_LINUX_CLANG = "x86_64-linux-clang"
    X86_64_WINDOWS_MSVC = "x86_64-windows-msvc"
    WOS = "wos"           # Windows on Snapdragon


class AccDebuggerHostDevice(str, Enum):
    """Host device running the conversion."""
    X86 = "x86"
    X86_64_WINDOWS_MSVC = "x86_64-windows-msvc"
    WOS = "wos"


class VerifierType(str, Enum):
    """Supported accuracy verifiers.

    Each verifier compares framework and inference outputs using a specific metric.
    CosineSimilarity: 0–1 range, higher = more similar (typically use > 0.99).
    SQNR: Signal-to-quantization noise ratio (higher = better).
    RtolAtol: Relative + absolute tolerance checks.
    """
    RTOLATOL = "RtolAtol"
    ADJUSTED_RTOLATOL = "AdjustedRtolAtol"
    TOPK = "TopK"
    L1ERROR = "L1Error"
    COSINE_SIMILARITY = "CosineSimilarity"
    MSE = "MSE"
    MAE = "MAE"
    SQNR = "SQNR"
    MEAN_IOU = "MeanIOU"
    SCALED_DIFF = "ScaledDiff"


class AccPrecision(str, Enum):
    """Inference precision for accuracy debugger."""
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


class InferenceEngineStage(str, Enum):
    """Starting stage for inference engine."""
    SOURCE = "source"       # Start from source framework model (default)
    CONVERTED = "converted"
    COMPILED = "compiled"


class TensorDataType(str, Enum):
    """Tensor inspection data types."""
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"
    UINT16 = "uint16"
    FLOAT32 = "float32"


class QuantAlgorithm(str, Enum):
    """Quantization checker analysis algorithms."""
    MINMAX = "minmax"
    MAXDIFF = "maxdiff"
    SQNR = "sqnr"
    STATS = "stats"
    DATA_RANGE_ANALYZER = "data_range_analyzer"
    DATA_DISTRIBUTION_ANALYZER = "data_distribution_analyzer"


# ══════════════════════════════════════════════════════════════════════════════
# Shared Specs
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class InputTensorSpec:
    """Specification for an accuracy debugger input tensor.

    Format: "input_name" comma-separated-dimensions path-to-raw-file [dtype]
    Example: "data" 1,224,224,3 data.raw float32
    For TF models: "input:0" 1,299,299,3 chairs.raw
    For ONNX models: drop the :0 index (just "Input")
    """
    name: str
    dimensions: tuple[int, ...]
    raw_path: str
    dtype: str = "float32"

    @property
    def dim_string(self) -> str:
        return ",".join(str(d) for d in self.dimensions)

    def to_args(self) -> list[str]:
        """Emit: --input_tensor "name" dims raw_path [dtype]"""
        args = ["--input_tensor", f'"{self.name}"', self.dim_string, self.raw_path]
        if self.dtype and self.dtype != "float32":
            args.append(self.dtype)
        return args


@dataclass
class VerifierSpec:
    """A verifier with optional hyperparameters.

    Format: VerifierType [param1 val1 param2 val2 ...]
    Example: --default_verifier CosineSimilarity param1 1 param2 2
    Example: --default_verifier rtolatol,rtolmargin,0.01,atolmargin,0.01
    """
    verifier: VerifierType
    params: dict[str, Any] = field(default_factory=dict)
    # For comma-style params: rtolatol,rtolmargin,0.01,atolmargin,0.01
    comma_params: Optional[str] = None

    def to_flag_value(self) -> str:
        """Build the string passed after --default_verifier or --verifier."""
        if self.comma_params:
            return f"{self.verifier.value},{self.comma_params}"
        if self.params:
            parts = [self.verifier.value]
            for k, v in self.params.items():
                parts += [k, str(v)]
            return " ".join(parts)
        return self.verifier.value

    def to_args(self, flag: str = "--default_verifier") -> list[str]:
        """Emit: --default_verifier CosineSimilarity param1 1 ..."""
        if self.comma_params:
            return [flag, f"{self.verifier.value},{self.comma_params}"]
        if self.params:
            parts = [flag, self.verifier.value]
            for k, v in self.params.items():
                parts += [str(k), str(v)]
            return parts
        return [flag, self.verifier.value]


# ══════════════════════════════════════════════════════════════════════════════
# Feature CLI Builders
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FrameworkRunnerArgs:
    """CLI arg builder for snpe-accuracy-debugger --framework_runner."""
    framework: AccDebuggerFramework
    model_path: str
    input_tensors: list[InputTensorSpec]
    output_tensor: str
    working_dir: Optional[str] = None
    output_dirname: Optional[str] = None
    verbose: bool = False

    def build(self) -> list[str]:
        args = [TOOL_NAME, "--framework_runner"]
        args += ["--framework", self.framework.value]
        args += ["--model_path", self.model_path]
        for spec in self.input_tensors:
            args += spec.to_args()
        args += ["--output_tensor", self.output_tensor]
        if self.working_dir:
            args += ["--working_dir", self.working_dir]
        if self.output_dirname:
            args += ["--output_dirname", self.output_dirname]
        if self.verbose:
            args.append("--verbose")
        return args


@dataclass
class InferenceEngineArgs:
    """CLI arg builder for snpe-accuracy-debugger --inference_engine."""
    runtime: AccDebuggerRuntime
    architecture: AccDebuggerArchitecture
    input_list: str
    engine_path: str
    # Source stage args (required when stage=source)
    framework: Optional[AccDebuggerFramework] = None
    model_path: Optional[str] = None
    input_tensors: list[InputTensorSpec] = field(default_factory=list)
    output_tensor: Optional[str] = None
    # Converted/Compiled stage
    stage: InferenceEngineStage = InferenceEngineStage.SOURCE
    static_model: Optional[str] = None
    # Optional
    precision: AccPrecision = AccPrecision.INT8
    device_id: Optional[str] = None
    host_device: Optional[AccDebuggerHostDevice] = None
    working_dir: Optional[str] = None
    output_dirname: Optional[str] = None
    verbose: bool = False
    debug_mode_off: bool = False
    offline_prepare: bool = False
    htp_socs: Optional[str] = None
    profiling_level: Optional[str] = None
    bias_bitwidth: int = 8
    act_bitwidth: int = 8
    weights_bitwidth: int = 8
    use_native_input_files: bool = False
    use_native_output_files: bool = False
    quantization_overrides: Optional[str] = None
    extra_converter_args: Optional[str] = None
    extra_runtime_args: Optional[str] = None
    extra_quantizer_args: Optional[str] = None
    golden_dir_for_mapping: Optional[str] = None
    model_name: Optional[str] = None
    args_config: Optional[str] = None

    def build(self) -> list[str]:
        args = [TOOL_NAME, "--inference_engine"]
        args += ["--runtime", self.runtime.value]
        args += ["--architecture", self.architecture.value]
        args += ["--input_list", self.input_list]
        args += ["--engine_path", self.engine_path]

        if self.stage != InferenceEngineStage.SOURCE:
            args += ["--stage", self.stage.value]

        if self.framework:
            args += ["--framework", self.framework.value]
        if self.model_path:
            args += ["--model_path", self.model_path]
        for spec in self.input_tensors:
            args += spec.to_args()
        if self.output_tensor:
            args += ["--output_tensor", self.output_tensor]
        if self.static_model:
            args += ["--static_model", self.static_model]

        if self.precision != AccPrecision.INT8:
            args += ["--precision", self.precision.value]
        if self.device_id:
            args += ["--deviceId", self.device_id]
        if self.host_device:
            args += ["--host_device", self.host_device.value]
        if self.working_dir:
            args += ["--working_dir", self.working_dir]
        if self.output_dirname:
            args += ["--output_dirname", self.output_dirname]
        if self.verbose:
            args.append("--verbose")
        if self.debug_mode_off:
            args.append("--debug_mode_off")
        if self.offline_prepare:
            args.append("--offline_prepare")
        if self.htp_socs:
            args += ["--htp_socs", self.htp_socs]
        if self.profiling_level:
            args += ["--profiling_level", self.profiling_level]
        if self.bias_bitwidth != 8:
            args += ["--bias_bitwidth", str(self.bias_bitwidth)]
        if self.act_bitwidth != 8:
            args += ["--act_bitwidth", str(self.act_bitwidth)]
        if self.weights_bitwidth != 8:
            args += ["--weights_bitwidth", str(self.weights_bitwidth)]
        if self.use_native_input_files:
            args.append("--use_native_input_files")
        if self.use_native_output_files:
            args.append("--use_native_output_files")
        if self.quantization_overrides:
            args += ["--quantization_overrides", self.quantization_overrides]
        if self.extra_converter_args:
            args += ["--extra_converter_args", f"'{self.extra_converter_args}'"]
        if self.extra_runtime_args:
            args += ["--extra_runtime_args", f"'{self.extra_runtime_args}'"]
        if self.extra_quantizer_args:
            args += ["--extra_quantizer_args", f"'{self.extra_quantizer_args}'"]
        if self.golden_dir_for_mapping:
            args += ["--golden_dir_for_mapping", self.golden_dir_for_mapping]
        if self.model_name:
            args += ["--model_name", self.model_name]
        if self.args_config:
            args += ["--args_config", self.args_config]
        return args


@dataclass
class VerificationArgs:
    """CLI arg builder for snpe-accuracy-debugger --verification."""
    default_verifiers: list[VerifierSpec]
    golden_output_reference_directory: str
    inference_results: str
    tensor_mapping: Optional[str] = None
    graph_struct: Optional[str] = None
    dlc_path: Optional[str] = None
    verifier_config: Optional[str] = None
    working_dir: Optional[str] = None
    output_dirname: Optional[str] = None
    verbose: bool = False
    engine: Optional[str] = None           # qnn or snpe

    def build(self) -> list[str]:
        args = [TOOL_NAME, "--verification"]
        for v in self.default_verifiers:
            args += v.to_args("--default_verifier")
        args += [
            "--golden_output_reference_directory", self.golden_output_reference_directory,
            "--inference_results", self.inference_results,
        ]
        if self.tensor_mapping:
            args += ["--tensor_mapping", self.tensor_mapping]
        if self.graph_struct:
            args += ["--graph_struct", self.graph_struct]
        if self.dlc_path:
            args += ["--dlc_path", self.dlc_path]
        if self.verifier_config:
            args += ["--verifier_config", self.verifier_config]
        if self.working_dir:
            args += ["--working_dir", self.working_dir]
        if self.output_dirname:
            args += ["--output_dirname", self.output_dirname]
        if self.verbose:
            args.append("--verbose")
        if self.engine:
            args += ["--engine", self.engine]
        return args


@dataclass
class CompareEncodingsArgs:
    """CLI arg builder for snpe-accuracy-debugger --compare_encodings."""
    input_dlc: str
    aimet_encodings_json: str
    precision: int = 17              # Decimal places for comparison
    params_only: bool = False
    activations_only: bool = False
    specific_node: Optional[str] = None
    working_dir: Optional[str] = None
    output_dirname: Optional[str] = None
    verbose: bool = False

    def build(self) -> list[str]:
        args = [TOOL_NAME, "--compare_encodings"]
        args += ["--input", self.input_dlc]
        args += ["--aimet_encodings_json", self.aimet_encodings_json]
        if self.precision != 17:
            args += ["--precision", str(self.precision)]
        if self.params_only:
            args.append("--params_only")
        if self.activations_only:
            args.append("--activations_only")
        if self.specific_node:
            args += ["--specific_node", self.specific_node]
        if self.working_dir:
            args += ["--working_dir", self.working_dir]
        if self.output_dirname:
            args += ["--output_dirname", self.output_dirname]
        if self.verbose:
            args.append("--verbose")
        return args


@dataclass
class TensorInspectionArgs:
    """CLI arg builder for snpe-accuracy-debugger --tensor_inspection."""
    golden_data: str
    target_data: str
    verifiers: list[VerifierSpec]
    working_dir: Optional[str] = None
    data_type: Optional[TensorDataType] = None
    target_encodings: Optional[str] = None
    verbose: bool = False

    def build(self) -> list[str]:
        args = [TOOL_NAME, "--tensor_inspection"]
        args += ["--golden_data", self.golden_data]
        args += ["--target_data", self.target_data]
        for v in self.verifiers:
            args += v.to_args("--verifier")
        if self.working_dir:
            args += ["--working_dir", self.working_dir]
        if self.data_type:
            args += ["--data_type", self.data_type.value]
        if self.target_encodings:
            args += ["--target_encodings", self.target_encodings]
        if self.verbose:
            args.append("--verbose")
        return args


@dataclass
class E2EAccuracyDebuggerArgs:
    """CLI arg builder for the all-in-one E2E mode (no mode flag = runs all 3 stages).

    Runs framework_runner → inference_engine → verification sequentially.
    Use --enable_tensor_inspection for per-layer plots (increases runtime).
    Use --deep_analyzer for deep model dissection analysis.
    """
    framework: AccDebuggerFramework
    model_path: str
    input_tensors: list[InputTensorSpec]
    output_tensor: str
    runtime: AccDebuggerRuntime
    architecture: AccDebuggerArchitecture
    input_list: str
    default_verifiers: list[VerifierSpec]
    working_dir: Optional[str] = None
    output_dirname: Optional[str] = None
    verbose: bool = False
    enable_tensor_inspection: bool = False
    golden_output_reference_directory: Optional[str] = None
    deep_analyzer: Optional[str] = None     # e.g. "modelDissectionAnalyzer"

    def build(self) -> list[str]:
        """Build E2E command (no mode flag — runs all three stages)."""
        args = [TOOL_NAME]
        args += ["--framework", self.framework.value]
        args += ["--model_path", self.model_path]
        for spec in self.input_tensors:
            args += spec.to_args()
        args += ["--output_tensor", self.output_tensor]
        args += ["--runtime", self.runtime.value]
        args += ["--architecture", self.architecture.value]
        args += ["--input_list", self.input_list]
        for v in self.default_verifiers:
            args += v.to_args("--default_verifier")
        if self.working_dir:
            args += ["--working_dir", self.working_dir]
        if self.output_dirname:
            args += ["--output_dirname", self.output_dirname]
        if self.verbose:
            args.append("--verbose")
        if self.enable_tensor_inspection:
            args.append("--enable_tensor_inspection")
        if self.golden_output_reference_directory:
            args += ["--golden_output_reference_directory",
                     self.golden_output_reference_directory]
        if self.deep_analyzer:
            args += ["--deep_analyzer", self.deep_analyzer]
        return args


# ══════════════════════════════════════════════════════════════════════════════
# Config Models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantCheckerAlgorithmConfig:
    """One algorithm entry in the quantization checker config."""
    algo_name: QuantAlgorithm
    threshold: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"algo_name": self.algo_name.value}
        if self.threshold is not None:
            d["threshold"] = str(self.threshold)
        return d


@dataclass
class QuantCheckerConfig:
    """Quantization checker config file structure.

    Written to a JSON file and passed as --config_file to quant_checker.

    Default algorithms from documentation sample:
      weights/biases: minmax, maxdiff, sqnr, stats, data_range_analyzer,
                      data_distribution_analyzer
      activations: minmax, data_range_analyzer
    """
    weight_algorithms: list[QuantCheckerAlgorithmConfig] = field(default_factory=list)
    bias_algorithms: list[QuantCheckerAlgorithmConfig] = field(default_factory=list)
    act_algorithms: list[QuantCheckerAlgorithmConfig] = field(default_factory=list)
    input_data_algorithms: list[QuantCheckerAlgorithmConfig] = field(default_factory=list)
    quantization_algorithms: list[str] = field(default_factory=lambda: ["cle", "None"])
    quantization_variations: list[str] = field(
        default_factory=lambda: ["tf", "enhanced", "symmetric", "asymmetric"]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "WEIGHT_COMPARISON_ALGORITHMS": [a.to_dict() for a in self.weight_algorithms],
            "BIAS_COMPARISON_ALGORITHMS": [a.to_dict() for a in self.bias_algorithms],
            "ACT_COMPARISON_ALGORITHMS": [a.to_dict() for a in self.act_algorithms],
            "INPUT_DATA_ANALYSIS_ALGORITHMS": [a.to_dict() for a in self.input_data_algorithms],
            "QUANTIZATION_ALGORITHMS": self.quantization_algorithms,
            "QUANTIZATION_VARIATIONS": self.quantization_variations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def default(cls) -> "QuantCheckerConfig":
        """Create the default config matching the documentation sample."""
        standard_algos = [
            QuantCheckerAlgorithmConfig(QuantAlgorithm.MINMAX, 10),
            QuantCheckerAlgorithmConfig(QuantAlgorithm.MAXDIFF, 10),
            QuantCheckerAlgorithmConfig(QuantAlgorithm.SQNR, 26),
            QuantCheckerAlgorithmConfig(QuantAlgorithm.STATS, 2),
            QuantCheckerAlgorithmConfig(QuantAlgorithm.DATA_RANGE_ANALYZER),
            QuantCheckerAlgorithmConfig(QuantAlgorithm.DATA_DISTRIBUTION_ANALYZER, 0.6),
        ]
        return cls(
            weight_algorithms=list(standard_algos),
            bias_algorithms=list(standard_algos),
            act_algorithms=[
                QuantCheckerAlgorithmConfig(QuantAlgorithm.MINMAX, 10),
                QuantCheckerAlgorithmConfig(QuantAlgorithm.DATA_RANGE_ANALYZER),
            ],
            input_data_algorithms=[
                QuantCheckerAlgorithmConfig(QuantAlgorithm.STATS, 2),
            ],
        )


@dataclass
class VerifierConfig:
    """Verification config file structure (verifier_config JSON).

    Maps verifier names to their parameters and the specific tensors they apply to.
    The "tensors" field is a list of lists because some verifiers (e.g. MeanIOU)
    operate on two tensors simultaneously.
    """
    verifiers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_verifier(
        self,
        verifier: VerifierType,
        parameters: dict[str, Any],
        tensors: list[list[str]],
    ) -> None:
        """Add a verifier configuration."""
        self.verifiers[verifier.value] = {
            "parameters": parameters,
            "tensors": tensors,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.verifiers

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: str) -> None:
        Path(path).write_text(self.to_json())


@dataclass
class TensorMappingConfig:
    """Tensor mapping JSON: maps inference engine tensor names to framework tensor names.

    When tensor_mapping is not provided, the tool assumes identical names.
    """
    mappings: dict[str, str] = field(default_factory=dict)  # inference → framework

    def add(self, inference_name: str, framework_name: str) -> None:
        self.mappings[inference_name] = framework_name

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.mappings, indent=indent)

    def write(self, path: str) -> None:
        Path(path).write_text(self.to_json())


# ══════════════════════════════════════════════════════════════════════════════
# Output Path Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_framework_runner_output_dir(working_dir: str = DEFAULT_WORKING_DIR) -> str:
    """Path to the latest framework runner output directory."""
    return str(Path(working_dir) / OUTPUT_DIRS["framework_runner"] / "latest")


def get_inference_engine_output_dir(working_dir: str = DEFAULT_WORKING_DIR) -> str:
    """Path to the latest inference engine output directory."""
    return str(Path(working_dir) / OUTPUT_DIRS["inference_engine"] / "latest")


def get_tensor_mapping_path(working_dir: str = DEFAULT_WORKING_DIR) -> str:
    """Path to the tensor_mapping.json generated by inference engine."""
    return str(
        Path(working_dir) / OUTPUT_DIRS["inference_engine"] / "latest" /
        KEY_OUTPUT_FILES["tensor_mapping"]
    )


def get_graph_struct_path(
    working_dir: str = DEFAULT_WORKING_DIR,
    model_name: str = "snpe_model",
) -> str:
    """Path to the *_graph_struct.json generated by inference engine."""
    return str(
        Path(working_dir) / OUTPUT_DIRS["inference_engine"] / "latest" /
        f"{model_name}_graph_struct.json"
    )


def get_verification_result_dir(
    working_dir: str = DEFAULT_WORKING_DIR,
    result_index: int = 0,
) -> str:
    """Path to a specific inference result directory (Result_N).

    Multiple result directories are created when multiple images are in input_list.
    Match result index to the position of the image in input_list.txt.
    """
    return str(
        Path(working_dir) / OUTPUT_DIRS["inference_engine"] / "latest" /
        "output" / f"Result_{result_index}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

ACCURACY_DEBUGGER_NOTES: dict[str, Any] = {
    "description": (
        "Experimental tool to find inaccuracies at the layer level. "
        "Compares ML framework golden outputs with SNPE inference engine outputs."
    ),
    "supported_models": "ONNX, TFLite, TensorFlow 1.x",
    "tool_name": TOOL_NAME,
    "modes": {
        "framework_runner": (
            "Runs model through ML framework (TF/ONNX/TFLite). "
            "Produces golden .raw outputs at each layer."
        ),
        "inference_engine": (
            "Runs model through SNPE on CPU/GPU/DSP. "
            "Produces inference .raw outputs + tensor_mapping.json + graph_struct.json."
        ),
        "verification": (
            "Compares golden vs inference outputs using verifiers. "
            "Produces per-tensor CSV/HTML + summary."
        ),
        "compare_encodings": (
            "Compares SNPE DLC encodings with AIMET encodings. "
            "Outputs Excel file with mismatches highlighted."
        ),
        "tensor_inspection": (
            "Compares target vs reference tensors. "
            "Plots histograms, CDF, deviation graphs, summary CSV."
        ),
        "quant_checker": (
            "Analyzes quantization quality: weights, biases, activations. "
            "Outputs CSV, HTML, histograms per quantization variation."
        ),
        "e2e": (
            "No mode flag = runs framework_runner → inference_engine → verification "
            "sequentially."
        ),
    },
    "verifiers": {
        "CosineSimilarity": "Range [0,1]; higher = more similar (aim for > 0.99)",
        "SQNR": "Signal-to-quantization noise ratio; higher = better",
        "RtolAtol": "Relative + absolute tolerance; pass/fail per element",
        "AdjustedRtolAtol": "RtolAtol with adjustments",
        "TopK": "Compares top-K class predictions",
        "L1Error": "L1 norm difference",
        "MSE": "Mean squared error",
        "MAE": "Mean absolute error",
        "MeanIOU": "Mean intersection-over-union (for detection, two tensors at a time)",
        "ScaledDiff": "Scaled difference (requires graph_struct)",
    },
    "output_structure": {
        "framework_runner": "working_dir/framework_runner/YYYY-MM-DD_HH:mm:ss/ + latest →",
        "inference_engine": "working_dir/inference_engine/ + latest →",
        "verification": "working_dir/verification/ + latest →",
        "compare_encodings": "working_dir/compare_encodings/ + latest →",
    },
    "key_output_files": KEY_OUTPUT_FILES,
    "result_index_tip": (
        "When input_list.txt has N images, inference_engine produces output/Result_0 … "
        "output/Result_N-1. Match result index to position of image in input_list.txt. "
        "Use Result_0 if input image was first in the list."
    ),
    "tf_tensor_naming": (
        "For TensorFlow, add :0 index to tensor names in framework_runner "
        "(e.g. 'input:0', 'InceptionV3/Predictions/Reshape_1:0'). "
        "Drop :0 for inference_engine and ONNX models."
    ),
    "working_dir_tip": (
        "Call all scripts from the same directory to keep outputs consolidated "
        "under a single working_directory."
    ),
    "windows_note": (
        "On Windows x86 / Windows on Snapdragon, only CPU runtime is currently tested."
    ),
    "tensor_inspection_note": (
        "Only data with matching target/golden filenames is inspected. "
        "Golden and target tensors must have same dimensions, datatypes, and layouts. "
        "Calibrated min/max requires target_encodings file; density plot skipped without it."
    ),
    "enable_tensor_inspection_warning": (
        "--enable_tensor_inspection significantly increases execution time for large models. "
        "Omit for faster runs."
    ),
    "quant_algorithms": {
        "minmax": "Compares min/max values between quantized and float",
        "maxdiff": "Maximum absolute difference",
        "sqnr": "Signal-to-quantization noise ratio",
        "stats": "Statistical comparison (mean, std)",
        "data_range_analyzer": "Analyzes data range suitability for quantization",
        "data_distribution_analyzer": "Analyzes data distribution quality",
    },
}
