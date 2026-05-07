"""Tests for SNPE Accuracy Debugger module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quad.sdk_tools.accuracy_debugger import (
    ACCURACY_DEBUGGER_NOTES,
    OUTPUT_DIRS,
    TOOL_NAME,
    AccDebuggerArchitecture,
    AccDebuggerFramework,
    AccDebuggerMode,
    AccDebuggerRuntime,
    AccPrecision,
    CompareEncodingsArgs,
    E2EAccuracyDebuggerArgs,
    FrameworkRunnerArgs,
    InferenceEngineArgs,
    InferenceEngineStage,
    InputTensorSpec,
    QuantAlgorithm,
    QuantCheckerAlgorithmConfig,
    QuantCheckerConfig,
    TensorDataType,
    TensorInspectionArgs,
    TensorMappingConfig,
    VerificationArgs,
    VerifierConfig,
    VerifierSpec,
    VerifierType,
    get_framework_runner_output_dir,
    get_graph_struct_path,
    get_inference_engine_output_dir,
    get_tensor_mapping_path,
    get_verification_result_dir,
)


# ══════════════════════════════════════════════════════════════════════════════
# InputTensorSpec
# ══════════════════════════════════════════════════════════════════════════════

class TestInputTensorSpec:
    def test_dim_string(self) -> None:
        spec = InputTensorSpec("data", (1, 224, 224, 3), "data.raw")
        assert spec.dim_string == "1,224,224,3"

    def test_to_args_basic(self) -> None:
        spec = InputTensorSpec("data", (1, 224, 224, 3), "data.raw")
        args = spec.to_args()
        assert "--input_tensor" in args
        assert '"data"' in args
        assert "1,224,224,3" in args
        assert "data.raw" in args

    def test_to_args_includes_dtype_when_not_float32(self) -> None:
        spec = InputTensorSpec("data", (1, 224, 224, 3), "data.raw", dtype="uint8")
        args = spec.to_args()
        assert "uint8" in args

    def test_to_args_no_dtype_for_float32(self) -> None:
        spec = InputTensorSpec("data", (1, 224, 224, 3), "data.raw", dtype="float32")
        args = spec.to_args()
        assert "float32" not in args  # default not emitted

    def test_tf_tensor_name_with_colon(self) -> None:
        """TF tensors use 'input:0' naming with colon-zero."""
        spec = InputTensorSpec("input:0", (1, 299, 299, 3), "chairs.raw")
        args = spec.to_args()
        assert '"input:0"' in args


# ══════════════════════════════════════════════════════════════════════════════
# VerifierSpec
# ══════════════════════════════════════════════════════════════════════════════

class TestVerifierSpec:
    def test_cosine_similarity_basic(self) -> None:
        v = VerifierSpec(VerifierType.COSINE_SIMILARITY)
        args = v.to_args()
        assert "--default_verifier" in args
        assert "CosineSimilarity" in args

    def test_cosine_with_params(self) -> None:
        v = VerifierSpec(VerifierType.COSINE_SIMILARITY, params={"param1": 1, "param2": 2})
        args = v.to_args()
        assert "param1" in args
        assert "1" in args
        assert "param2" in args
        assert "2" in args

    def test_rtolatol_comma_params(self) -> None:
        v = VerifierSpec(
            VerifierType.RTOLATOL,
            comma_params="rtolmargin,0.01,atolmargin,0.01",
        )
        args = v.to_args()
        assert "RtolAtol,rtolmargin,0.01,atolmargin,0.01" in args

    def test_sqnr_with_params(self) -> None:
        v = VerifierSpec(VerifierType.SQNR, params={"param1": 5, "param2": 1})
        args = v.to_args()
        assert "SQNR" in args

    def test_custom_flag(self) -> None:
        v = VerifierSpec(VerifierType.COSINE_SIMILARITY)
        args = v.to_args("--verifier")
        assert "--verifier" in args
        assert "--default_verifier" not in args

    def test_flag_value_basic(self) -> None:
        v = VerifierSpec(VerifierType.MSE)
        assert v.to_flag_value() == "MSE"

    def test_flag_value_with_comma_params(self) -> None:
        v = VerifierSpec(VerifierType.RTOLATOL, comma_params="rtolmargin,0.01")
        assert v.to_flag_value() == "RtolAtol,rtolmargin,0.01"


# ══════════════════════════════════════════════════════════════════════════════
# FrameworkRunnerArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestFrameworkRunnerArgs:
    def _make(self) -> FrameworkRunnerArgs:
        return FrameworkRunnerArgs(
            framework=AccDebuggerFramework.TENSORFLOW,
            model_path="/models/inception_v3.pb",
            input_tensors=[InputTensorSpec("input:0", (1, 299, 299, 3), "chairs.raw")],
            output_tensor="InceptionV3/Predictions/Reshape_1:0",
        )

    def test_framework_runner_flag(self) -> None:
        args = self._make().build()
        assert "--framework_runner" in args

    def test_tool_name(self) -> None:
        assert TOOL_NAME in self._make().build()

    def test_framework_arg(self) -> None:
        args = self._make().build()
        assert "--framework" in args
        assert "tensorflow" in args

    def test_model_path_arg(self) -> None:
        args = self._make().build()
        assert "--model_path" in args
        assert "/models/inception_v3.pb" in args

    def test_output_tensor_arg(self) -> None:
        args = self._make().build()
        assert "--output_tensor" in args
        assert "InceptionV3/Predictions/Reshape_1:0" in args

    def test_input_tensor_present(self) -> None:
        args = self._make().build()
        assert "--input_tensor" in args
        assert "1,299,299,3" in args

    def test_working_dir_optional(self) -> None:
        a = FrameworkRunnerArgs(
            AccDebuggerFramework.ONNX,
            "model.onnx",
            [InputTensorSpec("Input", (1, 3, 513, 513), "data.raw")],
            "Output",
            working_dir="./my_wd",
        )
        assert "--working_dir" in a.build()
        assert "./my_wd" in a.build()

    def test_verbose_flag(self) -> None:
        a = FrameworkRunnerArgs(
            AccDebuggerFramework.ONNX, "m.onnx",
            [InputTensorSpec("in", (1, 3, 224, 224), "d.raw")],
            "out", verbose=True,
        )
        assert "--verbose" in a.build()

    def test_documentation_tensorflow_sample(self) -> None:
        """Reproduce the InceptionV3 TF sample command from documentation."""
        a = FrameworkRunnerArgs(
            framework=AccDebuggerFramework.TENSORFLOW,
            model_path="$RESOURCESPATH/samples/InceptionV3Model/inception_v3_2016_08_28_frozen.pb",
            input_tensors=[InputTensorSpec(
                "input:0", (1, 299, 299, 3),
                "$RESOURCESPATH/samples/InceptionV3Model/data/chairs.raw",
            )],
            output_tensor="InceptionV3/Predictions/Reshape_1:0",
        )
        args = a.build()
        assert "--framework_runner" in args
        assert "tensorflow" in args
        assert '"input:0"' in args

    def test_documentation_onnx_sample(self) -> None:
        """Reproduce the ONNX sample command from documentation."""
        a = FrameworkRunnerArgs(
            framework=AccDebuggerFramework.ONNX,
            model_path="dlv3plus_mbnet_513-513_op9_mod_basic.onnx",
            input_tensors=[InputTensorSpec(
                "Input", (1, 3, 513, 513), "00000_1_3_513_513.raw",
            )],
            output_tensor="Output",
        )
        args = a.build()
        assert "onnx" in args
        assert "Input" in " ".join(args)  # no :0 for ONNX


# ══════════════════════════════════════════════════════════════════════════════
# InferenceEngineArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestInferenceEngineArgs:
    def _make(self, runtime=AccDebuggerRuntime.CPU,
              arch=AccDebuggerArchitecture.X86_64_LINUX_CLANG) -> InferenceEngineArgs:
        return InferenceEngineArgs(
            runtime=runtime,
            architecture=arch,
            input_list="image_list.txt",
            engine_path="/sdk",
            framework=AccDebuggerFramework.TENSORFLOW,
            model_path="/models/inception_v3.pb",
            input_tensors=[InputTensorSpec("input:0", (1, 299, 299, 3), "chairs.raw")],
            output_tensor="InceptionV3/Predictions/Reshape_1:0",
        )

    def test_inference_engine_flag(self) -> None:
        assert "--inference_engine" in self._make().build()

    def test_runtime_cpu(self) -> None:
        args = self._make().build()
        assert "--runtime" in args
        assert "cpu" in args

    def test_architecture_x86(self) -> None:
        args = self._make().build()
        assert "--architecture" in args
        assert "x86_64-linux-clang" in args

    def test_dsp_runtime(self) -> None:
        a = self._make(runtime=AccDebuggerRuntime.DSP,
                       arch=AccDebuggerArchitecture.AARCH64_ANDROID)
        args = a.build()
        assert "dsp" in args
        assert "aarch64-android" in args

    def test_device_id(self) -> None:
        a = self._make()
        a.device_id = "357415c4"
        assert "357415c4" in a.build()

    def test_offline_prepare(self) -> None:
        a = self._make()
        a.offline_prepare = True
        assert "--offline_prepare" in a.build()

    def test_profiling_level(self) -> None:
        a = self._make()
        a.profiling_level = "detailed"
        args = a.build()
        assert "--profiling_level" in args
        assert "detailed" in args

    def test_htp_socs(self) -> None:
        a = self._make()
        a.htp_socs = "sm8750"
        assert "sm8750" in a.build()

    def test_act_bitwidth_16(self) -> None:
        a = self._make()
        a.act_bitwidth = 16
        assert "--act_bitwidth" in a.build()
        assert "16" in a.build()

    def test_extra_converter_args(self) -> None:
        a = self._make()
        a.extra_converter_args = "input_dtype=data float;input_layout=data1 NCHW"
        args = a.build()
        assert "--extra_converter_args" in args

    def test_windows_snapdragon_architecture(self) -> None:
        a = self._make(arch=AccDebuggerArchitecture.WOS)
        assert "wos" in a.build()

    def test_documentation_android_dsp_sample(self) -> None:
        """Reproduce the Android DSP sample command."""
        a = InferenceEngineArgs(
            runtime=AccDebuggerRuntime.DSP,
            architecture=AccDebuggerArchitecture.AARCH64_ANDROID,
            input_list="image_list.txt",
            engine_path="/sdk",
            device_id="357415c4",
            framework=AccDebuggerFramework.TENSORFLOW,
            model_path="inception_v3.pb",
            input_tensors=[InputTensorSpec("input:0", (1, 299, 299, 3), "chairs.raw")],
            output_tensor="InceptionV3/Predictions/Reshape_1:0",
            verbose=True,
        )
        args = a.build()
        assert "357415c4" in args
        assert "dsp" in args
        assert "aarch64-android" in args


# ══════════════════════════════════════════════════════════════════════════════
# VerificationArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestVerificationArgs:
    def test_verification_flag(self) -> None:
        a = VerificationArgs(
            default_verifiers=[VerifierSpec(VerifierType.COSINE_SIMILARITY)],
            golden_output_reference_directory="./fw_runner/latest/",
            inference_results="./inf_engine/latest/output/Result_0/",
        )
        args = a.build()
        assert "--verification" in args

    def test_multiple_verifiers(self) -> None:
        a = VerificationArgs(
            default_verifiers=[
                VerifierSpec(VerifierType.COSINE_SIMILARITY, params={"param1": 1, "param2": 2}),
                VerifierSpec(VerifierType.SQNR, params={"param1": 5, "param2": 1}),
            ],
            golden_output_reference_directory="./fw/",
            inference_results="./inf/",
        )
        args = a.build()
        assert args.count("--default_verifier") == 2

    def test_tensor_mapping_and_graph_struct(self) -> None:
        a = VerificationArgs(
            default_verifiers=[VerifierSpec(VerifierType.COSINE_SIMILARITY)],
            golden_output_reference_directory="./fw/",
            inference_results="./inf/",
            tensor_mapping="tensor_mapping.json",
            graph_struct="snpe_model_graph_struct.json",
        )
        args = a.build()
        assert "--tensor_mapping" in args
        assert "tensor_mapping.json" in args
        assert "--graph_struct" in args

    def test_documentation_verification_sample(self) -> None:
        """Reproduce the verification sample command from documentation."""
        a = VerificationArgs(
            default_verifiers=[
                VerifierSpec(VerifierType.COSINE_SIMILARITY, params={"param1": 1, "param2": 2}),
                VerifierSpec(VerifierType.SQNR, params={"param1": 5, "param2": 1}),
            ],
            golden_output_reference_directory="working_directory/framework_runner/2022-10-31_17-07-58/",
            inference_results="working_directory/inference_engine/latest/output/Result_0/",
            tensor_mapping="working_directory/inference_engine/latest/tensor_mapping.json",
            graph_struct="working_directory/inference_engine/latest/snpe_model_graph_struct.json",
        )
        args = a.build()
        assert "--verification" in args
        assert args.count("--default_verifier") == 2
        assert "tensor_mapping.json" in " ".join(args)


# ══════════════════════════════════════════════════════════════════════════════
# CompareEncodingsArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestCompareEncodingsArgs:
    def test_compare_encodings_flag(self) -> None:
        a = CompareEncodingsArgs("mv2_quantized.dlc", "aimet_encodings.json")
        assert "--compare_encodings" in a.build()

    def test_required_args(self) -> None:
        a = CompareEncodingsArgs("mv2_quantized.dlc", "aimet_encodings.json")
        args = a.build()
        assert "--input" in args
        assert "mv2_quantized.dlc" in args
        assert "--aimet_encodings_json" in args

    def test_params_only(self) -> None:
        a = CompareEncodingsArgs("m.dlc", "enc.json", params_only=True)
        assert "--params_only" in a.build()

    def test_activations_only(self) -> None:
        a = CompareEncodingsArgs("m.dlc", "enc.json", activations_only=True)
        assert "--activations_only" in a.build()

    def test_specific_node(self) -> None:
        a = CompareEncodingsArgs("m.dlc", "enc.json", specific_node="/1/Conv_output_0")
        args = a.build()
        assert "--specific_node" in args
        assert "/1/Conv_output_0" in args

    def test_precision_override(self) -> None:
        a = CompareEncodingsArgs("m.dlc", "enc.json", precision=10)
        assert "--precision" in a.build()
        assert "10" in a.build()

    def test_default_precision_not_emitted(self) -> None:
        a = CompareEncodingsArgs("m.dlc", "enc.json", precision=17)
        assert "--precision" not in a.build()

    def test_documentation_samples(self) -> None:
        """Reproduce the four documented sample commands."""
        base = CompareEncodingsArgs("mv2_quantized.dlc", "aimet_encodings.json")
        assert "--compare_encodings" in base.build()

        params = CompareEncodingsArgs("mv2_quantized.dlc", "aimet_encodings.json",
                                      params_only=True)
        assert "--params_only" in params.build()

        acts = CompareEncodingsArgs("mv2_quantized.dlc", "aimet_encodings.json",
                                    activations_only=True)
        assert "--activations_only" in acts.build()

        node = CompareEncodingsArgs("mv2_quantized.dlc", "aimet_encodings.json",
                                    specific_node="/1/Conv_output_0")
        assert "/1/Conv_output_0" in node.build()


# ══════════════════════════════════════════════════════════════════════════════
# TensorInspectionArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestTensorInspectionArgs:
    def test_tensor_inspection_flag(self) -> None:
        a = TensorInspectionArgs(
            golden_data="golden/", target_data="target/",
            verifiers=[VerifierSpec(VerifierType.SQNR)],
        )
        assert "--tensor_inspection" in a.build()

    def test_golden_and_target(self) -> None:
        a = TensorInspectionArgs(
            golden_data="golden_tensors/", target_data="target_tensors/",
            verifiers=[VerifierSpec(VerifierType.MSE)],
        )
        args = a.build()
        assert "--golden_data" in args
        assert "golden_tensors/" in args
        assert "--target_data" in args

    def test_multiple_verifiers(self) -> None:
        a = TensorInspectionArgs(
            golden_data="g/", target_data="t/",
            verifiers=[
                VerifierSpec(VerifierType.MSE),
                VerifierSpec(VerifierType.SQNR),
                VerifierSpec(VerifierType.RTOLATOL, comma_params="rtolmargin,0.01,atolmargin,0.01"),
            ],
        )
        args = a.build()
        assert args.count("--verifier") == 3

    def test_target_encodings(self) -> None:
        a = TensorInspectionArgs(
            "g/", "t/",
            verifiers=[VerifierSpec(VerifierType.SQNR)],
            target_encodings="reference_encodings.json",
        )
        assert "--target_encodings" in a.build()

    def test_data_type(self) -> None:
        a = TensorInspectionArgs(
            "g/", "t/",
            verifiers=[VerifierSpec(VerifierType.MSE)],
            data_type=TensorDataType.INT8,
        )
        assert "--data_type" in a.build()
        assert "int8" in a.build()


# ══════════════════════════════════════════════════════════════════════════════
# E2EAccuracyDebuggerArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestE2EAccuracyDebuggerArgs:
    def _make(self) -> E2EAccuracyDebuggerArgs:
        return E2EAccuracyDebuggerArgs(
            framework=AccDebuggerFramework.TENSORFLOW,
            model_path="/models/inception_v3.pb",
            input_tensors=[InputTensorSpec("input:0", (1, 299, 299, 3), "chairs.raw")],
            output_tensor="InceptionV3/Predictions/Reshape_1:0",
            runtime=AccDebuggerRuntime.CPU,
            architecture=AccDebuggerArchitecture.X86_64_LINUX_CLANG,
            input_list="image_list.txt",
            default_verifiers=[VerifierSpec(VerifierType.COSINE_SIMILARITY)],
        )

    def test_no_mode_flag_in_e2e(self) -> None:
        """E2E mode has no --framework_runner / --inference_engine / --verification flag."""
        args = self._make().build()
        assert "--framework_runner" not in args
        assert "--inference_engine" not in args
        assert "--verification" not in args

    def test_tool_name_present(self) -> None:
        assert TOOL_NAME in self._make().build()

    def test_all_required_args_present(self) -> None:
        args = self._make().build()
        assert "--framework" in args
        assert "--model_path" in args
        assert "--runtime" in args
        assert "--architecture" in args
        assert "--input_list" in args
        assert "--default_verifier" in args

    def test_enable_tensor_inspection(self) -> None:
        a = self._make()
        a.enable_tensor_inspection = True
        assert "--enable_tensor_inspection" in a.build()

    def test_deep_analyzer(self) -> None:
        a = self._make()
        a.deep_analyzer = "modelDissectionAnalyzer"
        args = a.build()
        assert "--deep_analyzer" in args
        assert "modelDissectionAnalyzer" in args

    def test_documentation_e2e_sample(self) -> None:
        """Reproduce the E2E sample command from documentation."""
        a = E2EAccuracyDebuggerArgs(
            framework=AccDebuggerFramework.TENSORFLOW,
            model_path="$RESOURCESPATH/samples/InceptionV3Model/inception_v3_2016_08_28_frozen.pb",
            input_tensors=[InputTensorSpec(
                "input:0", (1, 299, 299, 3),
                "$RESOURCESPATH/samples/InceptionV3Model/data/chairs.raw",
            )],
            output_tensor="InceptionV3/Predictions/Reshape_1:0",
            runtime=AccDebuggerRuntime.CPU,
            architecture=AccDebuggerArchitecture.X86_64_LINUX_CLANG,
            input_list="$RESOURCESPATH/samples/InceptionV3Model/data/image_list.txt",
            default_verifiers=[VerifierSpec(VerifierType.COSINE_SIMILARITY)],
            enable_tensor_inspection=True,
            verbose=True,
        )
        args = a.build()
        assert "--enable_tensor_inspection" in args
        assert "--verbose" in args
        assert "CosineSimilarity" in args


# ══════════════════════════════════════════════════════════════════════════════
# QuantCheckerConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantCheckerConfig:
    def test_default_config_structure(self) -> None:
        cfg = QuantCheckerConfig.default()
        d = cfg.to_dict()
        assert "WEIGHT_COMPARISON_ALGORITHMS" in d
        assert "BIAS_COMPARISON_ALGORITHMS" in d
        assert "ACT_COMPARISON_ALGORITHMS" in d
        assert "QUANTIZATION_ALGORITHMS" in d
        assert "QUANTIZATION_VARIATIONS" in d

    def test_default_has_standard_algorithms(self) -> None:
        cfg = QuantCheckerConfig.default()
        weight_names = [a.algo_name.value for a in cfg.weight_algorithms]
        assert "minmax" in weight_names
        assert "maxdiff" in weight_names
        assert "sqnr" in weight_names
        assert "data_range_analyzer" in weight_names

    def test_default_quantization_variations(self) -> None:
        cfg = QuantCheckerConfig.default()
        assert "tf" in cfg.quantization_variations
        assert "enhanced" in cfg.quantization_variations
        assert "symmetric" in cfg.quantization_variations
        assert "asymmetric" in cfg.quantization_variations

    def test_default_quantization_algorithms(self) -> None:
        cfg = QuantCheckerConfig.default()
        assert "cle" in cfg.quantization_algorithms

    def test_to_json_valid(self) -> None:
        cfg = QuantCheckerConfig.default()
        parsed = json.loads(cfg.to_json())
        assert "WEIGHT_COMPARISON_ALGORITHMS" in parsed

    def test_algorithm_with_threshold(self) -> None:
        algo = QuantCheckerAlgorithmConfig(QuantAlgorithm.MINMAX, threshold=10)
        d = algo.to_dict()
        assert d["algo_name"] == "minmax"
        assert d["threshold"] == "10"

    def test_algorithm_without_threshold(self) -> None:
        algo = QuantCheckerAlgorithmConfig(QuantAlgorithm.DATA_RANGE_ANALYZER)
        d = algo.to_dict()
        assert d["algo_name"] == "data_range_analyzer"
        assert "threshold" not in d

    def test_write_creates_file(self, tmp_path: Path) -> None:
        cfg = QuantCheckerConfig.default()
        path = str(tmp_path / "quant_config.json")
        cfg.write(path)
        data = json.loads(open(path).read())
        assert "WEIGHT_COMPARISON_ALGORITHMS" in data


# ══════════════════════════════════════════════════════════════════════════════
# VerifierConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestVerifierConfig:
    def test_add_verifier(self) -> None:
        cfg = VerifierConfig()
        cfg.add_verifier(
            VerifierType.TOPK,
            parameters={"k": 5, "ordered": False},
            tensors=[["Reshape_1:0"], ["detection_classes:0"]],
        )
        d = cfg.to_dict()
        assert "TopK" in d
        assert d["TopK"]["parameters"]["k"] == 5

    def test_meaniou_two_tensor_format(self) -> None:
        """MeanIOU runs on two tensors simultaneously — list of two."""
        cfg = VerifierConfig()
        cfg.add_verifier(
            VerifierType.MEAN_IOU,
            parameters={"background_classification": 1.0},
            tensors=[["detection_boxes", "detection_classes:0"]],
        )
        d = cfg.to_dict()
        assert "MeanIOU" in d
        assert len(d["MeanIOU"]["tensors"][0]) == 2

    def test_to_json_valid(self) -> None:
        cfg = VerifierConfig()
        cfg.add_verifier(VerifierType.COSINE_SIMILARITY, {}, [["output:0"]])
        parsed = json.loads(cfg.to_json())
        assert "CosineSimilarity" in parsed

    def test_write_creates_file(self, tmp_path: Path) -> None:
        cfg = VerifierConfig()
        cfg.add_verifier(VerifierType.MSE, {}, [["out"]])
        path = str(tmp_path / "verifier_config.json")
        cfg.write(path)
        assert Path(path).exists()


# ══════════════════════════════════════════════════════════════════════════════
# TensorMappingConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestTensorMappingConfig:
    def test_add_mapping(self) -> None:
        cfg = TensorMappingConfig()
        cfg.add("Postprocessor/BatchMultiClassNonMaxSuppression_boxes", "detection_boxes:0")
        cfg.add("Postprocessor/BatchMultiClassNonMaxSuppression_scores", "detection_scores:0")
        d = json.loads(cfg.to_json())
        assert "Postprocessor/BatchMultiClassNonMaxSuppression_boxes" in d
        assert d["Postprocessor/BatchMultiClassNonMaxSuppression_boxes"] == "detection_boxes:0"

    def test_write_creates_file(self, tmp_path: Path) -> None:
        cfg = TensorMappingConfig()
        cfg.add("tensor_a", "golden_tensor_a")
        path = str(tmp_path / "tensor_mapping.json")
        cfg.write(path)
        data = json.loads(open(path).read())
        assert data["tensor_a"] == "golden_tensor_a"


# ══════════════════════════════════════════════════════════════════════════════
# Output Path Helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputPathHelpers:
    def test_framework_runner_output_dir(self) -> None:
        path = get_framework_runner_output_dir("my_wd")
        assert "my_wd" in path
        assert "framework_runner" in path
        assert "latest" in path

    def test_inference_engine_output_dir(self) -> None:
        path = get_inference_engine_output_dir("my_wd")
        assert "inference_engine" in path

    def test_tensor_mapping_path(self) -> None:
        path = get_tensor_mapping_path("my_wd")
        assert "tensor_mapping.json" in path
        assert "inference_engine" in path

    def test_graph_struct_path(self) -> None:
        path = get_graph_struct_path("my_wd", "inception_v3")
        assert "inception_v3_graph_struct.json" in path

    def test_verification_result_dir_index_0(self) -> None:
        path = get_verification_result_dir("my_wd", result_index=0)
        assert "Result_0" in path

    def test_verification_result_dir_index_2(self) -> None:
        path = get_verification_result_dir("my_wd", result_index=2)
        assert "Result_2" in path


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracyDebuggerNotes:
    def test_all_modes_documented(self) -> None:
        modes = ACCURACY_DEBUGGER_NOTES["modes"]
        for mode in ("framework_runner", "inference_engine", "verification",
                     "compare_encodings", "tensor_inspection", "quant_checker", "e2e"):
            assert mode in modes

    def test_all_verifiers_documented(self) -> None:
        verifiers = ACCURACY_DEBUGGER_NOTES["verifiers"]
        for v in ("CosineSimilarity", "SQNR", "RtolAtol", "TopK", "MSE"):
            assert v in verifiers

    def test_cosine_similarity_range_documented(self) -> None:
        desc = ACCURACY_DEBUGGER_NOTES["verifiers"]["CosineSimilarity"]
        assert "0" in desc and "1" in desc

    def test_output_structure_has_latest_links(self) -> None:
        struct = ACCURACY_DEBUGGER_NOTES["output_structure"]
        for key in ("framework_runner", "inference_engine", "verification"):
            assert "latest" in struct[key]

    def test_tf_tensor_naming_note(self) -> None:
        note = ACCURACY_DEBUGGER_NOTES["tf_tensor_naming"]
        assert ":0" in note
        assert "ONNX" in note

    def test_result_index_tip(self) -> None:
        tip = ACCURACY_DEBUGGER_NOTES["result_index_tip"]
        assert "Result_0" in tip

    def test_enable_tensor_inspection_warning(self) -> None:
        warning = ACCURACY_DEBUGGER_NOTES["enable_tensor_inspection_warning"]
        assert "time" in warning.lower() or "slow" in warning.lower()

    def test_windows_note(self) -> None:
        note = ACCURACY_DEBUGGER_NOTES["windows_note"]
        assert "CPU" in note or "cpu" in note.lower()

    def test_quant_algorithms_documented(self) -> None:
        algos = ACCURACY_DEBUGGER_NOTES["quant_algorithms"]
        for name in ("minmax", "maxdiff", "sqnr", "stats", "data_range_analyzer"):
            assert name in algos

    def test_supported_models(self) -> None:
        supported = ACCURACY_DEBUGGER_NOTES["supported_models"]
        for fw in ("ONNX", "TFLite", "TensorFlow"):
            assert fw in supported

    def test_exported_from_sdk_tools_package(self) -> None:
        from quad.sdk_tools import (  # noqa: F401
            ACCURACY_DEBUGGER_NOTES,
            AccDebuggerFramework,
            E2EAccuracyDebuggerArgs,
            FrameworkRunnerArgs,
            InferenceEngineArgs,
            VerificationArgs,
            VerifierType,
        )
