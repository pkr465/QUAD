"""Tests for model conversion utilities."""

from __future__ import annotations

import pytest

from quad.compiler.model_conversion import (
    ConversionConfig,
    InputSpec,
    LEGACY_CONVERTERS,
    SourceFramework,
    TF_LAYER_MAPPINGS,
    TensorFlowInputFormat,
    UNIFIED_CONVERTER,
    get_all_supported_tf_ops,
    get_tf_layer_mapping,
)


class TestInputSpec:
    def test_dim_string(self) -> None:
        spec = InputSpec(name="input", dimensions=(1, 299, 299, 3))
        assert spec.dim_string == "1,299,299,3"

    def test_cli_args(self) -> None:
        spec = InputSpec(name="Placeholder", dimensions=(1, 224, 224, 3))
        args = spec.cli_args
        assert args == ["--input_dim", "Placeholder", "1,224,224,3"]

    def test_single_dim(self) -> None:
        spec = InputSpec(name="input", dimensions=(1000,))
        assert spec.dim_string == "1000"


class TestConversionConfig:
    def test_default_uses_unified_converter(self) -> None:
        cfg = ConversionConfig(model_path="model.onnx")
        assert cfg.use_unified_converter is True
        assert cfg.converter_tool == UNIFIED_CONVERTER

    def test_legacy_converter_selection(self) -> None:
        cfg = ConversionConfig(
            model_path="model.pb",
            source_framework=SourceFramework.TENSORFLOW,
            use_unified_converter=False,
        )
        assert cfg.converter_tool == "snpe-tensorflow-to-dlc"

    def test_output_dlc_auto_generated(self) -> None:
        cfg = ConversionConfig(model_path="/path/to/inception_v3.onnx")
        assert cfg.output_dlc_path == "/path/to/inception_v3.dlc"

    def test_output_dlc_explicit(self) -> None:
        cfg = ConversionConfig(model_path="model.onnx", output_path="/out/custom.dlc")
        assert cfg.output_dlc_path == "/out/custom.dlc"

    def test_build_cli_args_unified(self) -> None:
        cfg = ConversionConfig(model_path="model.onnx")
        args = cfg.build_cli_args()
        assert "--input_network" in args
        assert "model.onnx" in args

    def test_build_cli_args_unified_fp16(self) -> None:
        cfg = ConversionConfig(model_path="model.onnx", float_bitwidth=16)
        args = cfg.build_cli_args()
        assert "--float_bitwidth" in args
        assert "16" in args

    def test_build_cli_args_tf_legacy(self) -> None:
        """TensorFlow legacy converter needs --input_dim and --out_node."""
        cfg = ConversionConfig(
            model_path="inception.pb",
            source_framework=SourceFramework.TENSORFLOW,
            use_unified_converter=False,
            input_specs=[InputSpec("input", (1, 299, 299, 3))],
            output_nodes=["InceptionV3/Predictions/Reshape_1"],
        )
        args = cfg.build_cli_args()
        assert "--input_dim" in args
        assert "input" in args
        assert "1,299,299,3" in args
        assert "--out_node" in args
        assert "InceptionV3/Predictions/Reshape_1" in args
        assert "--output_path" in args

    def test_build_cli_args_with_udo(self) -> None:
        cfg = ConversionConfig(
            model_path="model.pb",
            source_framework=SourceFramework.TENSORFLOW,
            udo_config_paths=["Softmax_Htp.json"],
        )
        args = cfg.build_cli_args()
        assert "--udo_config_paths" in args
        assert "Softmax_Htp.json" in args

    def test_build_cli_args_dry_run(self) -> None:
        cfg = ConversionConfig(model_path="model.onnx", dry_run=True)
        args = cfg.build_cli_args()
        assert "--dry_run" in args

    def test_validate_empty_path(self) -> None:
        cfg = ConversionConfig(model_path="")
        errors = cfg.validate()
        assert any("model_path" in e for e in errors)

    def test_validate_tf_requires_output_nodes(self) -> None:
        """TF legacy converter requires --out_node."""
        cfg = ConversionConfig(
            model_path="model.pb",
            source_framework=SourceFramework.TENSORFLOW,
            use_unified_converter=False,
            output_nodes=[],  # Missing!
        )
        errors = cfg.validate()
        assert any("out_node" in e.lower() for e in errors)

    def test_validate_tf_with_output_nodes_passes(self) -> None:
        cfg = ConversionConfig(
            model_path="model.pb",
            source_framework=SourceFramework.TENSORFLOW,
            use_unified_converter=False,
            output_nodes=["output"],
        )
        errors = cfg.validate()
        assert not any("out_node" in e.lower() for e in errors)

    def test_validate_extension_mismatch(self) -> None:
        cfg = ConversionConfig(
            model_path="model.onnx",
            source_framework=SourceFramework.TENSORFLOW,
        )
        errors = cfg.validate()
        assert any("extension" in e.lower() for e in errors)

    def test_validate_correct_extension_no_error(self) -> None:
        cfg = ConversionConfig(
            model_path="model.pb",
            source_framework=SourceFramework.TENSORFLOW,
        )
        errors = cfg.validate()
        assert not any("extension" in e.lower() for e in errors)


class TestTFLayerMappings:
    def test_conv2d_mapping(self) -> None:
        result = get_tf_layer_mapping("Conv2D")
        assert result is not None
        assert result["snpe_layer"] == "Conv2d"

    def test_matmul_to_fc(self) -> None:
        result = get_tf_layer_mapping("MatMul")
        assert result is not None
        assert result["snpe_layer"] == "FullyConnected"

    def test_softmax_mapping(self) -> None:
        result = get_tf_layer_mapping("Softmax")
        assert result is not None
        assert result["snpe_layer"] == "Softmax"

    def test_relu_mapping(self) -> None:
        result = get_tf_layer_mapping("Relu")
        assert result is not None
        assert "Relu" in result["snpe_layer"]

    def test_unknown_op_returns_none(self) -> None:
        assert get_tf_layer_mapping("UnknownCustomOp") is None

    def test_all_supported_ops_non_empty(self) -> None:
        ops = get_all_supported_tf_ops()
        assert len(ops) > 20  # At least 20 supported TF ops

    def test_concat_v2_supported(self) -> None:
        result = get_tf_layer_mapping("ConcatV2")
        assert result is not None

    def test_fused_batch_norm_supported(self) -> None:
        result = get_tf_layer_mapping("FusedBatchNorm")
        assert result is not None


class TestSourceFramework:
    def test_all_frameworks(self) -> None:
        assert SourceFramework.ONNX.value == "onnx"
        assert SourceFramework.TENSORFLOW.value == "tensorflow"
        assert SourceFramework.TFLITE.value == "tflite"
        assert SourceFramework.PYTORCH.value == "pytorch"

    def test_legacy_converter_for_each(self) -> None:
        for fw in SourceFramework:
            assert fw.value in LEGACY_CONVERTERS


class TestTFInputFormats:
    def test_all_formats(self) -> None:
        assert TensorFlowInputFormat.FROZEN_PB.value == "frozen_pb"
        assert TensorFlowInputFormat.CHECKPOINT.value == "checkpoint"
        assert TensorFlowInputFormat.SAVED_MODEL.value == "saved_model"


class TestONNXConversionNotes:
    def test_notes_exist(self) -> None:
        from quad.compiler.model_conversion import ONNX_CONVERSION_NOTES
        assert "symbolic_shapes" in ONNX_CONVERSION_NOTES
        assert "data_types_ignored" in ONNX_CONVERSION_NOTES
        assert "functions_inlined" in ONNX_CONVERSION_NOTES
        assert "windows_powershell" in ONNX_CONVERSION_NOTES

    def test_symbolic_shapes_warning_content(self) -> None:
        from quad.compiler.model_conversion import ONNX_CONVERSION_NOTES
        note = ONNX_CONVERSION_NOTES["symbolic_shapes"]
        assert "NOT supported" in note or "not supported" in note.lower()
        assert "Network Resizing" in note or "input_dimensions" in note

    def test_data_types_warning_content(self) -> None:
        from quad.compiler.model_conversion import ONNX_CONVERSION_NOTES
        note = ONNX_CONVERSION_NOTES["data_types_ignored"]
        assert "IGNORED" in note or "ignored" in note

    def test_get_onnx_warnings_nonexistent_file(self) -> None:
        from quad.compiler.model_conversion import get_onnx_conversion_warnings
        # Non-existent file — returns general guidance or empty
        warnings = get_onnx_conversion_warnings("/nonexistent/model.onnx")
        # Should not raise, may return warnings or empty list
        assert isinstance(warnings, list)

    def test_onnx_conversion_config(self) -> None:
        """ONNX conversion is simplest — just input_network + output_path."""
        cfg = ConversionConfig(
            model_path="alexnet.onnx",
            source_framework=SourceFramework.ONNX,
            use_unified_converter=False,
        )
        args = cfg.build_cli_args()
        assert "--input_network" in args
        assert "alexnet.onnx" in args
        assert "--output_path" in args

    def test_onnx_unified_converter_minimal(self) -> None:
        """Unified converter for ONNX: just --input_network."""
        cfg = ConversionConfig(model_path="model.onnx")
        args = cfg.build_cli_args()
        assert args == ["--input_network", "model.onnx"]

    def test_onnx_validation_passes(self) -> None:
        cfg = ConversionConfig(model_path="model.onnx", source_framework=SourceFramework.ONNX)
        assert cfg.validate() == []


class TestTFLiteConversion:
    def test_tflite_notes_exist(self) -> None:
        from quad.compiler.model_conversion import TFLITE_CONVERSION_NOTES
        assert "float_inputs_only" in TFLITE_CONVERSION_NOTES
        assert "mlir_issues" in TFLITE_CONVERSION_NOTES

    def test_tflite_float_only_warning(self) -> None:
        from quad.compiler.model_conversion import TFLITE_CONVERSION_NOTES
        assert "FLOAT" in TFLITE_CONVERSION_NOTES["float_inputs_only"].upper()

    def test_tflite_legacy_converter(self) -> None:
        cfg = ConversionConfig(
            model_path="inception_v3.tflite",
            source_framework=SourceFramework.TFLITE,
            use_unified_converter=False,
        )
        assert cfg.converter_tool == "snpe-tflite-to-dlc"

    def test_tflite_with_input_dim(self) -> None:
        cfg = ConversionConfig(
            model_path="model.tflite",
            source_framework=SourceFramework.TFLITE,
            use_unified_converter=False,
            input_specs=[InputSpec("input", (1, 299, 299, 3))],
        )
        args = cfg.build_cli_args()
        assert "--input_network" in args
        assert "model.tflite" in args
        assert "--output_path" in args

    def test_tflite_validation_passes(self) -> None:
        cfg = ConversionConfig(
            model_path="model.tflite",
            source_framework=SourceFramework.TFLITE,
        )
        errors = cfg.validate()
        assert not any("extension" in e.lower() for e in errors)


class TestPyTorchConversion:
    def test_pytorch_notes_exist(self) -> None:
        from quad.compiler.model_conversion import PYTORCH_CONVERSION_NOTES
        assert "torchscript_deprecated" in PYTORCH_CONVERSION_NOTES
        assert "float_inputs_only" in PYTORCH_CONVERSION_NOTES
        assert "nchw_layout" in PYTORCH_CONVERSION_NOTES

    def test_pytorch_legacy_converter(self) -> None:
        cfg = ConversionConfig(
            model_path="resnet18.pt",
            source_framework=SourceFramework.PYTORCH,
            use_unified_converter=False,
        )
        assert cfg.converter_tool == "snpe-pytorch-to-dlc"

    def test_pytorch_nchw_input_dim(self) -> None:
        """PyTorch uses NCHW (not NHWC like TensorFlow)."""
        spec = InputSpec("input", (1, 3, 224, 224))
        assert spec.dim_string == "1,3,224,224"  # NCHW order preserved

    def test_generate_torchscript_code_legacy(self) -> None:
        from quad.compiler.model_conversion import generate_torchscript_export_code
        code = generate_torchscript_export_code(
            model_import="torchvision.models.resnet18()",
            input_shape=(1, 3, 224, 224),
            output_path="resnet18.pt",
        )
        assert "torch.jit.trace" in code
        assert "resnet18.pt" in code
        assert "model.eval()" in code

    def test_generate_onnx_export_code_recommended(self) -> None:
        """ONNX export is the RECOMMENDED path (TorchScript is deprecated)."""
        from quad.compiler.model_conversion import generate_onnx_export_code
        code = generate_onnx_export_code(
            model_import="torchvision.models.resnet18(pretrained=True)",
            input_shape=(1, 3, 224, 224),
            output_path="resnet18.onnx",
        )
        assert "torch.onnx.export" in code
        assert "resnet18.onnx" in code
        assert "opset_version" in code
        assert "dynamic_axes" in code

    def test_pytorch_notes_mention_deprecation(self) -> None:
        from quad.compiler.model_conversion import PYTORCH_CONVERSION_NOTES
        assert "deprecated" in PYTORCH_CONVERSION_NOTES["torchscript_deprecated"].lower()
        assert "torch.export" in PYTORCH_CONVERSION_NOTES["torchscript_deprecated"]

    def test_pytorch_validation_passes(self) -> None:
        cfg = ConversionConfig(
            model_path="model.pt",
            source_framework=SourceFramework.PYTORCH,
        )
        errors = cfg.validate()
        assert not any("extension" in e.lower() for e in errors)


class TestQuantizationNotes:
    def test_notes_exist(self) -> None:
        from quad.compiler.model_conversion import QUANTIZATION_NOTES
        assert "batch_must_be_1" in QUANTIZATION_NOTES
        assert "input_data_quick" in QUANTIZATION_NOTES
        assert "input_data_robust" in QUANTIZATION_NOTES
        assert "op_guarantees" in QUANTIZATION_NOTES

    def test_calibration_sizes(self) -> None:
        from quad.compiler.model_conversion import (
            QUANTIZATION_CALIBRATION_MIN,
            QUANTIZATION_CALIBRATION_RECOMMENDED,
            QUANTIZATION_CALIBRATION_ROBUST,
        )
        assert QUANTIZATION_CALIBRATION_MIN == 5
        assert QUANTIZATION_CALIBRATION_RECOMMENDED == 50
        assert QUANTIZATION_CALIBRATION_ROBUST == 100
        assert QUANTIZATION_CALIBRATION_MIN < QUANTIZATION_CALIBRATION_RECOMMENDED < QUANTIZATION_CALIBRATION_ROBUST

    def test_batch_must_be_1_note(self) -> None:
        from quad.compiler.model_conversion import QUANTIZATION_NOTES
        assert "1" in QUANTIZATION_NOTES["batch_must_be_1"]

    def test_build_quantize_args_unified(self) -> None:
        from quad.compiler.model_conversion import build_quantize_cli_args
        args = build_quantize_cli_args("model.dlc", "inputs.txt", use_unified=True)
        assert "--input_dlc" in args
        assert "model.dlc" in args
        assert "--input_list" in args
        assert "--weights_bitwidth" in args
        assert "--act_bitwidth" in args

    def test_build_quantize_args_snpe_dlc_quantize(self) -> None:
        from quad.compiler.model_conversion import build_quantize_cli_args
        args = build_quantize_cli_args("model.dlc", "inputs.txt", use_unified=False)
        assert "--input_dlc" in args
        assert "--weights_bitwidth" in args

    def test_build_quantize_with_htp(self) -> None:
        from quad.compiler.model_conversion import build_quantize_cli_args
        args = build_quantize_cli_args(
            "model.dlc", "inputs.txt", htp_socs="sm8650", use_unified=False
        )
        assert "--enable_htp" in args
        assert "--htp_socs" in args
        assert "sm8650" in args

    def test_build_quantize_per_channel(self) -> None:
        from quad.compiler.model_conversion import build_quantize_cli_args
        args = build_quantize_cli_args(
            "model.dlc", "inputs.txt", use_per_channel=True
        )
        assert "--use_per_channel_quantization" in args

    def test_build_quantize_with_cle(self) -> None:
        from quad.compiler.model_conversion import build_quantize_cli_args
        args = build_quantize_cli_args(
            "model.dlc", "inputs.txt", algorithms=["cle"], use_unified=True
        )
        assert "--apply_algorithms" in args
        assert "cle" in args


class TestOfflineGraphCaching:
    def test_notes_exist(self) -> None:
        from quad.compiler.model_conversion import OFFLINE_CACHE_NOTES
        assert "workflow" in OFFLINE_CACHE_NOTES
        assert "output_matching" in OFFLINE_CACHE_NOTES
        assert "soc_compatibility" in OFFLINE_CACHE_NOTES
        assert "input_dimensions" in OFFLINE_CACHE_NOTES
        assert "optimization_level" in OFFLINE_CACHE_NOTES

    def test_graph_prepare_preferred(self) -> None:
        from quad.compiler.model_conversion import OFFLINE_CACHE_NOTES
        assert "DEPRECATED" in OFFLINE_CACHE_NOTES["graph_prepare_preferred"]

    def test_cache_compat_same_arch(self) -> None:
        from quad.compiler.model_conversion import is_cache_compatible
        assert is_cache_compatible("v73", "v73") is True
        assert is_cache_compatible("v75", "v75") is True

    def test_cache_compat_v68_v73_incompatible(self) -> None:
        """v68/v69 cache will NOT run on v73 device (documented explicitly)."""
        from quad.compiler.model_conversion import is_cache_compatible
        assert is_cache_compatible("v68", "v73") is False
        assert is_cache_compatible("v69", "v73") is False

    def test_cache_compat_newer_on_older_incompatible(self) -> None:
        """Newer arch cache cannot run on older arch device."""
        from quad.compiler.model_conversion import is_cache_compatible
        assert is_cache_compatible("v73", "v68") is False
        assert is_cache_compatible("v75", "v73") is False

    def test_cache_compat_older_on_newer_may_work(self) -> None:
        """Older arch cache MAY run on newer device (same features, smaller VTCM)."""
        from quad.compiler.model_conversion import is_cache_compatible
        # v73 on v75 — may be compatible
        result = is_cache_compatible("v73", "v75")
        assert result is True or result is None  # Compatible or unknown

    def test_build_graph_prepare_basic(self) -> None:
        from quad.compiler.model_conversion import build_graph_prepare_cli_args
        args = build_graph_prepare_cli_args("model_q.dlc", htp_socs="sm8750")
        assert "--input_dlc=model_q.dlc" in args
        assert "--htp_socs=sm8750" in args
        assert "--optimization_level=2" in args

    def test_build_graph_prepare_level3(self) -> None:
        from quad.compiler.model_conversion import build_graph_prepare_cli_args
        args = build_graph_prepare_cli_args("m.dlc", optimization_level=3)
        assert "--optimization_level=3" in args

    def test_build_graph_prepare_output_tensors(self) -> None:
        from quad.compiler.model_conversion import build_graph_prepare_cli_args
        args = build_graph_prepare_cli_args(
            "m.dlc",
            set_output_tensors=["output:0", "output:1"],
        )
        assert "--set_output_tensors=output:0" in args
        assert "--set_output_tensors=output:1" in args

    def test_build_graph_prepare_resized_input(self) -> None:
        from quad.compiler.model_conversion import build_graph_prepare_cli_args
        args = build_graph_prepare_cli_args(
            "m.dlc",
            input_name="input",
            input_dimensions="2,224,224,3",
        )
        assert "--input_name=input" in args
        assert "--input_dimensions=2,224,224,3" in args

    def test_build_graph_prepare_multiple_socs(self) -> None:
        from quad.compiler.model_conversion import build_graph_prepare_cli_args
        args = build_graph_prepare_cli_args("m.dlc", htp_socs="sm8350,sm8450,sm8550")
        assert "sm8350,sm8450,sm8550" in " ".join(args)


class TestQAIRTConverter:
    def test_notes_exist(self) -> None:
        from quad.compiler.model_conversion import QAIRT_CONVERTER_NOTES
        assert "layout_preserved" in QAIRT_CONVERTER_NOTES
        assert "renamed_flags" in QAIRT_CONVERTER_NOTES
        assert "tf_required_args" in QAIRT_CONVERTER_NOTES

    def test_renamed_flags(self) -> None:
        from quad.compiler.model_conversion import get_legacy_to_qairt_flag_mapping
        m = get_legacy_to_qairt_flag_mapping()
        assert m["--input_dim"] == "--source_model_input_shape"
        assert m["--out_node"] == "--out_tensor_node"
        assert m["--input_encoding"] == "--input_color_encoding"
        assert m["--define_symbol"] == "--onnx_define_symbol"

    def test_qairt_config_minimal_onnx(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(model_path="model.onnx")
        args = cfg.build_cli_args()
        assert "--input_network" in args
        assert "model.onnx" in args
        assert cfg.converter_tool == "qairt-converter"

    def test_qairt_config_tf_requires_shape_and_node(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(model_path="model.pb")
        errors = cfg.validate()
        assert any("source_model_input_shape" in e for e in errors)
        assert any("out_tensor_node" in e for e in errors)

    def test_qairt_config_tf_with_args_passes(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(
            model_path="inception.pb",
            input_shapes=[InputSpec("input", (1, 299, 299, 3))],
            output_tensor_nodes=["InceptionV3/Predictions/Reshape_1"],
        )
        errors = cfg.validate()
        assert errors == []
        args = cfg.build_cli_args()
        assert "--source_model_input_shape" in args
        assert "input" in args
        assert "1,299,299,3" in args
        assert "--out_tensor_node" in args

    def test_qairt_config_float16(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(
            model_path="model.onnx",
            float_bitwidth=16,
            float_bias_bitwidth=32,
        )
        args = cfg.build_cli_args()
        assert "--float_bitwidth" in args
        assert "16" in args
        assert "--float_bias_bitwidth" in args
        assert "32" in args

    def test_qairt_config_strip_quant(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig, ExportFormat
        cfg = QAIRTConversionConfig(
            model_path="quant_model.onnx",
            export_format=ExportFormat.DLC_STRIP_QUANT,
        )
        args = cfg.build_cli_args()
        assert "--export_format" in args
        assert "DLC_STRIP_QUANT" in args

    def test_qairt_config_dry_run(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(model_path="model.onnx", dry_run=True)
        args = cfg.build_cli_args()
        assert "--dry_run" in args

    def test_qairt_config_io_yaml(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(
            model_path="model.onnx",
            io_config_yaml="io_config.yaml",
        )
        args = cfg.build_cli_args()
        assert "--config" in args
        assert "io_config.yaml" in args

    def test_qairt_config_dump_template(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(
            model_path="model.onnx",
            dump_config_template="./output/io_config.yaml",
        )
        args = cfg.build_cli_args()
        assert "--dump_config_template" in args

    def test_qairt_config_onnx_define_symbol(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(
            model_path="model.onnx",
            onnx_define_symbols={"height": 224, "width": 448},
        )
        args = cfg.build_cli_args()
        assert "--onnx_define_symbol" in args
        assert "height" in args
        assert "224" in args

    def test_qairt_config_remove_unused_inputs(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(
            model_path="model.onnx",
            remove_unused_inputs=True,
        )
        args = cfg.build_cli_args()
        assert "--remove_unused_inputs" in args

    def test_qairt_config_quantization_overrides(self) -> None:
        from quad.compiler.model_conversion import QAIRTConversionConfig
        cfg = QAIRTConversionConfig(
            model_path="model.onnx",
            quantization_overrides="overrides.json",
        )
        args = cfg.build_cli_args()
        assert "--quantization_overrides" in args
        assert "overrides.json" in args


class TestQAIRTQuantizer:
    def test_notes_exist(self) -> None:
        from quad.compiler.model_conversion import QAIRT_QUANTIZER_NOTES
        assert "fills_gaps" in QAIRT_QUANTIZER_NOTES
        assert "noop_flags" in QAIRT_QUANTIZER_NOTES
        assert "mutually_exclusive" in QAIRT_QUANTIZER_NOTES
        assert "aimet_setup" in QAIRT_QUANTIZER_NOTES

    def test_noop_flags_documented(self) -> None:
        from quad.compiler.model_conversion import QAIRT_QUANTIZER_NOTES
        note = QAIRT_QUANTIZER_NOTES["noop_flags"]
        assert "no-op" in note.lower() or "NO-OPS" in note
        assert "ignore_quantization_overrides" in note
        assert "enable_float_fallback" in note

    def test_build_quantizer_basic(self) -> None:
        from quad.compiler.model_conversion import build_qairt_quantizer_args
        args = build_qairt_quantizer_args("model.dlc", input_list="calib.txt")
        assert "--input_dlc" in args
        assert "model.dlc" in args
        assert "--input_list" in args
        assert "--weights_bitwidth" in args
        assert "--act_bitwidth" in args

    def test_build_quantizer_with_aimet(self) -> None:
        from quad.compiler.model_conversion import build_qairt_quantizer_args
        args = build_qairt_quantizer_args(
            "model.dlc",
            input_list="calib.txt",
            use_aimet=True,
            algorithms=["adaround"],
            aimet_config="adaround.yaml",
        )
        assert "--use_aimet_quantizer" in args
        assert "--apply_algorithms" in args
        assert "adaround" in args
        assert "--config" in args

    def test_build_quantizer_ignore_overrides(self) -> None:
        from quad.compiler.model_conversion import build_qairt_quantizer_args
        args = build_qairt_quantizer_args(
            "model.dlc",
            input_list="calib.txt",
            ignore_quantization_overrides=True,
        )
        assert "--ignore_quantization_overrides" in args

    def test_generate_amp_yaml(self) -> None:
        from quad.compiler.model_conversion import generate_aimet_amp_yaml
        yaml = generate_aimet_amp_yaml()
        assert "aimet_quantizer" in yaml
        assert "amp" in yaml
        assert "allowed_accuracy_drop" in yaml
        assert "candidates" in yaml

    def test_generate_adaround_yaml(self) -> None:
        from quad.compiler.model_conversion import generate_aimet_adaround_yaml
        yaml = generate_aimet_adaround_yaml(num_batches=5)
        assert "aimet_quantizer" in yaml
        assert "adaround" in yaml
        assert "num_batches: 5" in yaml

    def test_aimet_min_version_documented(self) -> None:
        from quad.compiler.model_conversion import QAIRT_QUANTIZER_NOTES
        assert "1.33.0" in QAIRT_QUANTIZER_NOTES["aimet_setup"]


class TestModelTipsMobilenetSSD:
    def test_tips_dict_exists(self) -> None:
        from quad.compiler.model_conversion import MODEL_TIPS
        assert "mobilenet_ssd" in MODEL_TIPS
        tip = MODEL_TIPS["mobilenet_ssd"]
        assert tip["input_shape"] == (1, 300, 300, 3)
        assert "detection_classes" in tip["output_nodes"]
        assert "detection_boxes" in tip["output_nodes"]
        assert "detection_scores" in tip["output_nodes"]

    def test_limitations_documented(self) -> None:
        from quad.compiler.model_conversion import MODEL_TIPS
        limits = MODEL_TIPS["mobilenet_ssd"]["limitations"]
        assert any("Batch" in l for l in limits)
        assert any("CPU" in l for l in limits)
        assert any("top_k" in l for l in limits)
        assert any("resizing" in l.lower() or "PriorBox" in l for l in limits)

    def test_output_buffers(self) -> None:
        from quad.compiler.model_conversion import MODEL_TIPS
        bufs = MODEL_TIPS["mobilenet_ssd"]["output_buffers"]
        assert "classes_offset" in bufs
        assert "boxes" in bufs
        assert "scores" in bufs

    def test_build_legacy_conversion_args(self) -> None:
        from quad.compiler.model_conversion import build_mobilenet_ssd_conversion_args
        args = build_mobilenet_ssd_conversion_args(
            "frozen_inference_graph.pb",
            use_qairt=False,
        )
        assert "--input_network" in args
        assert "--input_dim" in args
        assert "Preprocessor/sub" in args
        assert "1,300,300,3" in args
        assert "--out_node" in args
        assert "--allow_unconsumed_nodes" in args
        assert args.count("--out_node") == 3  # Three output nodes

    def test_build_qairt_conversion_args(self) -> None:
        from quad.compiler.model_conversion import build_mobilenet_ssd_conversion_args
        args = build_mobilenet_ssd_conversion_args(
            "frozen_inference_graph.pb",
            use_qairt=True,
        )
        assert "--source_model_input_shape" in args
        assert "--out_tensor_node" in args
        assert "--allow_unconsumed_nodes" not in args  # qairt doesn't need this flag

    def test_generate_export_script(self) -> None:
        from quad.compiler.model_conversion import generate_mobilenet_ssd_export_script
        script = generate_mobilenet_ssd_export_script()
        assert "export_inference_graph.py" in script
        assert "image_tensor" in script
        assert "frozen_inference_graph.pb" in script


class TestModelTipsDeepLabv3:
    def test_tip_exists(self) -> None:
        from quad.compiler.model_conversion import MODEL_TIPS
        assert "deeplabv3" in MODEL_TIPS
        tip = MODEL_TIPS["deeplabv3"]
        assert tip["input_shape"] == (1, 513, 513, 3)
        assert "ArgMax" in tip["output_nodes"]

    def test_preprocessing_has_six_steps(self) -> None:
        from quad.compiler.model_conversion import MODEL_TIPS
        steps = MODEL_TIPS["deeplabv3"]["preprocessing"]
        assert len(steps) == 6

    def test_preprocessing_steps_order(self) -> None:
        from quad.compiler.model_conversion import MODEL_TIPS
        steps = MODEL_TIPS["deeplabv3"]["preprocessing"]
        assert "513.0" in steps[0]           # Step 1: resize ratio
        assert "resize" in steps[1].lower()   # Step 2: resize
        assert "pad" in steps[2].lower()      # Step 3: pad with 128
        assert "float32" in steps[3]          # Step 4: convert
        assert "0.00784" in steps[4]          # Step 5: multiply
        assert "1.0" in steps[5]              # Step 6: subtract 1.0

    def test_postprocessing_has_two_steps(self) -> None:
        from quad.compiler.model_conversion import MODEL_TIPS
        steps = MODEL_TIPS["deeplabv3"]["postprocessing"]
        assert len(steps) == 2

    def test_preprocessing_no_snpe_support(self) -> None:
        """SNPE does not support DeepLabv3 preprocessing internally."""
        from quad.compiler.model_conversion import MODEL_TIPS
        limits = MODEL_TIPS["deeplabv3"]["limitations"]
        assert any("preprocessing" in l.lower() for l in limits)

    def test_build_legacy_args(self) -> None:
        from quad.compiler.model_conversion import build_deeplabv3_conversion_args
        args = build_deeplabv3_conversion_args("frozen_inference_graph.pb")
        assert "--input_network" in args
        assert "--input_dim" in args
        assert "sub_7" in args
        assert "1,513,513,3" in args
        assert "--out_node" in args
        assert "ArgMax" in args

    def test_build_qairt_args(self) -> None:
        from quad.compiler.model_conversion import build_deeplabv3_conversion_args
        args = build_deeplabv3_conversion_args("frozen_inference_graph.pb", use_qairt=True)
        assert "--source_model_input_shape" in args
        assert "--out_tensor_node" in args

    def test_generate_preprocess_code(self) -> None:
        from quad.compiler.model_conversion import generate_deeplabv3_preprocess_code
        code = generate_deeplabv3_preprocess_code()
        assert "513.0 / max" in code
        assert "0.00784313771874" in code
        assert "-= 1.0" in code
        assert "LANCZOS" in code
        assert "constant_values=128" in code

    def test_normalization_constant(self) -> None:
        """Verify the normalization constant maps [0,255] → [-1,1]."""
        # 0 * 0.00784313771874 - 1.0 = -1.0
        # 255 * 0.00784313771874 - 1.0 ≈ 1.0
        k = 0.00784313771874
        assert abs(0 * k - 1.0 - (-1.0)) < 1e-6
        assert abs(255 * k - 1.0 - 1.0) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Image Format Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestImageFormatNotes:
    def test_notes_keys_present(self) -> None:
        from quad.compiler.model_conversion import IMAGE_FORMAT_NOTES
        for key in ("snpe_layout", "pytorch_layout", "channel_order",
                    "batch_handling", "mnist_example", "alexnet_example"):
            assert key in IMAGE_FORMAT_NOTES

    def test_snpe_layout_format(self) -> None:
        from quad.compiler.model_conversion import IMAGE_FORMAT_NOTES
        layout = IMAGE_FORMAT_NOTES["snpe_layout"]
        assert "NHWC" in layout["name"]
        assert "channel" in layout["format"].lower()

    def test_pytorch_layout_conversion_hint(self) -> None:
        from quad.compiler.model_conversion import IMAGE_FORMAT_NOTES
        layout = IMAGE_FORMAT_NOTES["pytorch_layout"]
        assert "transpose" in layout["conversion"].lower()
        assert "NCHW" in layout["conversion"]
        assert "NHWC" in layout["conversion"]

    def test_channel_order_bgr_models(self) -> None:
        from quad.compiler.model_conversion import IMAGE_FORMAT_NOTES
        channel = IMAGE_FORMAT_NOTES["channel_order"]
        bgr_models = channel["bgr_models"]
        assert any("alexnet" in m.lower() or "googlenet" in m.lower() for m in bgr_models)

    def test_batch_handling_concat_hint(self) -> None:
        from quad.compiler.model_conversion import IMAGE_FORMAT_NOTES
        batch = IMAGE_FORMAT_NOTES["batch_handling"]
        assert "cat" in batch["concatenation"]
        assert "concatenate" in batch["python_concat"]

    def test_mnist_shape_description(self) -> None:
        from quad.compiler.model_conversion import IMAGE_FORMAT_NOTES
        mnist = IMAGE_FORMAT_NOTES["mnist_example"]
        assert "28" in mnist["pytorch_shape"]
        assert "28" in mnist["snpe_shape"]

    def test_generate_image_format_notes(self) -> None:
        from quad.compiler.model_conversion import generate_image_format_notes
        notes = generate_image_format_notes()
        assert "NHWC" in notes
        assert "NCHW" in notes
        assert "BGR" in notes
        assert "batch" in notes.lower()
        assert "MNIST" in notes


class TestConvertNchwToNhwc:
    def test_4d_batch(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_nchw_to_nhwc
        img = np.zeros((2, 3, 224, 224), dtype=np.float32)
        out = convert_nchw_to_nhwc(img)
        assert out.shape == (2, 224, 224, 3)

    def test_3d_single_image(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_nchw_to_nhwc
        img = np.zeros((3, 28, 28), dtype=np.float32)
        out = convert_nchw_to_nhwc(img)
        assert out.shape == (28, 28, 3)

    def test_mnist_single_channel(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_nchw_to_nhwc
        # MNIST: (1, 1, 28, 28) → (1, 28, 28, 1)
        img = np.zeros((1, 1, 28, 28), dtype=np.float32)
        out = convert_nchw_to_nhwc(img)
        assert out.shape == (1, 28, 28, 1)

    def test_alexnet_shape(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_nchw_to_nhwc
        # AlexNet: (1, 3, 227, 227) → (1, 227, 227, 3)
        img = np.zeros((1, 3, 227, 227), dtype=np.float32)
        out = convert_nchw_to_nhwc(img)
        assert out.shape == (1, 227, 227, 3)

    def test_values_transposed_correctly(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_nchw_to_nhwc
        # Create a (1,3,2,2) image where each channel has a distinct value
        img = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]],
                       dtype=np.float32)  # shape (1,3,2,2)
        out = convert_nchw_to_nhwc(img)
        # At position (0, 0, 0, :) we expect channel values [1, 5, 9]
        assert list(out[0, 0, 0, :]) == [1.0, 5.0, 9.0]

    def test_invalid_shape_raises(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_nchw_to_nhwc
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            convert_nchw_to_nhwc(np.zeros((224, 224), dtype=np.float32))


class TestConvertChannelOrder:
    def test_rgb_to_bgr(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_channel_order
        img = np.array([[[[1.0, 2.0, 3.0]]]])  # (1, 1, 1, 3) R=1, G=2, B=3
        out = convert_channel_order(img, "rgb", "bgr")
        assert list(out[0, 0, 0, :]) == [3.0, 2.0, 1.0]  # B=3, G=2, R=1

    def test_bgr_to_rgb(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_channel_order
        img = np.array([[[[3.0, 2.0, 1.0]]]])  # (1, 1, 1, 3) B=3, G=2, R=1
        out = convert_channel_order(img, "bgr", "rgb")
        assert list(out[0, 0, 0, :]) == [1.0, 2.0, 3.0]  # R=1, G=2, B=3

    def test_same_order_noop(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_channel_order
        img = np.zeros((1, 4, 4, 3), dtype=np.float32)
        out = convert_channel_order(img, "rgb", "rgb")
        assert out is img or (out == img).all()

    def test_case_insensitive(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_channel_order
        img = np.array([[[[1.0, 2.0, 3.0]]]])
        out = convert_channel_order(img, "RGB", "BGR")
        assert list(out[0, 0, 0, :]) == [3.0, 2.0, 1.0]

    def test_unsupported_order_raises(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_channel_order
        img = np.zeros((1, 4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Supported orders"):
            convert_channel_order(img, "rgba", "bgr")

    def test_hwc_3d_input(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import convert_channel_order
        img = np.array([[[100.0, 150.0, 200.0]]])  # HWC (1,1,3)
        out = convert_channel_order(img, "rgb", "bgr")
        assert list(out[0, 0, :]) == [200.0, 150.0, 100.0]


class TestPrepareBatchInput:
    def test_single_image(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import prepare_batch_input
        img = np.ones((1, 4, 4, 3), dtype=np.float32)
        out_path = str(tmp_path / "batch.raw")
        n_bytes = prepare_batch_input([img], out_path)
        assert n_bytes == 4 * 4 * 3 * 4  # H*W*C * sizeof(float32)
        loaded = np.fromfile(out_path, dtype=np.float32).reshape(1, 4, 4, 3)
        assert loaded.shape == (1, 4, 4, 3)
        assert (loaded == 1.0).all()

    def test_batch_concatenation(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import prepare_batch_input
        imgs = [np.full((1, 4, 4, 3), float(i), dtype=np.float32) for i in range(3)]
        out_path = str(tmp_path / "batch3.raw")
        prepare_batch_input(imgs, out_path)
        loaded = np.fromfile(out_path, dtype=np.float32).reshape(3, 4, 4, 3)
        assert loaded.shape == (3, 4, 4, 3)
        assert loaded[0, 0, 0, 0] == 0.0
        assert loaded[1, 0, 0, 0] == 1.0
        assert loaded[2, 0, 0, 0] == 2.0

    def test_channel_order_conversion(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import prepare_batch_input
        # Image with distinct R/G/B values
        img = np.zeros((1, 2, 2, 3), dtype=np.float32)
        img[..., 0] = 1.0  # R channel
        img[..., 1] = 2.0  # G channel
        img[..., 2] = 3.0  # B channel
        out_path = str(tmp_path / "bgr.raw")
        prepare_batch_input([img], out_path, channel_order="rgb", target_channel_order="bgr")
        loaded = np.fromfile(out_path, dtype=np.float32).reshape(1, 2, 2, 3)
        # After RGB→BGR: index 0=B=3, index 1=G=2, index 2=R=1
        assert loaded[0, 0, 0, 0] == 3.0
        assert loaded[0, 0, 0, 2] == 1.0

    def test_3d_images_auto_expand(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]
        pytest.importorskip("numpy")
        import numpy as np
        from quad.compiler.model_conversion import prepare_batch_input
        # 3D images (HWC) should auto-expand to 4D
        imgs = [np.ones((4, 4, 3), dtype=np.float32) for _ in range(2)]
        out_path = str(tmp_path / "batch_3d.raw")
        prepare_batch_input(imgs, out_path)
        loaded = np.fromfile(out_path, dtype=np.float32).reshape(2, 4, 4, 3)
        assert loaded.shape == (2, 4, 4, 3)

