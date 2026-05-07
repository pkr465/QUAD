"""Tests for SNPE/QAIRT SDK Tools reference module."""

from __future__ import annotations

import pytest

from quad.sdk_tools.platform_matrix import (
    PLATFORM_NOTES,
    SDK_TOOLS,
    ToolCategory,
    ToolPlatform,
    get_tools_by_category,
    get_tools_for_platform,
    is_tool_available,
)
from quad.sdk_tools.tool_specs import (
    CacheCompatibilityMode,
    ConverterInputSpec,
    DIAGVIEW_NOTES,
    DIAGVIEW_TIMING_LAYERS,
    DiagviewArgs,
    DlcDiffArgs,
    DlcInfoArgs,
    ExportFormat,
    FloatBitwidth,
    GpuMode,
    InputEncoding,
    InputLayout,
    InputType,
    OnnxConverterArgs,
    PerfProfile,
    PriorityHint,
    ProfilingLevelNet,
    QairtConverterArgs,
    QairtQuantizerArgs,
    QuantBitwidth,
    QuantCalibration,
    QuantSchema,
    RuntimeOrder,
    SnpeNetRunArgs,
    TargetBackend,
)


# ══════════════════════════════════════════════════════════════════════════════
# Platform Matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestPlatformMatrix:
    def test_all_tool_categories_present(self) -> None:
        cats = {t.category for t in SDK_TOOLS.values()}
        assert ToolCategory.MODEL_CONVERSION in cats
        assert ToolCategory.MODEL_PREPARATION in cats
        assert ToolCategory.EXECUTION in cats
        assert ToolCategory.ANALYSIS in cats

    def test_qairt_converter_not_on_android(self) -> None:
        # qairt-converter is available on Linux host but model prep is done on host
        # Android device listed as android_device — converter runs on host
        td = SDK_TOOLS["qairt-converter"]
        assert ToolPlatform.UBUNTU in td.available_on

    def test_snpe_net_run_on_all_platforms(self) -> None:
        td = SDK_TOOLS["snpe-net-run"]
        for p in ToolPlatform:
            assert p in td.available_on, f"snpe-net-run should be available on {p}"

    def test_snpe_pytorch_linux_only(self) -> None:
        td = SDK_TOOLS["snpe-pytorch-to-dlc"]
        assert ToolPlatform.WINDOWS_X86_64 not in td.available_on
        assert ToolPlatform.WINDOWS_SNAPDRAGON not in td.available_on

    def test_snpe_tflite_linux_only(self) -> None:
        td = SDK_TOOLS["snpe-tflite-to-dlc"]
        assert ToolPlatform.WINDOWS_X86_64 not in td.available_on

    def test_qairt_converter_tflite_windows_note(self) -> None:
        td = SDK_TOOLS["qairt-converter"]
        assert any("TFLite" in n or "tflite" in n.lower() for n in td.notes)

    def test_snpe_diagview_not_on_windows_native(self) -> None:
        td = SDK_TOOLS["snpe-diagview"]
        assert ToolPlatform.WINDOWS_X86_64 not in td.available_on
        assert ToolPlatform.WINDOWS_SNAPDRAGON not in td.available_on

    def test_snpe_throughput_net_run_all_platforms(self) -> None:
        td = SDK_TOOLS["snpe-throughput-net-run"]
        assert ToolPlatform.WINDOWS_SNAPDRAGON in td.available_on

    def test_get_tools_for_platform_ubuntu(self) -> None:
        tools = get_tools_for_platform(ToolPlatform.UBUNTU)
        names = [t.name for t in tools]
        assert "snpe-net-run" in names
        assert "qairt-converter" in names
        assert "snpe_bench.py" in names

    def test_get_tools_for_platform_windows_snapdragon(self) -> None:
        tools = get_tools_for_platform(ToolPlatform.WINDOWS_SNAPDRAGON)
        names = [t.name for t in tools]
        assert "snpe-net-run" in names
        assert "snpe-throughput-net-run" in names

    def test_get_tools_by_category_conversion(self) -> None:
        tools = get_tools_by_category(ToolCategory.MODEL_CONVERSION)
        names = [t.name for t in tools]
        assert "snpe-onnx-to-dlc" in names
        assert "qairt-converter" in names
        assert "snpe-pytorch-to-dlc" in names

    def test_get_tools_by_category_execution(self) -> None:
        tools = get_tools_by_category(ToolCategory.EXECUTION)
        names = [t.name for t in tools]
        assert "snpe-net-run" in names
        assert "snpe-parallel-run" in names
        assert "snpe-throughput-net-run" in names

    def test_is_tool_available_positive(self) -> None:
        assert is_tool_available("snpe-net-run", ToolPlatform.UBUNTU) is True
        assert is_tool_available("qairt-converter", ToolPlatform.UBUNTU) is True

    def test_is_tool_available_negative(self) -> None:
        assert is_tool_available("snpe-pytorch-to-dlc", ToolPlatform.WINDOWS_X86_64) is False
        assert is_tool_available("nonexistent-tool", ToolPlatform.UBUNTU) is False

    def test_platform_notes_keys(self) -> None:
        for key in ("wsl_x86_64", "windows_arm64x", "windows_star", "powershell", "tflite_windows"):
            assert key in PLATFORM_NOTES

    def test_powershell_note_mentions_venv(self) -> None:
        assert "venv" in PLATFORM_NOTES["powershell"].lower()

    def test_tflite_windows_note_mentions_tvm(self) -> None:
        assert "TVM" in PLATFORM_NOTES["tflite_windows"]


# ══════════════════════════════════════════════════════════════════════════════
# Input Encoding and Layout Enums
# ══════════════════════════════════════════════════════════════════════════════

class TestInputEnums:
    def test_input_encoding_values(self) -> None:
        assert InputEncoding.BGR.value == "bgr"
        assert InputEncoding.RGB.value == "rgb"
        assert InputEncoding.NV21.value == "nv21"
        assert InputEncoding.TIME_SERIES.value == "time_series"

    def test_input_layout_values(self) -> None:
        assert InputLayout.NCHW.value == "NCHW"
        assert InputLayout.NHWC.value == "NHWC"
        assert InputLayout.NONTRIVIAL.value == "NONTRIVIAL"

    def test_input_type_values(self) -> None:
        assert InputType.IMAGE.value == "image"
        assert InputType.DEFAULT.value == "default"
        assert InputType.OPAQUE.value == "opaque"


# ══════════════════════════════════════════════════════════════════════════════
# ConverterInputSpec
# ══════════════════════════════════════════════════════════════════════════════

class TestConverterInputSpec:
    def test_dim_string(self) -> None:
        spec = ConverterInputSpec("data", dimensions=(1, 224, 224, 3))
        assert spec.dim_string == "1,224,224,3"

    def test_legacy_dim_args(self) -> None:
        spec = ConverterInputSpec("data", dimensions=(1, 3, 224, 224))
        args = spec.legacy_dim_args()
        assert "--input_dim" in args
        assert "'data'" in args
        assert "1,3,224,224" in args

    def test_qairt_shape_args(self) -> None:
        spec = ConverterInputSpec("input", dimensions=(1, 224, 224, 3))
        args = spec.qairt_shape_args()
        assert "--source_model_input_shape" in args
        assert "1,224,224,3" in args

    def test_encoding_args_default_bgr(self) -> None:
        spec = ConverterInputSpec("data")
        args = spec.encoding_args()
        assert "--input_encoding" in args
        assert "bgr" in args

    def test_encoding_args_rgba_with_output(self) -> None:
        spec = ConverterInputSpec(
            "data",
            encoding_in=InputEncoding.RGBA,
            encoding_out=InputEncoding.RGB,
        )
        args = spec.encoding_args()
        assert "rgba" in args
        assert "rgb" in args

    def test_layout_args(self) -> None:
        spec = ConverterInputSpec("data", layout=InputLayout.NCHW)
        args = spec.layout_args()
        assert "--input_layout" in args
        assert "NCHW" in args

    def test_layout_args_none(self) -> None:
        spec = ConverterInputSpec("data", layout=None)
        assert spec.layout_args() == []

    def test_input_type_args(self) -> None:
        spec = ConverterInputSpec("data", input_type=InputType.IMAGE)
        args = spec.input_type_args()
        assert "--input_type" in args
        assert "image" in args

    def test_no_dims_returns_empty_dim_args(self) -> None:
        spec = ConverterInputSpec("data")
        assert spec.legacy_dim_args() == []
        assert spec.qairt_shape_args() == []


# ══════════════════════════════════════════════════════════════════════════════
# OnnxConverterArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestOnnxConverterArgs:
    def test_minimal_args(self) -> None:
        a = OnnxConverterArgs(input_network="model.onnx")
        args = a.build()
        assert "snpe-onnx-to-dlc" in args
        assert "--input_network" in args
        assert "model.onnx" in args

    def test_output_path(self) -> None:
        a = OnnxConverterArgs("model.onnx", output_path="out.dlc")
        assert "--output_path" in a.build()
        assert "out.dlc" in a.build()

    def test_out_nodes(self) -> None:
        a = OnnxConverterArgs("model.onnx", out_nodes=["output_1", "output_2"])
        args = a.build()
        assert args.count("--out_node") == 2

    def test_float16(self) -> None:
        a = OnnxConverterArgs("model.onnx", float_bitwidth=FloatBitwidth.FP16)
        assert "--float_bitwidth" in a.build()
        assert "16" in a.build()

    def test_batch_override(self) -> None:
        a = OnnxConverterArgs("model.onnx", batch_override=8)
        assert "--batch" in a.build()
        assert "8" in a.build()

    def test_symbol_overrides(self) -> None:
        a = OnnxConverterArgs("model.onnx", symbol_overrides={"height": 224, "width": 448})
        args = a.build()
        assert args.count("--define_symbol") == 2
        assert "height" in args
        assert "224" in args

    def test_dry_run(self) -> None:
        a = OnnxConverterArgs("model.onnx", dry_run=True)
        assert "--dry_run" in a.build()

    def test_no_simplification(self) -> None:
        a = OnnxConverterArgs("model.onnx", no_simplification=True)
        assert "--no_simplification" in a.build()

    def test_masked_softmax(self) -> None:
        from quad.sdk_tools.tool_specs import MaskedSoftmaxMode
        a = OnnxConverterArgs("model.onnx", apply_masked_softmax=MaskedSoftmaxMode.COMPRESSED)
        assert "--apply_masked_softmax" in a.build()
        assert "compressed" in a.build()

    def test_with_input_specs(self) -> None:
        spec = ConverterInputSpec("data", dimensions=(1, 3, 224, 224), layout=InputLayout.NCHW)
        a = OnnxConverterArgs("model.onnx", input_specs=[spec])
        args = a.build()
        assert "--input_dim" in args
        assert "NCHW" in args


# ══════════════════════════════════════════════════════════════════════════════
# QairtConverterArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestQairtConverterArgs:
    def test_minimal_args(self) -> None:
        a = QairtConverterArgs(input_network="model.onnx")
        args = a.build()
        assert "qairt-converter" in args
        assert "--input_network" in args

    def test_out_tensor_node(self) -> None:
        a = QairtConverterArgs("model.pb", out_tensor_nodes=["out1", "out2"])
        args = a.build()
        assert args.count("--out_tensor_node") == 2
        assert "out1" in args
        assert "out2" in args

    def test_target_backend_htp(self) -> None:
        a = QairtConverterArgs("model.onnx", target_backend=TargetBackend.HTP,
                               target_soc_model="sm8750")
        args = a.build()
        assert "--target_backend" in args
        assert "HTP" in args
        assert "--target_soc_model" in args
        assert "sm8750" in args

    def test_target_soc_requires_backend(self) -> None:
        """soc_model only emitted when target_backend is also set."""
        a = QairtConverterArgs("model.onnx", target_soc_model="sm8750")
        assert "--target_soc_model" not in a.build()

    def test_export_format_strip_quant(self) -> None:
        a = QairtConverterArgs("model.onnx", export_format=ExportFormat.DLC_STRIP_QUANT)
        assert "--export_format" in a.build()
        assert "DLC_STRIP_QUANT" in a.build()

    def test_tf_saved_model_options(self) -> None:
        a = QairtConverterArgs(
            "model/saved_model",
            tf_saved_model_tag="serve",
            tf_saved_model_signature_key="serving_default",
        )
        args = a.build()
        assert "--tf_saved_model_tag" in args
        assert "serve" in args
        assert "--tf_saved_model_signature_key" in args

    def test_onnx_define_symbols(self) -> None:
        a = QairtConverterArgs("model.onnx", onnx_define_symbols={"height": 224, "width": 448})
        args = a.build()
        assert args.count("--onnx_define_symbol") == 2

    def test_lora_options(self) -> None:
        a = QairtConverterArgs("model.onnx", lora_weight_list="weights.txt",
                               quant_updatable_mode="adapter_only")
        args = a.build()
        assert "--lora_weight_list" in args
        assert "--quant_updatable_mode" in args
        assert "adapter_only" in args

    def test_remove_unused_inputs(self) -> None:
        a = QairtConverterArgs("model.onnx", remove_unused_inputs=True)
        assert "--remove_unused_inputs" in a.build()


# ══════════════════════════════════════════════════════════════════════════════
# QairtQuantizerArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestQairtQuantizerArgs:
    def test_minimal_args(self) -> None:
        a = QairtQuantizerArgs(input_dlc="model.dlc")
        args = a.build()
        assert "qairt-quantizer" in args
        assert "--input_dlc" in args
        assert "model.dlc" in args

    def test_with_input_list(self) -> None:
        a = QairtQuantizerArgs("model.dlc", input_list="calib_data.txt")
        args = a.build()
        assert "--input_list" in args
        assert "calib_data.txt" in args

    def test_int8_weights_is_default(self) -> None:
        a = QairtQuantizerArgs("m.dlc")
        assert "--weights_bitwidth" not in a.build()  # default, not emitted

    def test_int4_weights(self) -> None:
        a = QairtQuantizerArgs("m.dlc", weights_bitwidth=QuantBitwidth.INT4)
        assert "--weights_bitwidth" in a.build()
        assert "4" in a.build()

    def test_per_channel_quantization(self) -> None:
        a = QairtQuantizerArgs("m.dlc", use_per_channel_quantization=True)
        assert "--use_per_channel_quantization" in a.build()

    def test_aimet_quantizer(self) -> None:
        a = QairtQuantizerArgs("m.dlc", use_aimet_quantizer=True)
        assert "--use_aimet_quantizer" in a.build()

    def test_calibration_entropy(self) -> None:
        a = QairtQuantizerArgs("m.dlc",
                               act_quantizer_calibration=QuantCalibration.ENTROPY,
                               param_quantizer_calibration=QuantCalibration.MSE)
        args = a.build()
        assert "--act_quantizer_calibration" in args
        assert "entropy" in args
        assert "--param_quantizer_calibration" in args
        assert "mse" in args

    def test_symmetric_schema(self) -> None:
        a = QairtQuantizerArgs("m.dlc", act_quantizer_schema=QuantSchema.SYMMETRIC)
        assert "--act_quantizer_schema" in a.build()
        assert "symmetric" in a.build()

    def test_percentile_value(self) -> None:
        a = QairtQuantizerArgs("m.dlc",
                               act_quantizer_calibration=QuantCalibration.PERCENTILE,
                               percentile_calibration_value=99.5)
        args = a.build()
        assert "99.5" in args

    def test_target_backend_and_soc(self) -> None:
        a = QairtQuantizerArgs("m.dlc", target_backend=TargetBackend.HTP,
                               target_soc_model="sm8650")
        args = a.build()
        assert "HTP" in args
        assert "sm8650" in args

    def test_cle_algorithm(self) -> None:
        a = QairtQuantizerArgs("m.dlc", apply_algorithms=["cle"])
        assert "--apply_algorithms" in a.build()
        assert "cle" in a.build()

    def test_float_fallback(self) -> None:
        a = QairtQuantizerArgs("m.dlc", enable_float_fallback=True)
        assert "--enable_float_fallback" in a.build()

    def test_config_file(self) -> None:
        a = QairtQuantizerArgs("m.dlc", config_file="quant_config.yaml")
        assert "--config" in a.build()
        assert "quant_config.yaml" in a.build()


# ══════════════════════════════════════════════════════════════════════════════
# SnpeNetRunArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestSnpeNetRunArgs:
    def test_minimal_required(self) -> None:
        a = SnpeNetRunArgs(container="model.dlc", input_list="inputs.txt")
        args = a.build()
        assert "snpe-net-run" in args
        assert "--container" in args
        assert "model.dlc" in args
        assert "--input_list" in args
        assert "inputs.txt" in args

    def test_use_dsp(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", use_dsp=True)
        assert "--use_dsp" in a.build()

    def test_use_gpu(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", use_gpu=True)
        assert "--use_gpu" in a.build()

    def test_runtime_order(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt",
                           runtime_order=[RuntimeOrder.DSP, RuntimeOrder.CPU])
        args = a.build()
        assert "--runtime_order" in args
        runtime_val = args[args.index("--runtime_order") + 1]
        assert "dsp" in runtime_val
        assert "cpu" in runtime_val

    def test_perf_profile_burst(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", perf_profile=PerfProfile.BURST)
        args = a.build()
        assert "--perf_profile" in args
        assert "burst" in args

    def test_profiling_level_linting(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", profiling_level=ProfilingLevelNet.LINTING)
        args = a.build()
        assert "linting" in args

    def test_enable_cpu_fallback(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", enable_cpu_fallback=True)
        assert "--enable_cpu_fallback" in a.build()

    def test_enable_init_cache(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", enable_init_cache=True)
        assert "--enable_init_cache" in a.build()

    def test_cache_compatibility_strict(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt",
                           cache_compatibility_mode=CacheCompatibilityMode.STRICT)
        args = a.build()
        assert "strict" in " ".join(args)

    def test_platform_options_unsigned_pd(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", platform_options="unsignedPD:OFF")
        args = a.build()
        assert "--platform_options" in args
        assert "unsignedPD:OFF" in args

    def test_duration(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", duration=30)
        assert "30" in a.build()

    def test_timeout_htp(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", timeout=1000000, use_dsp=True)
        assert "1000000" in " ".join(a.build())

    def test_priority_hint_normal_high(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", priority_hint=PriorityHint.NORMAL_HIGH)
        args = a.build()
        assert "normal_high" in args

    def test_gpu_mode_float16(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", use_gpu=True, gpu_mode=GpuMode.FLOAT16)
        args = a.build()
        assert "--gpu_mode" in args
        assert "float16" in args

    def test_set_output_tensors(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", set_output_tensors=["out1", "out2"])
        args = a.build()
        assert args.count("--set_output_tensors") == 2

    def test_cpu_fxp(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", enable_cpu_fxp=True, use_cpu=True)
        assert "--enable_cpu_fxp" in a.build()

    def test_graph_init_and_execute(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt",
                           graph_init="graph1,graph2", graph_execute="graph1")
        args = a.build()
        assert "--graph_init" in args
        assert "--graph_execute" in args

    def test_deferred_init(self) -> None:
        a = SnpeNetRunArgs("m.dlc", "i.txt", deferred_init="weights")
        assert "weights" in " ".join(a.build())


# ══════════════════════════════════════════════════════════════════════════════
# DiagviewArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagviewArgs:
    def test_minimal(self) -> None:
        a = DiagviewArgs(input_log="SNPEDiag_0.log")
        args = a.build()
        assert "snpe-diagview" in args
        assert "--input_log" in args

    def test_with_csv_output(self) -> None:
        a = DiagviewArgs("diag.log", output_csv="results.csv")
        assert "--output" in a.build()
        assert "results.csv" in a.build()

    def test_chrometrace(self) -> None:
        a = DiagviewArgs("diag.log", chrometrace="linting_trace")
        args = a.build()
        assert "--chrometrace" in args
        assert "linting_trace" in args

    def test_csv_format_version_2(self) -> None:
        a = DiagviewArgs("diag.log", output_csv="out.csv", csv_format_version=2)
        assert "--csv_format_version" in a.build()
        assert "2" in a.build()


# ══════════════════════════════════════════════════════════════════════════════
# DlcInfoArgs / DlcDiffArgs
# ══════════════════════════════════════════════════════════════════════════════

class TestDlcInfoArgs:
    def test_snpe_dlc_info(self) -> None:
        a = DlcInfoArgs("model.dlc")
        args = a.build()
        assert "snpe-dlc-info" in args
        assert "--input_dlc" in args

    def test_qairt_dlc_info(self) -> None:
        a = DlcInfoArgs("model.dlc", tool="qairt-dlc-info")
        assert "qairt-dlc-info" in a.build()

    def test_save_csv(self) -> None:
        a = DlcInfoArgs("model.dlc", save="info.csv")
        assert "--save" in a.build()
        assert "info.csv" in a.build()

    def test_all_flags(self) -> None:
        a = DlcInfoArgs("model.dlc", display_memory=True,
                        display_all_encodings=True, dump_framework_trace=True)
        args = a.build()
        assert "--display_all_encodings" in args
        assert "--dump_framework_trace" in args


class TestDlcDiffArgs:
    def test_snpe_dlc_diff(self) -> None:
        a = DlcDiffArgs("m1.dlc", "m2.dlc")
        args = a.build()
        assert "snpe-dlc-diff" in args
        assert "-i1" in args
        assert "-i2" in args

    def test_qairt_dlc_diff(self) -> None:
        a = DlcDiffArgs("m1.dlc", "m2.dlc", tool="qairt-dlc-diff")
        assert "qairt-dlc-diff" in a.build()

    def test_compare_weights(self) -> None:
        a = DlcDiffArgs("m1.dlc", "m2.dlc", compare_weights=True)
        assert "-w" in a.build()

    def test_save(self) -> None:
        a = DlcDiffArgs("m1.dlc", "m2.dlc", save="diff.csv")
        assert "diff.csv" in a.build()


# ══════════════════════════════════════════════════════════════════════════════
# Diagview Constants
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagviewConstants:
    def test_timing_layers_present(self) -> None:
        for key in ("Total Inference Time", "Forward Propagate Time",
                    "RPC Execute Time", "SNPE Acc Time", "Acc Time"):
            assert key in DIAGVIEW_TIMING_LAYERS

    def test_hierarchy_described(self) -> None:
        ti = DIAGVIEW_TIMING_LAYERS["Total Inference Time"]
        fp = DIAGVIEW_TIMING_LAYERS["Forward Propagate Time"]
        acc = DIAGVIEW_TIMING_LAYERS["Acc Time"]
        assert "SNPE" in ti or "end-to-end" in ti.lower()
        assert "RPC" in fp or "Backend" in fp
        assert "HTP" in acc or "compute" in acc.lower()

    def test_diagview_notes_averaging(self) -> None:
        assert "averaged" in DIAGVIEW_NOTES["averaging"].lower()

    def test_diagview_notes_log_files(self) -> None:
        assert "SNPEDiag" in DIAGVIEW_NOTES["log_files"]

    def test_exported_from_package(self) -> None:
        from quad.sdk_tools import (  # noqa: F401
            DIAGVIEW_TIMING_LAYERS,
            DiagviewArgs,
            OnnxConverterArgs,
            QairtConverterArgs,
            QairtQuantizerArgs,
            SDK_TOOLS,
            SnpeNetRunArgs,
        )
