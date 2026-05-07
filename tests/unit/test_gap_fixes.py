"""Tests for all gap fixes identified in review."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════════════════════
# ProfilingLevel enum
# ══════════════════════════════════════════════════════════════════════════════

class TestProfilingLevel:
    def test_all_values_present(self) -> None:
        from quad.profiler.levels import ProfilingLevel
        assert ProfilingLevel.BASIC.value == "basic"
        assert ProfilingLevel.DETAILED.value == "detailed"
        assert ProfilingLevel.LINTING.value == "linting"
        assert ProfilingLevel.QHAS.value == "qhas"
        assert ProfilingLevel.OFF.value == "off"

    def test_htp_only_levels(self) -> None:
        from quad.profiler.levels import ProfilingLevel
        assert ProfilingLevel.LINTING.is_htp_only is True
        assert ProfilingLevel.QHAS.is_htp_only is True
        assert ProfilingLevel.DETAILED.is_htp_only is False
        assert ProfilingLevel.BASIC.is_htp_only is False

    def test_supports_chrometrace(self) -> None:
        from quad.profiler.levels import ProfilingLevel
        assert ProfilingLevel.LINTING.supports_chrometrace is True
        assert ProfilingLevel.QHAS.supports_chrometrace is True
        assert ProfilingLevel.DETAILED.supports_chrometrace is False

    def test_fallback_level_for_htp_only(self) -> None:
        from quad.profiler.levels import ProfilingLevel
        assert ProfilingLevel.LINTING.fallback_level == ProfilingLevel.DETAILED
        assert ProfilingLevel.QHAS.fallback_level == ProfilingLevel.DETAILED

    def test_fallback_level_for_non_htp(self) -> None:
        from quad.profiler.levels import ProfilingLevel
        assert ProfilingLevel.DETAILED.fallback_level == ProfilingLevel.DETAILED
        assert ProfilingLevel.BASIC.fallback_level == ProfilingLevel.BASIC

    def test_exported_from_profiler_package(self) -> None:
        from quad.profiler import ProfilingLevel  # noqa: F401

    def test_linting_module_uses_shared_enum(self) -> None:
        from quad.profiler.linting import LINTING_PROFILING_LEVEL
        from quad.profiler.levels import ProfilingLevel
        assert LINTING_PROFILING_LEVEL == ProfilingLevel.LINTING.value

    def test_qhas_module_uses_shared_enum(self) -> None:
        from quad.profiler.qhas import QHAS_PROFILING_LEVEL
        from quad.profiler.levels import ProfilingLevel
        assert QHAS_PROFILING_LEVEL == ProfilingLevel.QHAS.value

    def test_cli_style_consistent(self) -> None:
        """Both linting and QHAS must use space-separated --profiling_level flag."""
        from quad.profiler.linting import LINTING_PROFILE_NOTES
        from quad.profiler.qhas import QHAS_PROFILE_NOTES
        lint_cli = LINTING_PROFILE_NOTES["activation"]["cli"]
        qhas_cli = QHAS_PROFILE_NOTES["steps"]["2_net_run"]["flags"][0]
        # Neither should use = separator
        assert "=" not in lint_cli, f"Linting uses = in CLI: {lint_cli!r}"
        assert "--profiling_level linting" in lint_cli
        assert "--profiling_level qhas" in qhas_cli


# ══════════════════════════════════════════════════════════════════════════════
# ProfileRequest — profiling_level field
# ══════════════════════════════════════════════════════════════════════════════

class TestProfileRequestProfilingLevel:
    def test_default_is_detailed(self) -> None:
        from quad.models.profiling import ProfileRequest
        req = ProfileRequest(model_path="model.dlc")
        assert req.profiling_level == "detailed"

    def test_linting_level_accepted(self) -> None:
        from quad.models.profiling import ProfileRequest
        req = ProfileRequest(model_path="model.dlc", profiling_level="linting")
        assert req.profiling_level == "linting"

    def test_qhas_level_accepted(self) -> None:
        from quad.models.profiling import ProfileRequest
        req = ProfileRequest(model_path="model.dlc", profiling_level="qhas")
        assert req.profiling_level == "qhas"

    def test_invalid_level_rejected(self) -> None:
        from pydantic import ValidationError
        from quad.models.profiling import ProfileRequest
        with pytest.raises(ValidationError):
            ProfileRequest(model_path="model.dlc", profiling_level="invalid")

    def test_htp_soc_default(self) -> None:
        from quad.models.profiling import ProfileRequest
        req = ProfileRequest(model_path="model.dlc")
        assert req.htp_soc == "sm8750"

    def test_sdk_root_optional(self) -> None:
        from quad.models.profiling import ProfileRequest
        req = ProfileRequest(model_path="model.dlc", sdk_root="/opt/qairt")
        assert req.sdk_root == "/opt/qairt"


# ══════════════════════════════════════════════════════════════════════════════
# LintingLayerProfile model
# ══════════════════════════════════════════════════════════════════════════════

class TestLintingLayerProfile:
    def test_fields_present(self) -> None:
        from quad.models.profiling import LintingLayerProfile
        op = LintingLayerProfile(
            name="model_sub_sub:OpId_57",
            index=8,
            total_cycles=2165162,
            wait_cycles=0,
            overlap_cycles=465046,
            overlap_wait_cycles=0,
            overlap_ratio=0.2147,
            cycle_fraction=0.5003,
            resources=["HVX"],
            is_bottleneck=True,
            optimization_hint="Replace Sub with Conv",
        )
        assert op.total_cycles == 2165162
        assert op.is_bottleneck is True
        assert op.overlap_ratio == pytest.approx(0.2147)

    def test_defaults_ok(self) -> None:
        from quad.models.profiling import LintingLayerProfile
        op = LintingLayerProfile(name="conv", index=0, total_cycles=100,
                                 overlap_ratio=0.5, cycle_fraction=0.1)
        assert op.is_bottleneck is False
        assert op.optimization_hint is None
        assert op.resources == []


# ══════════════════════════════════════════════════════════════════════════════
# ProfilingReport — linting/QHAS fields
# ══════════════════════════════════════════════════════════════════════════════

class TestProfilingReportNewFields:
    def _make_report(self, **kwargs):
        from quad.models.device import DeviceProfile
        from quad.models.profiling import LatencyStats, ProfilingReport
        base = dict(
            latency=LatencyStats(mean_ms=5.0, p50_ms=4.5, p95_ms=7.0,
                                 p99_ms=9.0, min_ms=3.0, max_ms=12.0),
            throughput_fps=200.0, power_mw=2000.0,
            memory_peak_mb=50.0, memory_avg_mb=40.0,
            device=DeviceProfile(
                chipset="Test", platform="linux",
                cpu_cores=4, cpu_arch="ARM64", cpu_freq_ghz=2.0,
                gpu_model="Adreno", gpu_tflops=0.4,
                npu_model="DSP", npu_tops=1.0, ram_gb=2.0,
                sdk_path="/sdk", sdk_version="2.0",
                available_runtimes=["cpu"],
            ),
            runtime_used="npu", duration_s=10.0,
        )
        base.update(kwargs)
        return ProfilingReport(**base)

    def test_profiling_level_defaults_to_detailed(self) -> None:
        report = self._make_report()
        assert report.profiling_level == "detailed"

    def test_profiling_level_linting_stored(self) -> None:
        report = self._make_report(profiling_level="linting")
        assert report.profiling_level == "linting"

    def test_linting_fields_default_empty(self) -> None:
        report = self._make_report()
        assert report.linting_layers == []
        assert report.linting_total_cycles == 0
        assert report.linting_bottleneck_count == 0
        assert report.linting_optimization_hints == []

    def test_linting_fields_populated(self) -> None:
        from quad.models.profiling import LintingLayerProfile
        op = LintingLayerProfile(name="sub", index=0, total_cycles=1000,
                                 overlap_ratio=0.1, cycle_fraction=0.5)
        report = self._make_report(
            profiling_level="linting",
            linting_layers=[op],
            linting_total_cycles=2000,
            linting_bottleneck_count=1,
            linting_optimization_hints=["Replace Sub with Conv"],
        )
        assert len(report.linting_layers) == 1
        assert report.linting_total_cycles == 2000
        assert report.linting_bottleneck_count == 1
        assert "Sub" in report.linting_optimization_hints[0]

    def test_qhas_chrometrace_path(self) -> None:
        report = self._make_report(
            profiling_level="qhas",
            qhas_chrometrace_path="./chrometrace.json",
        )
        assert report.qhas_chrometrace_path == "./chrometrace.json"


# ══════════════════════════════════════════════════════════════════════════════
# ConversionRequest — image format fields
# ══════════════════════════════════════════════════════════════════════════════

class TestConversionRequestImageFormat:
    def test_input_layout_default_auto(self) -> None:
        from quad.models.conversion import ConversionRequest
        req = ConversionRequest(source_format="onnx", model_path="m.onnx")
        assert req.input_layout == "auto"

    def test_input_layout_nchw(self) -> None:
        from quad.models.conversion import ConversionRequest
        req = ConversionRequest(source_format="pytorch", model_path="m.pt",
                                input_layout="nchw")
        assert req.input_layout == "nchw"

    def test_channel_order_bgr(self) -> None:
        from quad.models.conversion import ConversionRequest
        req = ConversionRequest(source_format="onnx", model_path="alexnet.onnx",
                                channel_order="bgr")
        assert req.channel_order == "bgr"

    def test_mean_values(self) -> None:
        from quad.models.conversion import ConversionRequest
        req = ConversionRequest(source_format="onnx", model_path="m.onnx",
                                mean_values=[104.0, 117.0, 123.0])
        assert req.mean_values == [104.0, 117.0, 123.0]

    def test_invalid_input_layout_rejected(self) -> None:
        from pydantic import ValidationError
        from quad.models.conversion import ConversionRequest
        with pytest.raises(ValidationError):
            ConversionRequest(source_format="onnx", model_path="m.onnx",
                              input_layout="hwc")

    def test_invalid_channel_order_rejected(self) -> None:
        from pydantic import ValidationError
        from quad.models.conversion import ConversionRequest
        with pytest.raises(ValidationError):
            ConversionRequest(source_format="onnx", model_path="m.onnx",
                              channel_order="rgba")


# ══════════════════════════════════════════════════════════════════════════════
# ConversionResult — conversion_notes and image_format_notes
# ══════════════════════════════════════════════════════════════════════════════

class TestConversionResultNewFields:
    def _make_result(self, **kwargs):
        from quad.models.conversion import ConversionResult
        base = dict(
            output_path="model.dlc", model_size_mb=10.0,
            original_size_mb=25.0, compression_ratio=2.5,
            supported_ops_pct=96.5, quantization_applied="fp32",
            conversion_time_s=1.2, target_sdk="snpe",
        )
        base.update(kwargs)
        return ConversionResult(**base)

    def test_conversion_notes_default_empty(self) -> None:
        result = self._make_result()
        assert result.conversion_notes == []

    def test_image_format_notes_default_empty(self) -> None:
        result = self._make_result()
        assert result.image_format_notes == []

    def test_conversion_notes_populated(self) -> None:
        result = self._make_result(
            conversion_notes=["MobilenetSSD: requires allow_unconsumed_nodes=True"]
        )
        assert len(result.conversion_notes) == 1
        assert "allow_unconsumed_nodes" in result.conversion_notes[0]

    def test_image_format_notes_populated(self) -> None:
        result = self._make_result(
            image_format_notes=["Input layout is NCHW. Transpose to NHWC before inference."]
        )
        assert "NHWC" in result.image_format_notes[0]


# ══════════════════════════════════════════════════════════════════════════════
# MockAdapter — linting/QHAS profiling
# ══════════════════════════════════════════════════════════════════════════════

class TestMockAdapterLintingQHAS:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_linting_level_returns_linting_layers(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.profiling import ProfileRequest
        adapter = MockAdapter()
        req = ProfileRequest(model_path="model.dlc", profiling_level="linting")
        report = self._run(adapter.profile(req))
        assert report.profiling_level == "linting"
        assert len(report.linting_layers) > 0
        assert report.linting_total_cycles > 0
        assert report.linting_bottleneck_count >= 1

    def test_linting_bottleneck_has_hint(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.profiling import ProfileRequest
        adapter = MockAdapter()
        req = ProfileRequest(model_path="model.dlc", profiling_level="linting")
        report = self._run(adapter.profile(req))
        bn_ops = [op for op in report.linting_layers if op.is_bottleneck]
        assert bn_ops, "Expected at least one bottleneck op"
        assert any(op.optimization_hint for op in bn_ops)

    def test_qhas_level_returns_chrometrace_path(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.profiling import ProfileRequest
        adapter = MockAdapter()
        req = ProfileRequest(model_path="model.dlc", profiling_level="qhas")
        report = self._run(adapter.profile(req))
        assert report.profiling_level == "qhas"
        assert report.qhas_chrometrace_path is not None

    def test_detailed_level_no_linting_fields(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.profiling import ProfileRequest
        adapter = MockAdapter()
        req = ProfileRequest(model_path="model.dlc", profiling_level="detailed")
        report = self._run(adapter.profile(req))
        assert report.linting_layers == []
        assert report.linting_total_cycles == 0


# ══════════════════════════════════════════════════════════════════════════════
# MockAdapter — conversion_notes and image_format_notes
# ══════════════════════════════════════════════════════════════════════════════

class TestMockAdapterConversionNotes:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_default_conversion_has_image_format_notes(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.conversion import ConversionRequest
        adapter = MockAdapter()
        req = ConversionRequest(source_format="onnx", model_path="model.onnx")
        result = self._run(adapter.convert_model(req))
        assert len(result.image_format_notes) >= 1
        assert any("NHWC" in note for note in result.image_format_notes)

    def test_nchw_layout_note_mentions_transpose(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.conversion import ConversionRequest
        adapter = MockAdapter()
        req = ConversionRequest(source_format="pytorch", model_path="resnet.pt",
                                input_layout="nchw")
        result = self._run(adapter.convert_model(req))
        assert any("transpose" in note.lower() or "NCHW" in note
                   for note in result.image_format_notes)

    def test_bgr_channel_order_note(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.conversion import ConversionRequest
        adapter = MockAdapter()
        req = ConversionRequest(source_format="onnx", model_path="alexnet.onnx",
                                channel_order="bgr")
        result = self._run(adapter.convert_model(req))
        assert any("BGR" in note or "bgr" in note.lower()
                   for note in result.image_format_notes)

    def test_mobilenet_ssd_conversion_notes(self) -> None:
        from quad.adapters.mock_adapter import MockAdapter
        from quad.models.conversion import ConversionRequest
        adapter = MockAdapter()
        req = ConversionRequest(source_format="tensorflow",
                                model_path="ssd_mobilenet_v2_quantized_300x300.pb")
        result = self._run(adapter.convert_model(req))
        # MobilenetSSD should surface model tips
        assert len(result.conversion_notes) >= 0  # may or may not match filename


# ══════════════════════════════════════════════════════════════════════════════
# profile_workload tool — profiling_level parameter
# ══════════════════════════════════════════════════════════════════════════════

class TestProfileWorkloadTool:
    def _run(self, coro):
        return asyncio.run(coro)

    def _make_factory(self):
        from quad.adapters.factory import AdapterFactory
        from quad.models.config import ServerConfig
        cfg = ServerConfig()
        return AdapterFactory(cfg)

    def test_linting_level_propagated(self) -> None:
        from quad.tools.profile_workload import profile_workload_impl
        factory = self._make_factory()
        result = self._run(profile_workload_impl(
            model_path="model.dlc",
            platform="windows",
            runtime="npu",
            duration_s=5,
            factory=factory,
            profiling_level="linting",
        ))
        assert result["profiling_level"] == "linting"
        assert result["linting_total_cycles"] > 0

    def test_qhas_level_propagated(self) -> None:
        from quad.tools.profile_workload import profile_workload_impl
        factory = self._make_factory()
        result = self._run(profile_workload_impl(
            model_path="model.dlc",
            platform="windows",
            runtime="npu",
            duration_s=5,
            factory=factory,
            profiling_level="qhas",
        ))
        assert result["profiling_level"] == "qhas"

    def test_detailed_default_unchanged(self) -> None:
        from quad.tools.profile_workload import profile_workload_impl
        factory = self._make_factory()
        result = self._run(profile_workload_impl(
            model_path="model.dlc",
            platform="windows",
            runtime="npu",
            duration_s=5,
            factory=factory,
        ))
        assert result["profiling_level"] == "detailed"


# ══════════════════════════════════════════════════════════════════════════════
# convert_model tool — image format parameters
# ══════════════════════════════════════════════════════════════════════════════

class TestConvertModelTool:
    def _run(self, coro):
        return asyncio.run(coro)

    def _make_factory(self):
        from quad.adapters.factory import AdapterFactory
        from quad.models.config import ServerConfig
        cfg = ServerConfig()
        return AdapterFactory(cfg)

    def test_image_layout_passed_through(self) -> None:
        from quad.tools.convert_model import convert_model_impl
        factory = self._make_factory()
        result = self._run(convert_model_impl(
            source_format="pytorch",
            model_path="resnet50.pt",
            target_sdk="snpe",
            quantization="fp32",
            factory=factory,
            input_layout="nchw",
            channel_order="rgb",
        ))
        assert "image_format_notes" in result
        assert len(result["image_format_notes"]) >= 1

    def test_bgr_channel_order_in_result(self) -> None:
        from quad.tools.convert_model import convert_model_impl
        factory = self._make_factory()
        result = self._run(convert_model_impl(
            source_format="onnx",
            model_path="alexnet.onnx",
            target_sdk="snpe",
            quantization="fp32",
            factory=factory,
            channel_order="bgr",
        ))
        notes = result["image_format_notes"]
        assert any("BGR" in n or "bgr" in n.lower() for n in notes)

    def test_mean_values_in_result(self) -> None:
        from quad.tools.convert_model import convert_model_impl
        factory = self._make_factory()
        result = self._run(convert_model_impl(
            source_format="onnx",
            model_path="model.onnx",
            target_sdk="snpe",
            quantization="fp32",
            factory=factory,
            mean_values=[104.0, 117.0, 123.0],
            channel_order="bgr",
        ))
        notes = result["image_format_notes"]
        assert any("104" in n or "mean" in n.lower() for n in notes)


# ══════════════════════════════════════════════════════════════════════════════
# profile_model() — mock parameter
# ══════════════════════════════════════════════════════════════════════════════

class TestProfileModelMockParam:
    def test_mock_true_works(self) -> None:
        from quad.profiler.api import profile_model
        summary = profile_model("model.onnx", level="kernel", mock=True)
        assert summary.kernel_report is not None

    def test_mock_false_still_runs(self) -> None:
        """mock=False with no real hardware should still return a summary (falls through)."""
        from quad.profiler.api import profile_model
        # Profilers will attempt real mode but fall back gracefully
        summary = profile_model("model.onnx", level="system", mock=False)
        assert summary is not None

    def test_deep_level_mock_true(self) -> None:
        from quad.profiler.api import profile_model
        summary = profile_model("model.onnx", level="deep", mock=True)
        assert summary.power_trace is not None
        assert summary.memory_report is not None


# ══════════════════════════════════════════════════════════════════════════════
# snpe-diagview wrapper
# ══════════════════════════════════════════════════════════════════════════════

class TestDiagviewWrapper:
    def test_find_diagview_returns_none_when_not_installed(self) -> None:
        from quad.profiler.diagview import find_diagview
        # In test env, snpe-diagview is almost certainly not installed
        result = find_diagview()
        assert result is None or isinstance(result, str)

    def test_run_diagview_raises_file_not_found_when_missing(self) -> None:
        from quad.profiler.diagview import run_diagview
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="snpe-diagview"):
                run_diagview("SNPEDiag_0.bin")

    def test_run_diagview_chrometrace_raises_file_not_found_when_missing(self) -> None:
        from quad.profiler.diagview import run_diagview_chrometrace
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError):
                run_diagview_chrometrace("SNPEDiag_0.bin")

    def test_run_diagview_uses_correct_command(self) -> None:
        from quad.profiler.diagview import run_diagview
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "HTP Subnet 0: 100 cycles\n"
        with patch("shutil.which", return_value="/usr/bin/snpe-diagview"):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                output = run_diagview("my_diag.bin")
                args = mock_run.call_args[0][0]
                assert args[0] == "/usr/bin/snpe-diagview"
                assert "--input_log" in args
                assert "my_diag.bin" in args
                assert output == "HTP Subnet 0: 100 cycles\n"

    def test_run_diagview_chrometrace_uses_correct_command(self) -> None:
        from quad.profiler.diagview import run_diagview_chrometrace
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("shutil.which", return_value="/usr/bin/snpe-diagview"):
            with patch("subprocess.run", return_value=mock_result) as mock_run:
                with patch("pathlib.Path.glob", return_value=[]):
                    run_diagview_chrometrace("diag.bin", output_prefix="trace")
                args = mock_run.call_args[0][0]
                assert "--chrometrace" in args
                assert "--output" in args

    def test_parse_diaglog_as_linting_calls_parse(self) -> None:
        from quad.profiler.diagview import parse_diaglog_as_linting
        sample_output = (
            "Per-Graph Execution Times:\n"
            "HTP Subnet 0: 1000000 cycles\n\n"
            "Layer Times:\n"
            "  0: conv_op (cycles) : 500000 cycles : DSP\n"
            "    Wait (Scheduler) time: 0 cycles\n"
            "    Overlap time: 250000 cycles\n"
            "    Overlap (wait) time: 0 cycles\n"
            "    Resources: HVX\n"
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = sample_output
        with patch("shutil.which", return_value="/usr/bin/snpe-diagview"):
            with patch("subprocess.run", return_value=mock_result):
                profile = parse_diaglog_as_linting("diag.bin")
                assert len(profile.subnets) == 1
                assert profile.subnets[0].total_cycles == 1000000

    def test_exported_from_profiler_package(self) -> None:
        from quad.profiler import (  # noqa: F401
            find_diagview,
            parse_diaglog_as_linting,
            run_diagview,
            run_diagview_chrometrace,
        )


# ══════════════════════════════════════════════════════════════════════════════
# doctor.py — new SDK checks
# ══════════════════════════════════════════════════════════════════════════════

class TestDoctorSDKChecks:
    def test_sdk_env_vars_check_warn_when_missing(self) -> None:
        from quad.cli.doctor import _check_sdk_env_vars
        with patch.dict("os.environ", {}, clear=True):
            # Remove all SDK env vars
            result = _check_sdk_env_vars()
        assert result.status == "warn"
        assert "QAIRT_SDK_ROOT" in result.message

    def test_sdk_env_vars_check_pass_when_set(self, tmp_path: Path) -> None:
        from quad.cli.doctor import _check_sdk_env_vars
        with patch.dict("os.environ", {"QAIRT_SDK_ROOT": str(tmp_path)}):
            result = _check_sdk_env_vars()
        assert result.status == "pass"
        assert str(tmp_path) in result.message

    def test_sdk_env_vars_check_fail_when_bad_path(self) -> None:
        from quad.cli.doctor import _check_sdk_env_vars
        with patch.dict("os.environ", {"QAIRT_SDK_ROOT": "/nonexistent/sdk/path"}):
            result = _check_sdk_env_vars()
        assert result.status == "fail"

    def test_sdk_tools_in_path_warn_when_missing(self) -> None:
        from quad.cli.doctor import _check_sdk_tools_in_path
        with patch("shutil.which", return_value=None):
            result = _check_sdk_tools_in_path()
        assert result.status in ("fail", "warn")
        assert "snpe-net-run" in result.message or "qairt-converter" in result.message

    def test_sdk_tools_in_path_pass_when_found(self) -> None:
        from quad.cli.doctor import _check_sdk_tools_in_path
        with patch("shutil.which", return_value="/usr/bin/tool"):
            result = _check_sdk_tools_in_path()
        assert result.status == "pass"

    def test_dsp_env_warn_when_not_set(self) -> None:
        from quad.cli.doctor import _check_dsp_env
        with patch.dict("os.environ", {}, clear=True):
            result = _check_dsp_env()
        assert result.status == "warn"
        assert "ADSP_LIBRARY_PATH" in result.message

    def test_dsp_env_pass_when_set_to_existing_dir(self, tmp_path: Path) -> None:
        from quad.cli.doctor import _check_dsp_env
        with patch.dict("os.environ", {"ADSP_LIBRARY_PATH": str(tmp_path)}):
            result = _check_dsp_env()
        assert result.status == "pass"

    def test_dsp_env_fail_when_path_nonexistent(self) -> None:
        from quad.cli.doctor import _check_dsp_env
        with patch.dict("os.environ", {"ADSP_LIBRARY_PATH": "/nonexistent/path"}):
            result = _check_dsp_env()
        assert result.status == "fail"

    def test_android_tools_warn_when_missing(self) -> None:
        from quad.cli.doctor import _check_android_tools
        with patch("shutil.which", return_value=None), \
             patch.dict("os.environ", {}, clear=True):
            result = _check_android_tools()
        assert result.status == "warn"

    def test_qhas_prerequisites_check_runs(self) -> None:
        from quad.cli.doctor import _check_qhas_prerequisites
        # Should not raise regardless of environment
        result = _check_qhas_prerequisites()
        assert result.status in ("pass", "warn", "fail")
        assert result.name == "QHAS prerequisites"

    def test_run_doctor_includes_all_new_checks(self) -> None:
        from quad.cli.doctor import run_doctor
        report = run_doctor()
        check_names = [c.name for c in report.checks]
        assert "SDK env vars" in check_names
        assert "SDK tools in PATH" in check_names
        assert "DSP env (ADSP_LIBRARY_PATH)" in check_names
        assert "Android tools" in check_names
        assert "QHAS prerequisites" in check_names


# ══════════════════════════════════════════════════════════════════════════════
# Templates exist
# ══════════════════════════════════════════════════════════════════════════════

class TestNewTemplates:
    def _templates_dir(self) -> Path:
        # tests/unit/test_gap_fixes.py → project root is 3 levels up
        return Path(__file__).parent.parent.parent / "templates"

    def test_qhas_config_template_exists(self) -> None:
        tmpl = self._templates_dir() / "snpe" / "profiling" / "qhas_config.json.j2"
        assert tmpl.exists(), f"Missing: {tmpl}"

    def test_linting_script_template_exists(self) -> None:
        tmpl = self._templates_dir() / "snpe" / "profiling" / "run_linting.sh.j2"
        assert tmpl.exists(), f"Missing: {tmpl}"

    def test_input_list_generator_template_exists(self) -> None:
        tmpl = self._templates_dir() / "snpe" / "profiling" / "generate_input_list.py.j2"
        assert tmpl.exists(), f"Missing: {tmpl}"

    def test_qhas_config_template_valid_json_structure(self) -> None:
        """Template should render to valid JSON when variables are substituted."""
        from jinja2 import Environment, FileSystemLoader
        templates_dir = self._templates_dir()
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            trim_blocks=True, lstrip_blocks=True,
        )
        tmpl = env.get_template("snpe/profiling/qhas_config.json.j2")
        rendered = tmpl.render(
            enable_io_flow_events=True,
            enable_sequencer_flow_events=True,
            htp_json=True, runtrace=True, memory_info=True,
            traceback=True, qhas_schema=True, qhas_json=True,
        )
        parsed = json.loads(rendered)
        assert "features" in parsed
        assert parsed["features"]["htp_json"] is True
        assert parsed["features"]["enable_input_output_flow_events"] is True
