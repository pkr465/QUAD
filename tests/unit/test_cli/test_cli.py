"""Tests for QUAD CLI tools — quickstart, doctor, benchmark."""

from __future__ import annotations

import pytest


class TestQuickstart:
    """Tests for the quickstart wizard."""

    def test_quickstart_runs_without_error(self) -> None:
        from quad.cli.quickstart import run_quickstart, QuickstartResult

        result = run_quickstart(mock=True)
        assert isinstance(result, QuickstartResult)

    def test_quickstart_returns_valid_result(self) -> None:
        from quad.cli.quickstart import run_quickstart, QuickstartResult

        result = run_quickstart(mock=True)
        assert result.device_detected != ""
        assert result.model_compiled != ""
        assert result.profile_generated is True
        assert result.code_generated is True
        assert result.total_time_s >= 0.0

    def test_quickstart_specific_model(self) -> None:
        from quad.cli.quickstart import run_quickstart

        result = run_quickstart(model_name="yolov8n", mock=True)
        assert result.model_compiled == "yolov8n"


class TestDoctor:
    """Tests for the doctor diagnostics."""

    def test_doctor_reports_all_checks(self) -> None:
        from quad.cli.doctor import run_doctor, DoctorReport, CheckResult

        report = run_doctor()
        assert isinstance(report, DoctorReport)
        # Should have at least 7 checks
        assert len(report.checks) >= 7

    def test_doctor_check_results_have_valid_status(self) -> None:
        from quad.cli.doctor import run_doctor

        report = run_doctor()
        valid_statuses = {"pass", "warn", "fail"}
        for check in report.checks:
            assert check.status in valid_statuses
            assert check.name != ""
            assert check.message != ""

    def test_doctor_python_version_passes(self) -> None:
        from quad.cli.doctor import run_doctor

        report = run_doctor()
        python_check = next(c for c in report.checks if "Python" in c.name)
        assert python_check.status == "pass"

    def test_doctor_quad_importable(self) -> None:
        from quad.cli.doctor import run_doctor

        report = run_doctor()
        quad_check = next(c for c in report.checks if "QUAD package" in c.name)
        assert quad_check.status == "pass"

    def test_doctor_properties(self) -> None:
        from quad.cli.doctor import run_doctor

        report = run_doctor()
        # Properties should work
        _ = report.all_passed
        _ = report.warnings
        _ = report.errors
        assert isinstance(report.warnings, list)
        assert isinstance(report.errors, list)


class TestBenchmark:
    """Tests for the benchmark suite."""

    def test_benchmark_produces_results_for_all_default_models(self) -> None:
        from quad.cli.benchmark import run_benchmark, BenchmarkReport

        report = run_benchmark(mock=True)
        assert isinstance(report, BenchmarkReport)
        # Should have results for all 3 default models
        assert len(report.results) == 3

    def test_benchmark_result_fields(self) -> None:
        from quad.cli.benchmark import run_benchmark

        report = run_benchmark(mock=True)
        for result in report.results:
            assert result.model_name != ""
            assert result.latency_ms > 0
            assert result.throughput_fps > 0
            assert result.power_mw > 0

    def test_benchmark_specific_models(self) -> None:
        from quad.cli.benchmark import run_benchmark

        report = run_benchmark(models=["mobilenetv2"], mock=True)
        assert len(report.results) == 1
        assert report.results[0].model_name == "mobilenetv2"

    def test_benchmark_device_reported(self) -> None:
        from quad.cli.benchmark import run_benchmark

        report = run_benchmark(device="npu", mock=True)
        assert report.device == "npu"

    def test_benchmark_has_timestamp(self) -> None:
        from quad.cli.benchmark import run_benchmark

        report = run_benchmark(mock=True)
        assert report.timestamp != ""
        assert "UTC" in report.timestamp

    def test_benchmark_list_models(self) -> None:
        from quad.cli.benchmark import list_benchmark_models

        models = list_benchmark_models()
        assert "mobilenetv2" in models
        assert "resnet50" in models
        assert "yolov8n" in models


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_cli_main_imports_without_error(self) -> None:
        from quad.cli.main import app, main

        assert app is not None
        assert callable(main)

    def test_cli_app_has_commands(self) -> None:
        from quad.cli.main import app

        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "quickstart" in result.output
        assert "doctor" in result.output
        assert "benchmark" in result.output
        assert "mode" in result.output


class TestDoctorRealMode:
    """Tests for `quad doctor --real-mode` strict pre-flight."""

    def _strip_sdk_env(self, monkeypatch):
        for var in (
            "QAIRT_SDK_ROOT",
            "QNN_SDK_ROOT",
            "SNPE_ROOT",
            "QUAD_ADAPTER_MODE",
            "QUAD_STRICT_REAL",
            "ADSP_LIBRARY_PATH",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_real_mode_adds_adapter_check(self, monkeypatch) -> None:
        from quad.cli.doctor import run_doctor

        self._strip_sdk_env(monkeypatch)
        report = run_doctor(real_mode=True)
        names = [c.name for c in report.checks]
        assert "Adapter mode (real)" in names

    def test_real_mode_escalates_sdk_warnings(self, monkeypatch) -> None:
        """SDK env-var warnings must escalate to fail under --real-mode."""
        from quad.cli.doctor import run_doctor

        self._strip_sdk_env(monkeypatch)
        report = run_doctor(real_mode=True)
        sdk_check = next(c for c in report.checks if c.name == "SDK env vars")
        assert sdk_check.status == "fail"
        assert "real-mode strict" in sdk_check.message

    def test_non_real_mode_keeps_sdk_warnings_as_warn(self, monkeypatch) -> None:
        from quad.cli.doctor import run_doctor

        self._strip_sdk_env(monkeypatch)
        report = run_doctor(real_mode=False)
        sdk_check = next(c for c in report.checks if c.name == "SDK env vars")
        assert sdk_check.status == "warn"

    def test_doctor_default_is_lenient(self, monkeypatch) -> None:
        """Default `quad doctor` (no --real-mode) shouldn't add the real check."""
        from quad.cli.doctor import run_doctor

        self._strip_sdk_env(monkeypatch)
        report = run_doctor()
        names = [c.name for c in report.checks]
        assert "Adapter mode (real)" not in names

    def test_doctor_cli_real_mode_exits_non_zero(self, monkeypatch) -> None:
        """`quad doctor --real-mode` must exit non-zero when SDK is missing."""
        from quad.cli.main import app
        from typer.testing import CliRunner

        self._strip_sdk_env(monkeypatch)
        runner = CliRunner()
        result = runner.invoke(app, ["doctor", "--real-mode"])
        assert result.exit_code == 1
        assert "Real-mode pre-flight failed" in result.output


class TestModeCommand:
    """Tests for the `quad mode` CLI command."""

    def _strip_sdk_env(self, monkeypatch):
        for var in (
            "QAIRT_SDK_ROOT",
            "QNN_SDK_ROOT",
            "SNPE_ROOT",
            "QUAD_ADAPTER_MODE",
            "QUAD_STRICT_REAL",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_mode_reports_mock_by_default(self, monkeypatch, tmp_path) -> None:
        from quad.cli.main import app
        from typer.testing import CliRunner

        self._strip_sdk_env(monkeypatch)
        monkeypatch.chdir(tmp_path)  # no quad.toml present → defaults
        runner = CliRunner()
        result = runner.invoke(app, ["mode"])
        assert result.exit_code == 0
        assert "adapter_mode:" in result.output
        assert "mock" in result.output
        assert "NOT READY" in result.output

    def test_mode_set_prints_export(self, monkeypatch) -> None:
        from quad.cli.main import app
        from typer.testing import CliRunner

        self._strip_sdk_env(monkeypatch)
        runner = CliRunner()
        result = runner.invoke(app, ["mode", "--set", "real"])
        assert result.exit_code == 0
        assert "export QUAD_ADAPTER_MODE=real" in result.output

    def test_mode_set_rejects_invalid(self, monkeypatch) -> None:
        from quad.cli.main import app
        from typer.testing import CliRunner

        self._strip_sdk_env(monkeypatch)
        runner = CliRunner()
        result = runner.invoke(app, ["mode", "--set", "bogus"])
        assert result.exit_code == 2

    def test_mode_reports_ready_when_sdk_present(self, monkeypatch, tmp_path) -> None:
        from quad.cli.main import app
        from typer.testing import CliRunner

        self._strip_sdk_env(monkeypatch)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(tmp_path))
        monkeypatch.setenv("QUAD_ADAPTER_MODE", "real")
        runner = CliRunner()
        result = runner.invoke(app, ["mode"])
        assert result.exit_code == 0
        assert "READY" in result.output
        # Make sure we didn't accidentally print NOT READY (substring trap)
        assert "NOT READY" not in result.output
