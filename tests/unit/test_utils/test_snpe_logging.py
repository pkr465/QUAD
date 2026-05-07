"""Tests for SNPE logging utilities."""

from __future__ import annotations

import pytest

from quad.utils.snpe_logging import (
    SNPELogLevel,
    SNPELoggingConfig,
    get_logging_config,
)


class TestSNPELogLevel:
    def test_all_levels_exist(self) -> None:
        assert SNPELogLevel.VERBOSE
        assert SNPELogLevel.INFO
        assert SNPELogLevel.WARN
        assert SNPELogLevel.ERROR
        assert SNPELogLevel.FATAL

    def test_increasing_severity(self) -> None:
        """Levels must be in VERBOSE < INFO < WARN < ERROR < FATAL order."""
        levels = [SNPELogLevel.VERBOSE, SNPELogLevel.INFO, SNPELogLevel.WARN,
                  SNPELogLevel.ERROR, SNPELogLevel.FATAL]
        severities = [l.severity for l in levels]
        assert severities == sorted(severities)

    def test_c_enum_names(self) -> None:
        assert SNPELogLevel.VERBOSE.c_enum == "SNPE_LOG_LEVEL_VERBOSE"
        assert SNPELogLevel.INFO.c_enum == "SNPE_LOG_LEVEL_INFO"
        assert SNPELogLevel.WARN.c_enum == "SNPE_LOG_LEVEL_WARN"
        assert SNPELogLevel.ERROR.c_enum == "SNPE_LOG_LEVEL_ERROR"
        assert SNPELogLevel.FATAL.c_enum == "SNPE_LOG_LEVEL_FATAL"

    def test_java_enum_names(self) -> None:
        assert SNPELogLevel.VERBOSE.java_enum == "NeuralNetwork.LogLevel.LOG_VERBOSE"
        assert SNPELogLevel.INFO.java_enum == "NeuralNetwork.LogLevel.LOG_INFO"
        assert SNPELogLevel.WARN.java_enum == "NeuralNetwork.LogLevel.LOG_WARN"
        assert SNPELogLevel.ERROR.java_enum == "NeuralNetwork.LogLevel.LOG_ERROR"
        assert SNPELogLevel.FATAL.java_enum == "NeuralNetwork.LogLevel.LOG_FATAL"

    def test_captures_same_level(self) -> None:
        assert SNPELogLevel.WARN.captures(SNPELogLevel.WARN) is True

    def test_captures_higher_levels(self) -> None:
        """Setting WARN captures WARN + ERROR + FATAL."""
        assert SNPELogLevel.WARN.captures(SNPELogLevel.ERROR) is True
        assert SNPELogLevel.WARN.captures(SNPELogLevel.FATAL) is True

    def test_does_not_capture_lower_levels(self) -> None:
        """Setting WARN does NOT capture VERBOSE or INFO."""
        assert SNPELogLevel.WARN.captures(SNPELogLevel.VERBOSE) is False
        assert SNPELogLevel.WARN.captures(SNPELogLevel.INFO) is False

    def test_production_safe_levels(self) -> None:
        assert SNPELogLevel.WARN.is_production_safe is True
        assert SNPELogLevel.ERROR.is_production_safe is True
        assert SNPELogLevel.FATAL.is_production_safe is True
        assert SNPELogLevel.INFO.is_production_safe is False
        assert SNPELogLevel.VERBOSE.is_production_safe is False

    def test_for_environment_production(self) -> None:
        assert SNPELogLevel.for_environment(is_production=True) == SNPELogLevel.WARN

    def test_for_environment_development(self) -> None:
        assert SNPELogLevel.for_environment(is_production=False) == SNPELogLevel.INFO


class TestSNPELoggingConfig:
    def test_android_init_call(self) -> None:
        cfg = SNPELoggingConfig(
            level=SNPELogLevel.INFO,
            is_android=True,
        )
        call = cfg.c_init_call
        assert "Snpe_Util_InitializeLogging(" in call
        assert "SNPE_LOG_LEVEL_INFO" in call
        # Android: no path parameter
        assert "Path" not in call

    def test_non_android_init_call_with_path(self) -> None:
        cfg = SNPELoggingConfig(
            level=SNPELogLevel.WARN,
            log_path="/tmp/snpe_logs",
            is_android=False,
        )
        call = cfg.c_init_call
        assert "Snpe_Util_InitializeLoggingPath(" in call
        assert "SNPE_LOG_LEVEL_WARN" in call
        assert "/tmp/snpe_logs" in call

    def test_set_level_call(self) -> None:
        cfg = SNPELoggingConfig(level=SNPELogLevel.ERROR)
        assert "Snpe_Util_SetLogLevel(SNPE_LOG_LEVEL_ERROR)" in cfg.c_set_level_call

    def test_terminate_call(self) -> None:
        cfg = SNPELoggingConfig()
        assert cfg.c_terminate_call == "Snpe_Util_TerminateLogging();"

    def test_java_init_call(self) -> None:
        cfg = SNPELoggingConfig(level=SNPELogLevel.INFO, is_android=True)
        call = cfg.java_init_call
        assert "SNPE.logger.initializeLogging(" in call
        assert "NeuralNetwork.LogLevel.LOG_INFO" in call

    def test_java_set_level_call(self) -> None:
        cfg = SNPELoggingConfig(level=SNPELogLevel.WARN)
        assert "SNPE.logger.setLogLevel(NeuralNetwork.LogLevel.LOG_WARN)" in cfg.java_set_level_call

    def test_java_terminate_call(self) -> None:
        cfg = SNPELoggingConfig()
        assert cfg.java_terminate_call == "SNPE.logger.terminateLogging();"

    def test_performance_warning_for_verbose(self) -> None:
        cfg = SNPELoggingConfig(level=SNPELogLevel.VERBOSE)
        warning = cfg.performance_warning()
        assert warning is not None
        assert "performance" in warning.lower()

    def test_no_performance_warning_for_warn(self) -> None:
        cfg = SNPELoggingConfig(level=SNPELogLevel.WARN)
        assert cfg.performance_warning() is None


class TestGetLoggingConfig:
    def test_production_returns_warn(self) -> None:
        cfg = get_logging_config(is_production=True)
        assert cfg.level == SNPELogLevel.WARN

    def test_development_returns_info(self) -> None:
        cfg = get_logging_config(is_production=False)
        assert cfg.level == SNPELogLevel.INFO

    def test_android_no_log_path(self) -> None:
        cfg = get_logging_config(is_android=True, log_path="/tmp/logs")
        assert cfg.log_path == ""  # Android ignores log_path

    def test_non_android_keeps_log_path(self) -> None:
        cfg = get_logging_config(is_android=False, log_path="/tmp/logs")
        assert cfg.log_path == "/tmp/logs"


class TestAndroidTemplateHasLogging:
    def test_kt_template_has_snpe_logging_imports(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/android"))
        t = env.get_template("InferenceEngine.kt.j2")
        rendered = t.render(model_path="model.dlc")
        assert "NeuralNetwork.LogLevel" in rendered
        assert "SNPE.logger.initializeLogging" in rendered
        assert "SNPE.logger.setLogLevel" in rendered
        assert "SNPE.logger.terminateLogging" in rendered

    def test_kt_template_production_uses_warn(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/android"))
        t = env.get_template("InferenceEngine.kt.j2")
        rendered = t.render(model_path="model.dlc", is_production=True)
        assert "LOG_WARN" in rendered

    def test_c_template_has_snpe_util_include(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        t = env.get_template("inference.c.j2")
        rendered = t.render(model_path="model.dlc")
        assert "SNPEUtil.h" in rendered

    def test_c_template_with_logging_enabled(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        t = env.get_template("inference.c.j2")
        rendered = t.render(
            model_path="model.dlc",
            log_level="warn",
            is_android=False,
            log_path="/tmp/snpe_logs",
        )
        assert "Snpe_Util_InitializeLoggingPath" in rendered
        assert "SNPE_LOG_LEVEL_WARN" in rendered
        assert "Snpe_Util_TerminateLogging" in rendered
