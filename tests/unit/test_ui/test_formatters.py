"""Tests for the UI formatters (Phase F)."""

from __future__ import annotations

import pytest

from quad.ui import (
    format_allocation,
    format_conversion,
    format_coverage,
    format_device,
    format_doctor,
    format_profile,
    format_sdk_status,
    format_table,
    format_utilization_bar,
)


# ─── Primitives ────────────────────────────────────────────────────────────


class TestFormatTable:
    def test_basic_table(self) -> None:
        out = format_table(["A", "B"], [[1, 2], [3, 4]])
        assert "| A | B |" in out
        assert "| 1 | 2 |" in out
        assert "| 3 | 4 |" in out

    def test_alignment_separator(self) -> None:
        out = format_table(["x", "y"], [[1, 2]], align=["l", "r"])
        # ":---" left, "---:" right
        assert ":---" in out
        assert "---:" in out

    def test_floats_formatted_to_2dp(self) -> None:
        out = format_table(["x"], [[3.14159]])
        assert "3.14" in out
        assert "3.141" not in out  # 3 decimal places shouldn't appear

    def test_empty_headers_returns_empty(self) -> None:
        assert format_table([], [[1]]) == ""


class TestUtilizationBar:
    def test_full_bar_at_100(self) -> None:
        out = format_utilization_bar(100, width=10)
        assert "█" * 10 in out
        assert "░" not in out

    def test_empty_bar_at_0(self) -> None:
        out = format_utilization_bar(0, width=10)
        assert "░" * 10 in out
        # Filled char shouldn't appear at all when value is 0
        assert "█" not in out

    def test_clamps_negative(self) -> None:
        out = format_utilization_bar(-50, width=10)
        # Negative clamped to 0 → bar should be all empty
        assert "░" * 10 in out

    def test_clamps_above_100(self) -> None:
        out = format_utilization_bar(200, width=10)
        # Above 100 clamped → bar should be all filled
        assert "█" * 10 in out

    def test_label_appended(self) -> None:
        out = format_utilization_bar(50, width=10, label="NPU")
        assert "NPU" in out
        assert "50%" in out


# ─── Device ───────────────────────────────────────────────────────────────


class TestFormatDevice:
    def test_renders_chipset_header(self) -> None:
        device = {
            "chipset": "Snapdragon X Elite",
            "platform": "windows",
            "cpu_cores": 12,
            "cpu_arch": "ARM64",
            "cpu_freq_ghz": 3.8,
            "gpu_model": "Adreno X1-85",
            "gpu_tflops": 4.6,
            "npu_model": "Hexagon NPU",
            "npu_tops": 45.0,
            "ram_gb": 32.0,
            "available_runtimes": ["cpu", "gpu", "npu"],
            "sdk_path": "/sdk",
            "sdk_version": "2.45.0",
        }
        out = format_device(device)
        assert "Snapdragon X Elite" in out
        assert "Hexagon NPU" in out
        assert "45.0 TOPS" in out
        assert "12 × ARM64" in out


# ─── Profile ─────────────────────────────────────────────────────────────


class TestFormatProfile:
    def test_basic_latency_table(self) -> None:
        profile = {
            "latency": {
                "mean_ms": 2.56,
                "p50_ms": 2.20,
                "p95_ms": 4.68,
                "p99_ms": 6.28,
                "min_ms": 2.0,
                "max_ms": 9.0,
            },
            "throughput_fps": 388,
            "power_mw": 2000,
            "memory_peak_mb": 50,
            "memory_avg_mb": 40,
            "utilization": {"npu": 95, "gpu": 0, "cpu": 5},
            "runtime_used": "npu",
            "profiling_level": "detailed",
        }
        out = format_profile(profile)
        assert "388 FPS" in out
        assert "2000 mW" in out
        assert "Mean" in out
        # Utilisation bars use the █ character
        assert "█" in out

    def test_linting_bottleneck_callout(self) -> None:
        profile = {
            "latency": {"mean_ms": 5.0, "p50_ms": 5.0, "p95_ms": 5.0, "p99_ms": 5.0, "min_ms": 5.0, "max_ms": 5.0},
            "throughput_fps": 200,
            "power_mw": 2000,
            "memory_peak_mb": 50,
            "memory_avg_mb": 40,
            "utilization": {"npu": 80},
            "runtime_used": "npu",
            "profiling_level": "linting",
            "linting_layers": [
                {
                    "name": "sub_op",
                    "index": 4,
                    "total_cycles": 2_000_000,
                    "overlap_ratio": 0.21,
                    "is_bottleneck": True,
                    "optimization_hint": "Replace Sub with Conv",
                }
            ],
        }
        out = format_profile(profile)
        assert "bottleneck" in out.lower()
        assert "sub_op" in out
        assert "Replace Sub with Conv" in out


# ─── Conversion ──────────────────────────────────────────────────────────


class TestFormatConversion:
    def test_basic_conversion(self) -> None:
        result = {
            "output_path": "model.dlc",
            "original_size_mb": 13.3,
            "model_size_mb": 3.3,
            "compression_ratio": 4.05,
            "quantization_applied": "int8",
            "supported_ops_pct": 100.0,
            "conversion_time_s": 1.33,
            "target_sdk": "qairt",
        }
        out = format_conversion(result)
        assert "model.dlc" in out
        assert "13.3 MB" in out
        assert "4.05" in out
        assert "int8" in out

    def test_unsupported_ops_warning(self) -> None:
        result = {
            "output_path": "m.dlc",
            "original_size_mb": 10,
            "model_size_mb": 3,
            "compression_ratio": 3.3,
            "quantization_applied": "int8",
            "supported_ops_pct": 90,
            "conversion_time_s": 1,
            "target_sdk": "qairt",
            "unsupported_ops": ["CustomOp1", "CustomOp2"],
        }
        out = format_conversion(result)
        assert "fall back to CPU" in out
        assert "CustomOp1" in out


# ─── Allocation ──────────────────────────────────────────────────────────


class TestFormatAllocation:
    def test_renders_power_mode_and_metrics(self) -> None:
        alloc = {
            "power_mode": "balanced",
            "projected_latency_ms": 2.55,
            "projected_power_mw": 1420,
            "projected_memory_mb": 50,
            "npu_utilization_pct": 70,
            "gpu_utilization_pct": 0,
            "cpu_utilization_pct": 30,
            "fallback_layers": [],
        }
        out = format_allocation(alloc)
        assert "balanced" in out
        assert "2.55 ms" in out
        assert "1420 mW" in out
        assert "█" in out  # bars rendered

    def test_fallback_layers_warning(self) -> None:
        alloc = {
            "power_mode": "performance",
            "projected_latency_ms": 1.0,
            "projected_power_mw": 1000,
            "projected_memory_mb": 30,
            "npu_utilization_pct": 80,
            "gpu_utilization_pct": 0,
            "cpu_utilization_pct": 20,
            "fallback_layers": ["bn1", "bn2"],
        }
        out = format_allocation(alloc)
        assert "fall back to CPU" in out
        assert "bn1" in out


# ─── Doctor ──────────────────────────────────────────────────────────────


class TestFormatDoctor:
    def test_dict_form(self) -> None:
        checks = [
            {"name": "Python", "status": "pass", "message": "3.12"},
            {"name": "SDK", "status": "warn", "message": "missing"},
            {"name": "Test", "status": "fail", "message": "broken"},
        ]
        out = format_doctor(checks)
        assert "Python" in out
        assert "SDK" in out
        assert "1 passed" in out
        assert "1 warnings" in out
        assert "1 errors" in out

    def test_object_form(self) -> None:
        from quad.cli.doctor import CheckResult
        checks = [CheckResult(name="X", status="pass", message="ok")]
        out = format_doctor(checks)
        assert "X" in out


# ─── Coverage ────────────────────────────────────────────────────────────


class TestFormatCoverage:
    def test_single_target_report(self) -> None:
        report = {
            "target": "hexagon_v75",
            "total_ops": 14,
            "supported_ops": 14,
            "coverage_pct": 100.0,
            "is_fully_covered": True,
            "unsupported_ops": [],
            "fallback_recommendation": "All ops supported",
        }
        out = format_coverage(report)
        assert "hexagon_v75" in out
        assert "100" in out
        assert "✓" in out

    def test_partial_coverage_lists_unsupported(self) -> None:
        report = {
            "target": "hexagon_v75",
            "total_ops": 5,
            "supported_ops": 3,
            "coverage_pct": 60.0,
            "is_fully_covered": False,
            "unsupported_ops": [
                {"op_type": "CustomOp", "name": "node_3"},
                {"op_type": "OtherCustom", "name": "node_4"},
            ],
            "fallback_recommendation": "2 op(s) will fall back",
        }
        out = format_coverage(report)
        assert "60" in out  # The percentage
        assert "CustomOp" in out
        assert "fall back" in out

    def test_multi_target_table(self) -> None:
        coverage = {
            "qnpu_v3": {
                "target": "qnpu_v3",
                "total_ops": 10,
                "supported_ops": 10,
                "coverage_pct": 100,
                "is_fully_covered": True,
            },
            "cpu_oryon": {
                "target": "cpu_oryon",
                "total_ops": 10,
                "supported_ops": 10,
                "coverage_pct": 100,
                "is_fully_covered": True,
            },
        }
        out = format_coverage(coverage)
        assert "qnpu_v3" in out
        assert "cpu_oryon" in out


# ─── SDK status ──────────────────────────────────────────────────────────


class TestFormatSdkStatus:
    def test_no_sdk_message(self) -> None:
        out = format_sdk_status(None)
        assert "No QAIRT/SNPE SDK detected" in out
        assert "quad sdk install" in out

    def test_with_sdk(self) -> None:
        info = {
            "flavor": "qairt",
            "version": "2.45.0",
            "root": "/opt/qairt",
            "bin_dir": "/opt/qairt/bin",
            "source": "env-var",
            "has_qairt_converter": True,
            "has_snpe_net_run": False,
        }
        out = format_sdk_status(info)
        assert "qairt" in out
        assert "2.45.0" in out
        assert "✓" in out  # has_qairt_converter
        assert "✗" in out  # has_snpe_net_run is False
