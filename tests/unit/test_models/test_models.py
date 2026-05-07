"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from quad.models import (
    AllocationMap,
    ConversionRequest,
    ConversionResult,
    DeviceProfile,
    GeneratedCode,
    LatencyStats,
    ProfilingReport,
    ServerConfig,
    ToolError,
)


class TestDeviceProfile:
    def test_valid_profile(self) -> None:
        profile = DeviceProfile(
            chipset="Snapdragon X Elite",
            platform="windows",
            cpu_cores=12,
            cpu_arch="Oryon ARM64",
            cpu_freq_ghz=3.8,
            gpu_model="Adreno X1-85",
            gpu_tflops=4.6,
            npu_model="Hexagon NPU",
            npu_tops=45.0,
            ram_gb=32.0,
            available_runtimes=["cpu", "gpu", "npu"],
        )
        assert profile.chipset == "Snapdragon X Elite"
        assert profile.npu_tops == 45.0

    def test_invalid_platform(self) -> None:
        with pytest.raises(ValidationError):
            DeviceProfile(
                chipset="test",
                platform="invalid_platform",
                cpu_cores=1,
                cpu_arch="x",
                cpu_freq_ghz=1.0,
                gpu_model="x",
                gpu_tflops=0,
                npu_model="x",
                npu_tops=0,
                ram_gb=1,
            )

    def test_cpu_cores_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DeviceProfile(
                chipset="test",
                platform="windows",
                cpu_cores=0,
                cpu_arch="x",
                cpu_freq_ghz=1.0,
                gpu_model="x",
                gpu_tflops=0,
                npu_model="x",
                npu_tops=0,
                ram_gb=1,
            )


class TestConversionResult:
    def test_valid_result(self) -> None:
        result = ConversionResult(
            output_path="model.bin",
            model_size_mb=6.25,
            original_size_mb=25.0,
            compression_ratio=4.0,
            supported_ops_pct=96.5,
            unsupported_ops=[],
            quantization_applied="int8",
            conversion_time_s=2.5,
            target_sdk="qnn",
        )
        assert result.compression_ratio == 4.0

    def test_ops_pct_must_be_0_to_100(self) -> None:
        with pytest.raises(ValidationError):
            ConversionResult(
                output_path="x",
                model_size_mb=1,
                original_size_mb=1,
                compression_ratio=1,
                supported_ops_pct=101,
                unsupported_ops=[],
                quantization_applied="fp32",
                conversion_time_s=0,
                target_sdk="qnn",
            )


class TestAllocationMap:
    def test_valid_allocation(self) -> None:
        alloc = AllocationMap(
            allocation={"conv1": "npu", "fc1": "cpu"},
            projected_latency_ms=5.0,
            projected_power_mw=2000.0,
            projected_memory_mb=45.0,
            power_mode="balanced",
            fallback_layers=["fc1"],
            npu_utilization_pct=70.0,
            gpu_utilization_pct=10.0,
            cpu_utilization_pct=20.0,
        )
        assert alloc.allocation["conv1"] == "npu"
        assert "fc1" in alloc.fallback_layers


class TestServerConfig:
    def test_defaults(self) -> None:
        config = ServerConfig()
        assert config.adapter_mode == "mock"
        assert config.log_level == "info"

    def test_override_values(self) -> None:
        config = ServerConfig(adapter_mode="real", log_level="debug")
        assert config.adapter_mode == "real"
        assert config.log_level == "debug"


class TestToolError:
    def test_error_creation(self) -> None:
        err = ToolError(
            code="SDK_NOT_FOUND",
            message="QNN SDK not found",
            recoverable=False,
            suggestion="Install QNN SDK from QDN portal",
        )
        assert err.code == "SDK_NOT_FOUND"
        assert not err.recoverable
