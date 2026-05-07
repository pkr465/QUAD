"""Integration tests — full tool calls through MockAdapter."""

from __future__ import annotations

import pytest

from quad.adapters.factory import AdapterFactory
from quad.models.config import ServerConfig
from quad.tools.convert_model import convert_model_impl
from quad.tools.generate_code import generate_code_impl
from quad.tools.hardware_detect import hardware_detect_impl
from quad.tools.orchestrate_workload import orchestrate_workload_impl
from quad.tools.profile_workload import profile_workload_impl

TEMPLATE_DIR = "templates"


@pytest.fixture
def factory() -> AdapterFactory:
    config = ServerConfig(adapter_mode="mock")
    return AdapterFactory(config)


class TestHardwareDetectTool:
    @pytest.mark.asyncio
    async def test_returns_device_profile(self, factory: AdapterFactory) -> None:
        result = await hardware_detect_impl("windows", factory)
        assert result["chipset"] == "Snapdragon X Elite X1E-80-100"
        assert result["npu_tops"] == 45.0
        assert result["cpu_cores"] == 12
        assert "npu" in result["available_runtimes"]

    @pytest.mark.asyncio
    async def test_linux_returns_qcs2210(self, factory: AdapterFactory) -> None:
        result = await hardware_detect_impl("linux", factory)
        assert "QCS2210" in result["chipset"]
        assert result["npu_tops"] == 1.0

    @pytest.mark.asyncio
    async def test_android_returns_snapdragon_8(self, factory: AdapterFactory) -> None:
        result = await hardware_detect_impl("android", factory)
        assert "8 Elite" in result["chipset"]
        assert result["npu_tops"] == 48.0


class TestConvertModelTool:
    @pytest.mark.asyncio
    async def test_onnx_to_qnn_int8(self, factory: AdapterFactory) -> None:
        result = await convert_model_impl("onnx", "model.onnx", "qnn", "int8", factory)
        assert result["output_path"].endswith(".bin")
        assert result["quantization_applied"] == "int8"
        assert result["compression_ratio"] == 4.0
        assert result["supported_ops_pct"] > 90

    @pytest.mark.asyncio
    async def test_snpe_produces_dlc(self, factory: AdapterFactory) -> None:
        result = await convert_model_impl("onnx", "model.onnx", "snpe", "fp32", factory)
        assert result["output_path"].endswith(".dlc")
        assert result["compression_ratio"] == 1.0

    @pytest.mark.asyncio
    async def test_int4_quantization(self, factory: AdapterFactory) -> None:
        result = await convert_model_impl("pytorch", "model.pt", "qnn", "int4", factory)
        assert result["quantization_applied"] == "int4"
        assert result["compression_ratio"] == 8.0


class TestProfileWorkloadTool:
    @pytest.mark.asyncio
    async def test_npu_profiling(self, factory: AdapterFactory) -> None:
        result = await profile_workload_impl("model.bin", "windows", "npu", 10, factory)
        assert result["runtime_used"] == "npu"
        assert result["latency"]["mean_ms"] > 0
        assert result["throughput_fps"] > 0
        assert result["power_mw"] > 0
        assert result["utilization"]["npu"] > 50

    @pytest.mark.asyncio
    async def test_cpu_profiling(self, factory: AdapterFactory) -> None:
        result = await profile_workload_impl("model.bin", "windows", "cpu", 5, factory)
        assert result["runtime_used"] == "cpu"
        assert result["utilization"]["cpu"] > 50

    @pytest.mark.asyncio
    async def test_layers_included(self, factory: AdapterFactory) -> None:
        result = await profile_workload_impl("model.bin", "linux", "npu", 10, factory)
        assert len(result["layers"]) > 0
        assert all("name" in layer for layer in result["layers"])

    @pytest.mark.asyncio
    async def test_device_profile_included(self, factory: AdapterFactory) -> None:
        result = await profile_workload_impl("model.bin", "android", "auto", 10, factory)
        assert "8 Elite" in result["device"]["chipset"]


class TestOrchestrateWorkloadTool:
    @pytest.mark.asyncio
    async def test_performance_mode_maximizes_npu(self, factory: AdapterFactory) -> None:
        result = await orchestrate_workload_impl("model.bin", "performance", factory)
        assert result["power_mode"] == "performance"
        assert result["npu_utilization_pct"] >= 60  # Most layers on NPU

    @pytest.mark.asyncio
    async def test_efficiency_mode_uses_more_cpu(self, factory: AdapterFactory) -> None:
        result = await orchestrate_workload_impl("model.bin", "efficiency", factory)
        assert result["power_mode"] == "efficiency"
        assert result["cpu_utilization_pct"] > 0

    @pytest.mark.asyncio
    async def test_all_layers_allocated(self, factory: AdapterFactory) -> None:
        result = await orchestrate_workload_impl("model.bin", "balanced", factory)
        assert len(result["allocation"]) > 0
        # Every layer must be assigned to a valid runtime
        for runtime in result["allocation"].values():
            assert runtime in ("cpu", "gpu", "npu")

    @pytest.mark.asyncio
    async def test_projected_metrics_positive(self, factory: AdapterFactory) -> None:
        result = await orchestrate_workload_impl("model.bin", "balanced", factory)
        assert result["projected_latency_ms"] > 0
        assert result["projected_power_mw"] > 0
        assert result["projected_memory_mb"] > 0


class TestGenerateCodeTool:
    @pytest.mark.asyncio
    async def test_windows_python(self) -> None:
        result = await generate_code_impl("windows", "qnn", "python", "model.bin", "templates")
        assert "inference.py" in result["source_files"]
        assert result["language"] == "python"
        assert result["platform"] == "windows"
        assert "model.bin" in result["source_files"]["inference.py"]

    @pytest.mark.asyncio
    async def test_windows_cpp(self) -> None:
        result = await generate_code_impl("windows", "qnn", "cpp", "model.bin", "templates")
        assert "inference.cpp" in result["source_files"]
        assert "CMakeLists.txt" in result["source_files"]

    @pytest.mark.asyncio
    async def test_linux_arduino(self) -> None:
        result = await generate_code_impl("linux", "snpe", "arduino_sketch", "model.dlc", "templates")
        assert "inference.ino" in result["source_files"]
        assert "model.dlc" in result["source_files"]["inference.ino"]

    @pytest.mark.asyncio
    async def test_android_kotlin(self) -> None:
        result = await generate_code_impl("android", "snpe", "kotlin", "model.dlc", "templates")
        assert "InferenceEngine.kt" in result["source_files"]

    @pytest.mark.asyncio
    async def test_build_instructions_present(self) -> None:
        result = await generate_code_impl("windows", "qnn", "python", "model.bin", "templates")
        assert len(result["build_instructions"]) > 0

    @pytest.mark.asyncio
    async def test_dependencies_present(self) -> None:
        result = await generate_code_impl("windows", "qnn", "python", "model.bin", "templates")
        assert len(result["dependencies"]) > 0
