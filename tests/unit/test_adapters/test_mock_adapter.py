"""Tests for MockAdapter."""

from __future__ import annotations

import pytest

from quad.adapters.mock_adapter import MockAdapter
from quad.models.conversion import ConversionRequest
from quad.models.profiling import ProfileRequest


@pytest.fixture
def adapter() -> MockAdapter:
    return MockAdapter(sdk_name="qnn")


class TestMockAdapterDetectHardware:
    @pytest.mark.asyncio
    async def test_windows_profile(self, adapter: MockAdapter) -> None:
        profile = await adapter.detect_hardware("windows")
        assert profile.chipset == "Snapdragon X Elite X1E-80-100"
        assert profile.npu_tops == 45.0
        assert "npu" in profile.available_runtimes

    @pytest.mark.asyncio
    async def test_linux_profile(self, adapter: MockAdapter) -> None:
        profile = await adapter.detect_hardware("linux")
        assert "QCS2210" in profile.chipset
        assert profile.npu_tops == 1.0

    @pytest.mark.asyncio
    async def test_android_profile(self, adapter: MockAdapter) -> None:
        profile = await adapter.detect_hardware("android")
        assert "8 Elite" in profile.chipset
        assert profile.npu_tops == 48.0

    @pytest.mark.asyncio
    async def test_unknown_platform_defaults_to_windows(self, adapter: MockAdapter) -> None:
        profile = await adapter.detect_hardware("unknown")
        assert profile.platform == "windows"


class TestMockAdapterConvertModel:
    @pytest.mark.asyncio
    async def test_onnx_to_qnn(self, adapter: MockAdapter) -> None:
        request = ConversionRequest(
            source_format="onnx",
            model_path="model.onnx",
            target_sdk="qnn",
            quantization="int8",
        )
        result = await adapter.convert_model(request)
        assert result.output_path.endswith(".bin")
        assert result.quantization_applied == "int8"
        assert result.supported_ops_pct > 90
        assert result.compression_ratio == 4.0

    @pytest.mark.asyncio
    async def test_snpe_target_produces_dlc(self, adapter: MockAdapter) -> None:
        request = ConversionRequest(
            source_format="onnx",
            model_path="model.onnx",
            target_sdk="snpe",
            quantization="fp32",
        )
        result = await adapter.convert_model(request)
        assert result.output_path.endswith(".dlc")
        assert result.compression_ratio == 1.0

    @pytest.mark.asyncio
    async def test_yolo_model_has_unsupported_ops(self, adapter: MockAdapter) -> None:
        request = ConversionRequest(
            source_format="onnx",
            model_path="yolov8n.onnx",
            target_sdk="qnn",
        )
        result = await adapter.convert_model(request)
        assert "NonMaxSuppression" in result.unsupported_ops


class TestMockAdapterProfile:
    @pytest.mark.asyncio
    async def test_npu_profiling(self, adapter: MockAdapter) -> None:
        request = ProfileRequest(model_path="model.bin", platform="windows", runtime="npu")
        report = await adapter.profile(request)
        assert report.runtime_used == "npu"
        assert report.latency.mean_ms > 0
        assert report.throughput_fps > 0
        assert report.power_mw > 0
        assert report.utilization["npu"] > 50

    @pytest.mark.asyncio
    async def test_cpu_slower_than_npu(self, adapter: MockAdapter) -> None:
        req_npu = ProfileRequest(model_path="model.bin", runtime="npu")
        req_cpu = ProfileRequest(model_path="model.bin", runtime="cpu")
        report_npu = await adapter.profile(req_npu)
        report_cpu = await adapter.profile(req_cpu)
        assert report_cpu.latency.mean_ms > report_npu.latency.mean_ms

    @pytest.mark.asyncio
    async def test_layers_populated(self, adapter: MockAdapter) -> None:
        request = ProfileRequest(model_path="model.bin", runtime="npu")
        report = await adapter.profile(request)
        assert len(report.layers) > 0
        assert all(layer.latency_ms > 0 for layer in report.layers)


class TestMockAdapterSupportedOps:
    @pytest.mark.asyncio
    async def test_returns_ops_list(self, adapter: MockAdapter) -> None:
        ops = await adapter.get_supported_ops()
        assert "Conv" in ops
        assert "MatMul" in ops
        assert len(ops) > 20


class TestMockAdapterInference:
    @pytest.mark.asyncio
    async def test_execute_returns_result(self, adapter: MockAdapter) -> None:
        result = await adapter.execute_inference("model.bin", None)
        assert result["status"] == "success"
        assert result["output_shape"] == [1, 1000]
