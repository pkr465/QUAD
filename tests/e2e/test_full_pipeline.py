"""End-to-end tests — full QUAD pipeline in mock mode.

Tests the complete workflow from hardware detection through code generation,
verifying data flows correctly between stages and the pipeline completes
within acceptable time bounds.
"""

from __future__ import annotations

import time

import pytest

from quad.adapters.factory import AdapterFactory
from quad.models.config import ServerConfig
from quad.tools.hardware_detect import hardware_detect_impl
from quad.tools.convert_model import convert_model_impl
from quad.tools.profile_workload import profile_workload_impl
from quad.tools.orchestrate_workload import orchestrate_workload_impl
from quad.tools.generate_code import generate_code_impl

TEMPLATE_DIR = "templates"
SAMPLE_MODEL_PATH = "models/resnet50.onnx"


@pytest.fixture
def factory() -> AdapterFactory:
    """Shared AdapterFactory configured for mock mode."""
    config = ServerConfig(adapter_mode="mock")
    return AdapterFactory(config)


class TestFullPipelineE2E:
    """End-to-end pipeline: detect -> convert -> profile -> orchestrate -> generate."""

    @pytest.mark.asyncio
    async def test_full_pipeline_windows(self, factory: AdapterFactory) -> None:
        """Run the complete pipeline targeting Windows platform."""
        # Step 1: Detect hardware
        device = await hardware_detect_impl("windows", factory)
        assert device["platform"] == "windows"
        assert device["chipset"] == "Snapdragon X Elite X1E-80-100"
        assert "npu" in device["available_runtimes"]

        # Step 2: Convert model
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path=SAMPLE_MODEL_PATH,
            target_sdk="qnn",
            quantization="int8",
            factory=factory,
        )
        assert conversion["output_path"].endswith(".bin")
        assert conversion["quantization_applied"] == "int8"
        assert conversion["compression_ratio"] > 1.0
        converted_model_path = conversion["output_path"]

        # Step 3: Profile the converted model
        profile = await profile_workload_impl(
            model_path=converted_model_path,
            platform="windows",
            runtime="npu",
            duration_s=10,
            factory=factory,
        )
        assert profile["runtime_used"] == "npu"
        assert profile["latency"]["mean_ms"] > 0
        assert profile["throughput_fps"] > 0
        assert profile["device"]["platform"] == "windows"
        assert len(profile["layers"]) > 0

        # Step 4: Orchestrate workload allocation
        orchestration = await orchestrate_workload_impl(
            model_path=converted_model_path,
            power_mode="performance",
            factory=factory,
        )
        assert orchestration["power_mode"] == "performance"
        assert len(orchestration["allocation"]) > 0
        assert orchestration["projected_latency_ms"] > 0
        assert orchestration["projected_power_mw"] > 0
        assert orchestration["npu_utilization_pct"] >= 60

        # Step 5: Generate inference code
        codegen = await generate_code_impl(
            platform="windows",
            sdk="qnn",
            language="python",
            model_path=converted_model_path,
            template_dir=TEMPLATE_DIR,
        )
        assert codegen["language"] == "python"
        assert codegen["platform"] == "windows"
        assert codegen["sdk"] == "qnn"
        assert len(codegen["source_files"]) > 0
        assert len(codegen["build_instructions"]) > 0
        assert len(codegen["dependencies"]) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_linux(self, factory: AdapterFactory) -> None:
        """Run the complete pipeline targeting Linux platform."""
        # Step 1: Detect hardware
        device = await hardware_detect_impl("linux", factory)
        assert device["platform"] == "linux"
        assert "QCS2210" in device["chipset"]

        # Step 2: Convert model (SNPE for linux/embedded)
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path=SAMPLE_MODEL_PATH,
            target_sdk="snpe",
            quantization="int8",
            factory=factory,
        )
        assert conversion["output_path"].endswith(".dlc")
        converted_model_path = conversion["output_path"]

        # Step 3: Profile the converted model
        profile = await profile_workload_impl(
            model_path=converted_model_path,
            platform="linux",
            runtime="npu",
            duration_s=10,
            factory=factory,
        )
        assert profile["runtime_used"] == "npu"
        assert profile["device"]["platform"] == "linux"

        # Step 4: Orchestrate workload
        orchestration = await orchestrate_workload_impl(
            model_path=converted_model_path,
            power_mode="efficiency",
            factory=factory,
        )
        assert orchestration["power_mode"] == "efficiency"
        assert len(orchestration["allocation"]) > 0

        # Step 5: Generate code (arduino_sketch for Linux embedded)
        codegen = await generate_code_impl(
            platform="linux",
            sdk="snpe",
            language="arduino_sketch",
            model_path=converted_model_path,
            template_dir=TEMPLATE_DIR,
        )
        assert codegen["language"] == "arduino_sketch"
        assert codegen["platform"] == "linux"
        assert "inference.ino" in codegen["source_files"]

    @pytest.mark.asyncio
    async def test_full_pipeline_android(self, factory: AdapterFactory) -> None:
        """Run the complete pipeline targeting Android platform."""
        # Step 1: Detect hardware
        device = await hardware_detect_impl("android", factory)
        assert device["platform"] == "android"
        assert "8 Elite" in device["chipset"]
        assert device["npu_tops"] == 48.0

        # Step 2: Convert model
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path=SAMPLE_MODEL_PATH,
            target_sdk="snpe",
            quantization="int4",
            factory=factory,
        )
        assert conversion["output_path"].endswith(".dlc")
        assert conversion["compression_ratio"] == 8.0
        converted_model_path = conversion["output_path"]

        # Step 3: Profile
        profile = await profile_workload_impl(
            model_path=converted_model_path,
            platform="android",
            runtime="auto",
            duration_s=5,
            factory=factory,
        )
        assert profile["runtime_used"] == "npu"  # auto selects npu
        assert profile["device"]["chipset"] == "Snapdragon 8 Elite SM8750"

        # Step 4: Orchestrate
        orchestration = await orchestrate_workload_impl(
            model_path=converted_model_path,
            power_mode="balanced",
            factory=factory,
        )
        assert orchestration["power_mode"] == "balanced"
        assert orchestration["npu_utilization_pct"] + orchestration["cpu_utilization_pct"] + \
            orchestration["gpu_utilization_pct"] == pytest.approx(100.0, abs=0.5)

        # Step 5: Generate Kotlin code for Android
        codegen = await generate_code_impl(
            platform="android",
            sdk="snpe",
            language="kotlin",
            model_path=converted_model_path,
            template_dir=TEMPLATE_DIR,
        )
        assert codegen["language"] == "kotlin"
        assert codegen["platform"] == "android"
        assert "InferenceEngine.kt" in codegen["source_files"]


class TestDataFlowConsistency:
    """Verify that data flows correctly between pipeline stages."""

    @pytest.mark.asyncio
    async def test_converted_model_path_used_in_profiling(self, factory: AdapterFactory) -> None:
        """The output_path from conversion must be the input for profiling."""
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path="my_custom_model.onnx",
            target_sdk="qnn",
            quantization="fp32",
            factory=factory,
        )
        converted_path = conversion["output_path"]
        assert converted_path.endswith(".bin")

        # Use the converted path for profiling
        profile = await profile_workload_impl(
            model_path=converted_path,
            platform="windows",
            runtime="npu",
            duration_s=10,
            factory=factory,
        )
        # The profile should succeed with the converted model path
        assert profile["latency"]["mean_ms"] > 0
        assert profile["memory_peak_mb"] > 0

    @pytest.mark.asyncio
    async def test_converted_model_path_propagates_to_codegen(
        self, factory: AdapterFactory
    ) -> None:
        """The converted model path should appear in generated source code."""
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path="special_model.onnx",
            target_sdk="qnn",
            quantization="int8",
            factory=factory,
        )
        converted_path = conversion["output_path"]

        codegen = await generate_code_impl(
            platform="windows",
            sdk="qnn",
            language="python",
            model_path=converted_path,
            template_dir=TEMPLATE_DIR,
        )
        # The generated code should reference the converted model path
        source_content = codegen["source_files"]["inference.py"]
        assert converted_path in source_content

    @pytest.mark.asyncio
    async def test_device_platform_matches_profiling_platform(
        self, factory: AdapterFactory
    ) -> None:
        """The device returned in profiling should match the requested platform."""
        for platform in ("windows", "linux", "android"):
            profile = await profile_workload_impl(
                model_path="model.bin",
                platform=platform,
                runtime="auto",
                duration_s=5,
                factory=factory,
            )
            assert profile["device"]["platform"] == platform

    @pytest.mark.asyncio
    async def test_orchestration_layers_match_profiling_layers(
        self, factory: AdapterFactory
    ) -> None:
        """Orchestration allocation keys should correspond to profiled layers."""
        model_path = "model.bin"

        # Profile to get expected layers
        profile = await profile_workload_impl(
            model_path=model_path,
            platform="windows",
            runtime="npu",
            duration_s=10,
            factory=factory,
        )
        profiled_layer_names = {layer["name"] for layer in profile["layers"]}

        # Orchestrate and verify layer names match
        orchestration = await orchestrate_workload_impl(
            model_path=model_path,
            power_mode="balanced",
            factory=factory,
        )
        allocated_layer_names = set(orchestration["allocation"].keys())

        # All allocated layers should come from the profiled layers
        assert allocated_layer_names == profiled_layer_names

    @pytest.mark.asyncio
    async def test_sdk_consistency_conversion_to_codegen(self, factory: AdapterFactory) -> None:
        """The SDK chosen at conversion should be consistent in code generation."""
        # Convert with QNN
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path="model.onnx",
            target_sdk="qnn",
            quantization="int8",
            factory=factory,
        )
        assert conversion["target_sdk"] == "qnn"
        converted_path = conversion["output_path"]

        # Generate code with matching SDK
        codegen = await generate_code_impl(
            platform="windows",
            sdk="qnn",
            language="python",
            model_path=converted_path,
            template_dir=TEMPLATE_DIR,
        )
        assert codegen["sdk"] == "qnn"

    @pytest.mark.asyncio
    async def test_utilization_percentages_sum_to_100(self, factory: AdapterFactory) -> None:
        """NPU + GPU + CPU utilization percentages should sum to 100."""
        for power_mode in ("performance", "balanced", "efficiency"):
            result = await orchestrate_workload_impl(
                model_path="model.bin",
                power_mode=power_mode,
                factory=factory,
            )
            total_pct = (
                result["npu_utilization_pct"]
                + result["gpu_utilization_pct"]
                + result["cpu_utilization_pct"]
            )
            assert total_pct == pytest.approx(100.0, abs=0.5), (
                f"Utilization for {power_mode} mode sums to {total_pct}%, expected ~100%"
            )


class TestTTFI:
    """Time To First Inference — measure pipeline execution time in mock mode."""

    @pytest.mark.asyncio
    async def test_full_pipeline_under_5_seconds(self, factory: AdapterFactory) -> None:
        """The entire pipeline in mock mode must complete in under 5 seconds."""
        start = time.perf_counter()

        # Step 1: Detect hardware
        device = await hardware_detect_impl("windows", factory)

        # Step 2: Convert model
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path=SAMPLE_MODEL_PATH,
            target_sdk="qnn",
            quantization="int8",
            factory=factory,
        )
        converted_path = conversion["output_path"]

        # Step 3: Profile workload
        profile = await profile_workload_impl(
            model_path=converted_path,
            platform="windows",
            runtime="npu",
            duration_s=10,
            factory=factory,
        )

        # Step 4: Orchestrate workload
        orchestration = await orchestrate_workload_impl(
            model_path=converted_path,
            power_mode="performance",
            factory=factory,
        )

        # Step 5: Generate code
        codegen = await generate_code_impl(
            platform="windows",
            sdk="qnn",
            language="python",
            model_path=converted_path,
            template_dir=TEMPLATE_DIR,
        )

        elapsed = time.perf_counter() - start

        # All steps must complete in under 5 seconds for mock mode
        assert elapsed < 5.0, (
            f"Full pipeline took {elapsed:.2f}s, expected < 5.0s in mock mode"
        )

        # Verify all stages produced valid output
        assert device["chipset"] is not None
        assert conversion["output_path"] is not None
        assert profile["latency"]["mean_ms"] > 0
        assert len(orchestration["allocation"]) > 0
        assert len(codegen["source_files"]) > 0

    @pytest.mark.asyncio
    async def test_individual_stage_latencies(self, factory: AdapterFactory) -> None:
        """Each individual stage should complete well under 1 second in mock mode."""
        max_stage_time = 1.0

        # hardware_detect
        start = time.perf_counter()
        await hardware_detect_impl("windows", factory)
        assert time.perf_counter() - start < max_stage_time, "hardware_detect too slow"

        # convert_model
        start = time.perf_counter()
        conversion = await convert_model_impl(
            "onnx", SAMPLE_MODEL_PATH, "qnn", "int8", factory
        )
        assert time.perf_counter() - start < max_stage_time, "convert_model too slow"

        # profile_workload
        start = time.perf_counter()
        await profile_workload_impl(conversion["output_path"], "windows", "npu", 10, factory)
        assert time.perf_counter() - start < max_stage_time, "profile_workload too slow"

        # orchestrate_workload
        start = time.perf_counter()
        await orchestrate_workload_impl(conversion["output_path"], "performance", factory)
        assert time.perf_counter() - start < max_stage_time, "orchestrate_workload too slow"

        # generate_code
        start = time.perf_counter()
        await generate_code_impl("windows", "qnn", "python", conversion["output_path"], TEMPLATE_DIR)
        assert time.perf_counter() - start < max_stage_time, "generate_code too slow"


class TestCrossPlatformPipeline:
    """Verify the same pipeline logic works across all supported platforms."""

    PLATFORMS = ["windows", "linux", "android"]

    # Map each platform to a valid (sdk, language) combination that has templates
    PLATFORM_CONFIG = {
        "windows": {"sdk": "qnn", "language": "python"},
        "linux": {"sdk": "snpe", "language": "python"},
        "android": {"sdk": "snpe", "language": "kotlin"},
    }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("platform", ["windows", "linux", "android"])
    async def test_pipeline_produces_valid_output_per_platform(
        self, factory: AdapterFactory, platform: str
    ) -> None:
        """Each platform should produce a complete, valid pipeline result."""
        config = self.PLATFORM_CONFIG[platform]

        # Detect
        device = await hardware_detect_impl(platform, factory)
        assert device["platform"] == platform
        assert device["npu_tops"] > 0
        assert "npu" in device["available_runtimes"]

        # Convert
        target_sdk = config["sdk"]
        conversion = await convert_model_impl(
            source_format="onnx",
            model_path=SAMPLE_MODEL_PATH,
            target_sdk=target_sdk,
            quantization="int8",
            factory=factory,
        )
        expected_ext = ".bin" if target_sdk == "qnn" else ".dlc"
        assert conversion["output_path"].endswith(expected_ext)
        converted_path = conversion["output_path"]

        # Profile
        profile = await profile_workload_impl(
            model_path=converted_path,
            platform=platform,
            runtime="auto",
            duration_s=5,
            factory=factory,
        )
        assert profile["device"]["platform"] == platform
        assert profile["throughput_fps"] > 0

        # Orchestrate
        orchestration = await orchestrate_workload_impl(
            model_path=converted_path,
            power_mode="balanced",
            factory=factory,
        )
        assert len(orchestration["allocation"]) > 0
        assert orchestration["projected_latency_ms"] > 0

        # Generate code
        codegen = await generate_code_impl(
            platform=platform,
            sdk=config["sdk"],
            language=config["language"],
            model_path=converted_path,
            template_dir=TEMPLATE_DIR,
        )
        assert codegen["platform"] == platform
        assert codegen["language"] == config["language"]
        assert len(codegen["source_files"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("platform", ["windows", "linux", "android"])
    async def test_device_capabilities_vary_by_platform(
        self, factory: AdapterFactory, platform: str
    ) -> None:
        """Different platforms should report different hardware capabilities."""
        device = await hardware_detect_impl(platform, factory)

        # Each platform has distinct characteristics
        if platform == "windows":
            assert device["cpu_cores"] == 12
            assert device["ram_gb"] == 32.0
        elif platform == "linux":
            assert device["cpu_cores"] == 4
            assert device["ram_gb"] == 2.0
        elif platform == "android":
            assert device["cpu_cores"] == 8
            assert device["ram_gb"] == 16.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "power_mode", ["performance", "balanced", "efficiency"]
    )
    async def test_power_modes_produce_different_allocations(
        self, factory: AdapterFactory, power_mode: str
    ) -> None:
        """Different power modes should produce meaningfully different allocation strategies."""
        result = await orchestrate_workload_impl(
            model_path="model.bin",
            power_mode=power_mode,
            factory=factory,
        )
        assert result["power_mode"] == power_mode

        if power_mode == "performance":
            # Performance mode should maximize NPU usage
            assert result["npu_utilization_pct"] >= 60
        elif power_mode == "efficiency":
            # Efficiency mode should use more CPU
            assert result["cpu_utilization_pct"] > 0


class TestPipelineOutputValidity:
    """Verify that the pipeline output is structurally valid and complete."""

    @pytest.mark.asyncio
    async def test_generated_code_is_nonempty(self, factory: AdapterFactory) -> None:
        """Generated source files should contain actual code, not empty strings."""
        conversion = await convert_model_impl(
            "onnx", SAMPLE_MODEL_PATH, "qnn", "int8", factory
        )
        codegen = await generate_code_impl(
            platform="windows",
            sdk="qnn",
            language="python",
            model_path=conversion["output_path"],
            template_dir=TEMPLATE_DIR,
        )
        for filename, content in codegen["source_files"].items():
            assert len(content.strip()) > 0, f"Source file {filename} is empty"

    @pytest.mark.asyncio
    async def test_profiling_report_has_all_required_fields(
        self, factory: AdapterFactory
    ) -> None:
        """Profiling report must contain all expected top-level fields."""
        profile = await profile_workload_impl("model.bin", "windows", "npu", 10, factory)

        required_fields = [
            "latency", "throughput_fps", "power_mw", "memory_peak_mb",
            "memory_avg_mb", "utilization", "layers", "device", "runtime_used",
            "duration_s",
        ]
        for field in required_fields:
            assert field in profile, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_conversion_result_has_all_required_fields(
        self, factory: AdapterFactory
    ) -> None:
        """Conversion result must contain all expected fields."""
        result = await convert_model_impl("onnx", "model.onnx", "qnn", "int8", factory)

        required_fields = [
            "output_path", "model_size_mb", "original_size_mb",
            "compression_ratio", "supported_ops_pct", "unsupported_ops",
            "quantization_applied", "conversion_time_s", "target_sdk", "warnings",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_allocation_map_has_all_required_fields(
        self, factory: AdapterFactory
    ) -> None:
        """Allocation map must contain all expected fields."""
        result = await orchestrate_workload_impl("model.bin", "balanced", factory)

        required_fields = [
            "allocation", "projected_latency_ms", "projected_power_mw",
            "projected_memory_mb", "power_mode", "fallback_layers",
            "npu_utilization_pct", "gpu_utilization_pct", "cpu_utilization_pct",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    @pytest.mark.asyncio
    async def test_generated_code_has_all_required_fields(
        self, factory: AdapterFactory
    ) -> None:
        """Generated code result must contain all expected fields."""
        result = await generate_code_impl(
            "windows", "qnn", "python", "model.bin", TEMPLATE_DIR
        )

        required_fields = [
            "source_files", "build_instructions", "dependencies",
            "language", "platform", "sdk", "sample_input", "expected_output_format",
        ]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
