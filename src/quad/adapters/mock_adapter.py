"""Mock adapter — simulates all SDK operations without real hardware or SDKs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from quad.adapters.base import SDKAdapter
from quad.models.conversion import ConversionRequest, ConversionResult
from quad.models.device import DeviceProfile
from quad.models.profiling import (
    LatencyStats,
    LayerProfile,
    ProfileRequest,
    ProfilingReport,
)

# Pre-defined device profiles for mock mode
_DEVICE_PROFILES: dict[str, dict[str, Any]] = {
    "windows": {
        "chipset": "Snapdragon X Elite X1E-80-100",
        "platform": "windows",
        "cpu_cores": 12,
        "cpu_arch": "Oryon ARM64",
        "cpu_freq_ghz": 3.8,
        "gpu_model": "Adreno X1-85",
        "gpu_tflops": 4.6,
        "npu_model": "Hexagon NPU",
        "npu_tops": 45.0,
        "ram_gb": 32.0,
        "sdk_path": "C:/Qualcomm/AIStack/QNN",
        "sdk_version": "2.28.0",
        "available_runtimes": ["cpu", "gpu", "npu"],
    },
    "linux": {
        "chipset": "QCS2210 (Qualcomm Robotics RB1)",
        "platform": "linux",
        "cpu_cores": 4,
        "cpu_arch": "Kryo ARM64",
        "cpu_freq_ghz": 2.0,
        "gpu_model": "Adreno 504",
        "gpu_tflops": 0.4,
        "npu_model": "Hexagon DSP V66",
        "npu_tops": 1.0,
        "ram_gb": 2.0,
        "sdk_path": "/opt/snpe-2.x",
        "sdk_version": "2.22.0",
        "available_runtimes": ["cpu", "gpu", "npu"],
    },
    "android": {
        "chipset": "Snapdragon 8 Elite SM8750",
        "platform": "android",
        "cpu_cores": 8,
        "cpu_arch": "Oryon ARM64",
        "cpu_freq_ghz": 4.32,
        "gpu_model": "Adreno 830",
        "gpu_tflops": 5.0,
        "npu_model": "Hexagon NPU (HTP)",
        "npu_tops": 48.0,
        "ram_gb": 16.0,
        "sdk_path": "/data/local/tmp/snpe",
        "sdk_version": "2.28.0",
        "available_runtimes": ["cpu", "gpu", "npu"],
    },
}

# Simulated supported ops for mock
_MOCK_SUPPORTED_OPS = [
    "Conv", "ConvTranspose", "BatchNormalization", "Relu", "LeakyRelu",
    "Sigmoid", "Tanh", "MaxPool", "AveragePool", "GlobalAveragePool",
    "Gemm", "MatMul", "Add", "Mul", "Sub", "Div", "Concat", "Reshape",
    "Transpose", "Flatten", "Softmax", "Squeeze", "Unsqueeze",
    "Clip", "Pad", "Resize", "ReduceMean", "Gather", "Slice",
]

# Ops that typically require CPU fallback
_MOCK_UNSUPPORTED_OPS = [
    "NonMaxSuppression", "TopK", "Loop", "If", "Scan", "CenterCropPad",
    "StringNormalizer", "TfIdfVectorizer",
]


class MockAdapter(SDKAdapter):
    """Simulates SDK operations with realistic responses.

    Uses model metadata (file size, name) to generate plausible
    conversion metrics, profiling data, and allocation maps.
    """

    def __init__(self, sdk_name: str = "mock"):
        self.sdk_name = sdk_name

    async def detect_hardware(self, platform: str) -> DeviceProfile:
        """Return pre-defined device profile for the platform."""
        profile_data = _DEVICE_PROFILES.get(platform)
        if profile_data is None:
            # Default to windows if unknown
            profile_data = _DEVICE_PROFILES["windows"]
        return DeviceProfile(**profile_data)

    async def convert_model(self, request: ConversionRequest) -> ConversionResult:
        """Simulate model conversion based on file metadata."""
        model_path = Path(request.model_path)

        # Estimate original size (use actual file if exists, else estimate)
        if model_path.exists():
            original_size_mb = model_path.stat().st_size / (1024 * 1024)
        else:
            # Estimate based on common model sizes
            original_size_mb = 25.0  # Default estimate

        # Calculate compressed size based on quantization
        compression_factors = {"fp32": 1.0, "int8": 0.25, "int4": 0.125}
        factor = compression_factors.get(request.quantization, 1.0)
        model_size_mb = original_size_mb * factor

        # Simulate conversion time (proportional to model size)
        conversion_time_s = original_size_mb * 0.1  # ~100ms per MB

        # Determine output path
        ext = ".bin" if request.target_sdk == "qnn" else ".dlc"
        output_path = str(model_path.with_suffix(ext))

        return ConversionResult(
            output_path=output_path,
            model_size_mb=round(model_size_mb, 2),
            original_size_mb=round(original_size_mb, 2),
            compression_ratio=round(1.0 / factor, 1),
            supported_ops_pct=96.5,
            unsupported_ops=["NonMaxSuppression"] if "yolo" in str(model_path).lower() else [],
            quantization_applied=request.quantization,
            conversion_time_s=round(conversion_time_s, 2),
            target_sdk=request.target_sdk,
            warnings=[],
            conversion_notes=_get_mock_model_tips(str(model_path)),
            image_format_notes=_get_mock_image_format_notes(request),
        )

    async def profile(self, request: ProfileRequest) -> ProfilingReport:
        """Generate realistic profiling metrics from model metadata."""
        device = await self.detect_hardware(request.platform)

        # Base latency depends on runtime
        runtime_latency_factors = {"cpu": 8.0, "gpu": 3.0, "npu": 1.0, "auto": 1.0}
        base_latency = runtime_latency_factors.get(request.runtime, 1.0)

        mean_latency = base_latency * 5.0  # ~5ms on NPU for typical model

        # Power depends on runtime
        runtime_power_factors = {"cpu": 5000.0, "gpu": 3000.0, "npu": 2000.0, "auto": 2000.0}
        power_mw = runtime_power_factors.get(request.runtime, 2000.0)

        # Generate mock layer profiles
        layer_names = [
            "conv1", "bn1", "relu1", "conv2", "bn2", "relu2",
            "conv3", "pool1", "fc1", "softmax",
        ]
        runtime_used = request.runtime if request.runtime != "auto" else "npu"
        layers = [
            LayerProfile(
                name=name,
                op_type=name.split("_")[0].rstrip("0123456789"),
                runtime=runtime_used,
                latency_ms=round(mean_latency / len(layer_names), 3),
                memory_mb=round(2.0 + i * 0.5, 2),
            )
            for i, name in enumerate(layer_names)
        ]

        profiling_level = getattr(request, "profiling_level", "detailed")

        # For linting/qhas in mock mode, return simulated cycle-based data
        linting_layers = []
        linting_total_cycles = 0
        linting_bottleneck_count = 0
        linting_hints: list[str] = []

        if profiling_level in ("linting", "qhas"):
            from quad.models.profiling import LintingLayerProfile
            # Simulate a realistic linting profile (sub op as bottleneck)
            mock_ops = [
                ("Input", 0, 0, 0, 0, 0),
                ("conv_start", 147075, 32, 85292, 32, 0),
                ("conv_left1", 288249, 425, 195988, 304, 0),
                ("conv_right1", 220391, 803, 135268, 557, 0),
                ("sub_op", 2165162, 0, 465046, 0, 1),  # bottleneck
                ("add_op", 525971, 0, 481468, 0, 0),
                ("output", 407091, 0, 115120, 0, 0),
            ]
            total = sum(op[1] for op in mock_ops)
            linting_total_cycles = total
            for name, cyc, wait, overlap, ov_wait, is_bn in mock_ops:
                frac = cyc / total if total > 0 else 0.0
                ratio = overlap / cyc if cyc > 0 else 0.0
                hint = "Consider replacing Sub op with Conv for better HTP utilization." if is_bn else None
                linting_layers.append(LintingLayerProfile(
                    name=name, index=len(linting_layers),
                    total_cycles=cyc, wait_cycles=wait,
                    overlap_cycles=overlap, overlap_wait_cycles=ov_wait,
                    overlap_ratio=round(ratio, 4), cycle_fraction=round(frac, 4),
                    resources=["HVX"] if name != "conv_start" else ["HVX", "HMX", "DMA"],
                    is_bottleneck=bool(is_bn), optimization_hint=hint,
                ))
            linting_bottleneck_count = sum(1 for op in mock_ops if op[5])
            linting_hints = ["Replace Sub op with Conv for better HTP parallelism."]

        return ProfilingReport(
            latency=LatencyStats(
                mean_ms=round(mean_latency, 3),
                p50_ms=round(mean_latency * 0.95, 3),
                p95_ms=round(mean_latency * 1.4, 3),
                p99_ms=round(mean_latency * 1.8, 3),
                min_ms=round(mean_latency * 0.8, 3),
                max_ms=round(mean_latency * 2.2, 3),
            ),
            throughput_fps=round(1000.0 / mean_latency, 1),
            power_mw=power_mw,
            memory_peak_mb=45.0,
            memory_avg_mb=38.0,
            utilization={"cpu": 15.0, "gpu": 5.0, "npu": 89.0}
            if runtime_used == "npu"
            else {"cpu": 85.0, "gpu": 10.0, "npu": 0.0},
            layers=layers,
            device=device,
            runtime_used=runtime_used,
            duration_s=float(request.duration_s),
            profiling_level=profiling_level,
            linting_layers=linting_layers,
            linting_total_cycles=linting_total_cycles,
            linting_bottleneck_count=linting_bottleneck_count,
            linting_optimization_hints=linting_hints,
            qhas_chrometrace_path="./chrometrace.json" if profiling_level == "qhas" else None,
        )

    async def get_supported_ops(self) -> list[str]:
        """Return mock supported ops list."""
        return list(_MOCK_SUPPORTED_OPS)

    async def execute_inference(self, model_path: str, input_data: Any) -> Any:
        """Simulate inference execution."""
        return {
            "status": "success",
            "output_shape": [1, 1000],
            "inference_time_ms": 5.2,
            "model_path": model_path,
        }


def _get_mock_model_tips(model_path: str) -> list[str]:
    """Return MODEL_TIPS notes for known model families (mock mode)."""
    try:
        from quad.compiler.model_conversion import MODEL_TIPS
    except ImportError:
        return []
    notes = []
    path_lower = model_path.lower()
    for model_key, tips in MODEL_TIPS.items():
        if model_key in path_lower:
            notes.extend(tips.get("limitations", [])[:3])
            notes.extend(tips.get("performance_tips", [])[:2])
    return notes


def _get_mock_image_format_notes(request: "ConversionRequest") -> list[str]:  # type: ignore[name-defined]
    """Build image format guidance notes (mock mode mirrors real adapter logic)."""
    notes = []
    layout = getattr(request, "input_layout", "auto")
    channel = getattr(request, "channel_order", "auto")
    means = getattr(request, "mean_values", None)

    if layout == "nchw":
        notes.append(
            "Input layout is NCHW (PyTorch). "
            "SNPE requires NHWC — transpose before inference: "
            "np.transpose(img, (0, 2, 3, 1))  # (N,C,H,W) → (N,H,W,C)"
        )
    if channel == "bgr":
        notes.append(
            "Channel order is BGR. "
            "Ensure inference inputs are in BGR order."
        )
    if means:
        notes.append(f"Mean subtraction required: mean={means}.")
    if not notes:
        notes.append(
            "SNPE requires NHWC inputs. Verify channel order matches training (RGB vs BGR)."
        )
    return notes
