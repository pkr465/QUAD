"""SNPE/QAIRT SDK Tools Reference — Platform availability matrix.

Based on SNPE Tools documentation (80-63442-10 Rev AH, Apr 13 2026).

The matrix captures which tools are available on each target OS.
Columns: Ubuntu, WSL_x86_64, Android_device, WSL_x86_64_Win, Windows_x86_64, Windows_on_Snapdragon

Notes:
  * WSL x86_64 binary is from Ubuntu (see Qualcomm NPUD Setup docs)
  * ARM64X Windows: only snpe-net-run and snpe-throughput-net-run supported
  ** Windows x86_64: requires python scripts AND executables from Windows x86_64 binary folder
  *** snpe-accuracy-debugger on Windows x86: CPU runtime only
  * PowerShell: must activate venv and run converters via:
      (venv-3.10) > python snpe-onnx-to-dlc <options>
  * TFLite via qairt-converter: NOT supported on Windows x86_64 or Windows on Snapdragon
    (TVM library dependency unavailable on Windows)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Platform / OS Enum
# ══════════════════════════════════════════════════════════════════════════════

class ToolPlatform(str, Enum):
    """Target OS/platform for tool execution."""
    UBUNTU = "ubuntu"                       # Linux Ubuntu (host)
    WSL_X86_64 = "wsl_x86_64"              # Windows WSL (Ubuntu binary)
    ANDROID_DEVICE = "android_device"       # Android on-device
    WSL_X86_64_WIN = "wsl_x86_64_win"      # Windows WSL x86_64 (Windows context)
    WINDOWS_X86_64 = "windows_x86_64"      # Windows x86_64 (standard PC)
    WINDOWS_SNAPDRAGON = "windows_snapdragon"  # Windows on Snapdragon ARM64


# ══════════════════════════════════════════════════════════════════════════════
# Tool Category
# ══════════════════════════════════════════════════════════════════════════════

class ToolCategory(str, Enum):
    MODEL_CONVERSION = "model_conversion"
    MODEL_PREPARATION = "model_preparation"
    EXECUTION = "execution"
    ANALYSIS = "analysis"
    EXPERIMENTAL = "experimental"


# ══════════════════════════════════════════════════════════════════════════════
# Tool Descriptor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolDescriptor:
    """Describes a single SNPE/QAIRT SDK tool and its platform support."""
    name: str
    category: ToolCategory
    description: str
    available_on: set[ToolPlatform] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    deprecated_by: Optional[str] = None   # e.g. "qairt-converter" for per-framework tools

    def is_available_on(self, platform: ToolPlatform) -> bool:
        return platform in self.available_on

    def is_available_on_windows(self) -> bool:
        return (
            ToolPlatform.WINDOWS_X86_64 in self.available_on
            or ToolPlatform.WINDOWS_SNAPDRAGON in self.available_on
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tool Registry
# ══════════════════════════════════════════════════════════════════════════════

_ALL_PLATFORMS = {
    ToolPlatform.UBUNTU,
    ToolPlatform.WSL_X86_64,
    ToolPlatform.ANDROID_DEVICE,
    ToolPlatform.WSL_X86_64_WIN,
    ToolPlatform.WINDOWS_X86_64,
    ToolPlatform.WINDOWS_SNAPDRAGON,
}

_LINUX_ONLY = {
    ToolPlatform.UBUNTU,
    ToolPlatform.WSL_X86_64,
    ToolPlatform.ANDROID_DEVICE,
    ToolPlatform.WSL_X86_64_WIN,
}

# ** = requires Python scripts + executables from Windows x86_64 binary folder
_LINUX_AND_WINDOWS_X86_STAR = _LINUX_ONLY | {ToolPlatform.WINDOWS_X86_64}

SDK_TOOLS: dict[str, ToolDescriptor] = {
    # ── Model Conversion ────────────────────────────────────────────────────
    "snpe-onnx-to-dlc": ToolDescriptor(
        name="snpe-onnx-to-dlc",
        category=ToolCategory.MODEL_CONVERSION,
        description="Converts ONNX model (.onnx) to DLC. Supports ONNX Opset up to 24.",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=[
            "Windows x86_64 and Snapdragon: requires python scripts + executables from Windows x86_64 binary folder",
            "PowerShell: run via: (venv-3.10) > python snpe-onnx-to-dlc <options>",
            "ONNX Opset 24 max",
        ],
    ),
    "snpe-pytorch-to-dlc": ToolDescriptor(
        name="snpe-pytorch-to-dlc",
        category=ToolCategory.MODEL_CONVERSION,
        description="Converts serialized PyTorch model (.pt/.pth) to DLC.",
        available_on=_LINUX_ONLY,
        notes=["Linux/Android only. Use qairt-converter for cross-platform PyTorch conversion."],
    ),
    "snpe-tensorflow-to-dlc": ToolDescriptor(
        name="snpe-tensorflow-to-dlc",
        category=ToolCategory.MODEL_CONVERSION,
        description="Converts TensorFlow model (frozen .pb, checkpoint, SavedModel) to DLC.",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=["** Windows: requires python scripts + executables from Windows x86_64 binary folder"],
    ),
    "snpe-tflite-to-dlc": ToolDescriptor(
        name="snpe-tflite-to-dlc",
        category=ToolCategory.MODEL_CONVERSION,
        description="Converts TFLite model (.tflite) to DLC.",
        available_on=_LINUX_ONLY,
        notes=["Linux/Android only. TFLite via qairt-converter not supported on Windows (TVM dependency)."],
    ),
    "qairt-converter": ToolDescriptor(
        name="qairt-converter",
        category=ToolCategory.MODEL_CONVERSION,
        description=(
            "Unified converter: auto-detects framework from extension "
            "(ONNX/TF/TFLite/PyTorch). ONNX Opset 24 max. Recommended for new projects."
        ),
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=[
            "** Windows: requires python scripts + executables from Windows x86_64 binary folder",
            "TFLite conversion NOT supported on Windows x86_64 or Windows on Snapdragon (TVM dependency)",
            "Auto-detects framework from source model file extension",
        ],
    ),
    # ── Model Preparation ───────────────────────────────────────────────────
    "snpe-dlc-graph-prepare": ToolDescriptor(
        name="snpe-dlc-graph-prepare",
        category=ToolCategory.MODEL_PREPARATION,
        description="Offline graph preparation for quantized DLCs on DSP/HTP runtimes. Generates init cache.",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=["** Windows: requires python scripts + executables from Windows x86_64 binary folder"],
    ),
    "snpe-dlc-quant": ToolDescriptor(
        name="snpe-dlc-quant",
        category=ToolCategory.MODEL_PREPARATION,
        description="Converts non-quantized DLC to 8/16-bit TF-style fixed-point quantized DLC.",
        available_on=_ALL_PLATFORMS,
        notes=["Supports CLE, per-channel, per-row quantization. Batch dim must be 1 in source DLC."],
    ),
    "snpe-dlc-quantize": ToolDescriptor(
        name="snpe-dlc-quantize",
        category=ToolCategory.MODEL_PREPARATION,
        description="Converts non-quantized DLC to quantized DLC with optional HTP cache packing.",
        available_on=_LINUX_ONLY,
        notes=[
            "Mandatory for HTA runtime",
            "Mandatory for DSP on Snapdragon 865",
            "Use --enable_htp for HTP offline cache generation",
        ],
    ),
    "snpe-udo-package-generator": ToolDescriptor(
        name="snpe-udo-package-generator",
        category=ToolCategory.MODEL_PREPARATION,
        description="Generates UDO (User-Defined Op) package from a config file.",
        available_on=_LINUX_ONLY,
    ),
    "qairt-quantizer": ToolDescriptor(
        name="qairt-quantizer",
        category=ToolCategory.MODEL_PREPARATION,
        description="Modern quantizer: converts DLC to quantized DLC with AIMET support, per-channel/per-row options.",
        available_on=_ALL_PLATFORMS,
        notes=["Supports AIMET quantizer (--use_aimet_quantizer)", "Supports calibration methods: min-max, sqnr, entropy, mse, percentile"],
    ),
    # ── Execution ────────────────────────────────────────────────────────────
    "snpe-net-run": ToolDescriptor(
        name="snpe-net-run",
        category=ToolCategory.EXECUTION,
        description="Loads and executes a DLC neural network. Outputs raw tensors to output directory.",
        available_on=_ALL_PLATFORMS,
        notes=["Supports auto-batching (pads incomplete batches with zeros)"],
    ),
    "snpe-parallel-run": ToolDescriptor(
        name="snpe-parallel-run",
        category=ToolCategory.EXECUTION,
        description="Multi-thread inference on the same network. Each thread can use a different runtime.",
        available_on=_ALL_PLATFORMS - {ToolPlatform.WINDOWS_X86_64},
        notes=["ARM64X Windows: only snpe-net-run and snpe-throughput-net-run supported"],
    ),
    "snpe-throughput-net-run": ToolDescriptor(
        name="snpe-throughput-net-run",
        category=ToolCategory.EXECUTION,
        description="Concurrent multi-instance SNPE for throughput benchmarking over a fixed duration.",
        available_on=_ALL_PLATFORMS,
        notes=["--duration is shared across all SNPE instances"],
    ),
    # ── Analysis ─────────────────────────────────────────────────────────────
    "snpe-diagview": ToolDescriptor(
        name="snpe-diagview",
        category=ToolCategory.ANALYSIS,
        description="Reads SNPEDiag_*.log files from snpe-net-run and outputs per-layer timing. Also exports chrometrace JSON for linting level.",
        available_on={ToolPlatform.UBUNTU, ToolPlatform.WSL_X86_64, ToolPlatform.ANDROID_DEVICE, ToolPlatform.WSL_X86_64_WIN},
        notes=["Timing is averaged over all inputs in the input_list", "--chrometrace only valid for linting profiling level"],
    ),
    "snpe-dlc-diff": ToolDescriptor(
        name="snpe-dlc-diff",
        category=ToolCategory.ANALYSIS,
        description="Compares two DLCs: unique layers, parameter diffs, dimension diffs, weight diffs, output tensor names.",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=["** Windows: requires python scripts + executables from Windows x86_64 binary folder"],
    ),
    "snpe-dlc-info": ToolDescriptor(
        name="snpe-dlc-info",
        category=ToolCategory.ANALYSIS,
        description="Outputs layer and model information from a DLC file. Optional memory and encoding details.",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=["** Windows: requires python scripts + executables from Windows x86_64 binary folder"],
    ),
    "snpe-dlc-viewer": ToolDescriptor(
        name="snpe-dlc-viewer",
        category=ToolCategory.ANALYSIS,
        description="Visualizes DLC network structure in a web browser (HTML with graph, tooltips, zoom).",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=["** Windows: requires python scripts + executables from Windows x86_64 binary folder",
               "Supported browsers: Chrome, Firefox, IE (Windows), Edge (Windows), Safari (Mac)"],
    ),
    "snpe-platform-validator": ToolDescriptor(
        name="snpe-platform-validator",
        category=ToolCategory.ANALYSIS,
        description="On-device tool to check CPU/GPU/DSP/AIP capabilities. Must be pushed and run on device.",
        available_on={ToolPlatform.UBUNTU, ToolPlatform.WSL_X86_64},
        notes=["Runs ON device (not host). Push binary + Stub.so + Skel.so for target DSP arch."],
    ),
    "snpe-platform-validator-py": ToolDescriptor(
        name="snpe-platform-validator-py",
        category=ToolCategory.ANALYSIS,
        description="Python wrapper for snpe-platform-validator. Manages ADB push/execute flow from host.",
        available_on={ToolPlatform.UBUNTU},
        notes=["Host-side Python script. Outputs to CSV in Output directory."],
    ),
    "snpe_bench.py": ToolDescriptor(
        name="snpe_bench.py",
        category=ToolCategory.ANALYSIS,
        description="Python benchmark script. Configures and runs snpe-net-run via JSON config, collects timing CSVs.",
        available_on={ToolPlatform.UBUNTU, ToolPlatform.WSL_X86_64},
        notes=["Results in HostResultsDir with latest_results symlink", "Timing in microseconds"],
    ),
    "qairt-dlc-diff": ToolDescriptor(
        name="qairt-dlc-diff",
        category=ToolCategory.ANALYSIS,
        description="QAIRT version of snpe-dlc-diff. Compares two DLCs for layer/param/dim/weight/output diffs.",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=["** Windows: requires python scripts + executables from Windows x86_64 binary folder"],
    ),
    "qairt-dlc-info": ToolDescriptor(
        name="qairt-dlc-info",
        category=ToolCategory.ANALYSIS,
        description="QAIRT version of snpe-dlc-info. Outputs layer information, memory, encoding details.",
        available_on=_LINUX_AND_WINDOWS_X86_STAR | {ToolPlatform.WINDOWS_SNAPDRAGON},
        notes=["** Windows: requires python scripts + executables from Windows x86_64 binary folder"],
    ),
}


def get_tools_for_platform(platform: ToolPlatform) -> list[ToolDescriptor]:
    """Return all tools available on a given platform."""
    return [t for t in SDK_TOOLS.values() if t.is_available_on(platform)]


def get_tools_by_category(category: ToolCategory) -> list[ToolDescriptor]:
    """Return all tools in a given category."""
    return [t for t in SDK_TOOLS.values() if t.category == category]


def is_tool_available(tool_name: str, platform: ToolPlatform) -> bool:
    """Check if a named tool is available on a given platform."""
    td = SDK_TOOLS.get(tool_name)
    return td is not None and td.is_available_on(platform)


# ══════════════════════════════════════════════════════════════════════════════
# Platform Notes
# ══════════════════════════════════════════════════════════════════════════════

PLATFORM_NOTES = {
    "wsl_x86_64": (
        "WSL x86_64 binary is from Ubuntu. "
        "Reference Qualcomm Neural Processing SDK Setup for installation."
    ),
    "windows_arm64x": (
        "For ARM64X in Windows, only snpe-net-run and snpe-throughput-net-run are supported."
    ),
    "windows_star": (
        "** Tools marked ** on Windows x86_64 / Snapdragon require both the Python scripts "
        "AND the executables from the Windows x86_64 binary folder."
    ),
    "powershell": (
        "When using converter tools in Windows PowerShell, activate a virtual environment "
        "with required packages and run converters via python:\n"
        "  (venv-3.10) > python snpe-onnx-to-dlc <options>"
    ),
    "tflite_windows": (
        "TFLite conversion using qairt-converter is NOT supported for Windows x86_64 "
        "or Windows on Snapdragon due to TVM library dependency."
    ),
}
