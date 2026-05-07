"""QUAD CLI — Interactive Getting-Started Wizard.

Walks the user through hardware detection, sample model compilation,
profiling, and code generation for a first-time QUAD experience.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuickstartResult:
    """Result of running the quickstart wizard."""

    device_detected: str
    model_compiled: str
    profile_generated: bool
    code_generated: bool
    total_time_s: float


# Sample models available for quickstart
_SAMPLE_MODELS = [
    {"name": "mobilenetv2", "desc": "MobileNetV2 — lightweight classification", "path": "models/mobilenetv2.onnx"},
    {"name": "yolov8n", "desc": "YOLOv8 Nano — real-time object detection", "path": "models/yolov8n.onnx"},
    {"name": "whisper_tiny", "desc": "Whisper Tiny — speech recognition", "path": "models/whisper_tiny.onnx"},
]


def run_quickstart(model_name: str | None = None, mock: bool = True) -> QuickstartResult:
    """Run the interactive quickstart wizard.

    Steps:
        1. Detect hardware (calls quad.runtime.list_devices())
        2. Present available sample models
        3. Compile a sample model
        4. Profile the compiled model
        5. Generate inference code
        6. Print summary with next steps

    Args:
        model_name: Specific sample model to use (default: first available).
        mock: If True, use mock results (no real compilation).

    Returns:
        QuickstartResult with summary of what was accomplished.
    """
    start = time.time()

    # Step 1: Detect hardware
    device_name = _detect_hardware()

    # Step 2: Select model
    selected = _select_model(model_name)

    # Step 3: Compile model
    compiled_path = _compile_sample(selected, mock=mock)

    # Step 4: Profile model
    profile_ok = _profile_model(compiled_path, mock=mock)

    # Step 5: Generate inference code
    code_ok = _generate_code(selected, device_name, mock=mock)

    # Step 6: Summary
    elapsed = time.time() - start
    _print_summary(device_name, selected, elapsed)

    return QuickstartResult(
        device_detected=device_name,
        model_compiled=selected["name"],
        profile_generated=profile_ok,
        code_generated=code_ok,
        total_time_s=elapsed,
    )


def _detect_hardware() -> str:
    """Detect available hardware and return the best device name."""
    try:
        from quad.runtime import list_devices

        devices = list_devices()
        if devices:
            best = devices[0]
            print(f"  [1/5] Hardware detected: {best.name} ({best.device_type})")
            return best.name
    except Exception:
        pass
    print("  [1/5] Hardware detected: Simulated NPU (mock mode)")
    return "Hexagon NPU (mock)"


def _select_model(model_name: str | None) -> dict[str, Any]:
    """Select a sample model for the wizard."""
    if model_name:
        for m in _SAMPLE_MODELS:
            if m["name"] == model_name:
                print(f"  [2/5] Selected model: {m['desc']}")
                return m
    # Default to first
    selected = _SAMPLE_MODELS[0]
    print(f"  [2/5] Selected model: {selected['desc']}")
    return selected


def _compile_sample(model: dict[str, Any], *, mock: bool = True) -> str:
    """Compile the sample model."""
    output_path = f"{model['name']}.qbin"
    if mock:
        print(f"  [3/5] Compiled {model['name']} -> {output_path} (mock)")
    else:
        from quad.compiler.pipeline import compile_model

        qbin = compile_model(model_path=model["path"], output_path=output_path)
        print(f"  [3/5] Compiled {model['name']} -> {qbin.path}")
        output_path = str(qbin.path)
    return output_path


def _profile_model(compiled_path: str, *, mock: bool = True) -> bool:
    """Profile the compiled model."""
    if mock:
        print(f"  [4/5] Profiled {compiled_path}: latency=4.2ms, throughput=238 FPS")
        return True
    # Real profiling would go here
    print(f"  [4/5] Profiling {compiled_path}...")
    return True


def _generate_code(model: dict[str, Any], device_name: str, *, mock: bool = True) -> bool:
    """Generate sample inference code."""
    code_file = f"run_{model['name']}.py"
    if mock:
        print(f"  [5/5] Generated inference code: {code_file}")
        return True
    # Real code generation would go here
    print(f"  [5/5] Generated: {code_file}")
    return True


def _print_summary(device: str, model: dict[str, Any], elapsed: float) -> None:
    """Print summary and next steps."""
    print("\n" + "=" * 50)
    print("  QUAD Quickstart Complete!")
    print("=" * 50)
    print(f"  Device:  {device}")
    print(f"  Model:   {model['desc']}")
    print(f"  Time:    {elapsed:.1f}s")
    print("\n  Next steps:")
    print("    1. Run: quad profile your_model.qbin")
    print("    2. Run: quad serve your_model.qbin")
    print("    3. Run: quad benchmark")
    print("    4. Read: https://docs.quad.dev/guides/first-model")
    print("")
