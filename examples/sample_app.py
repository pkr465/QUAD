#!/usr/bin/env python3
"""
QUAD Mock Mode — Sample Application
=====================================
Demonstrates the full QUAD workflow: detect hardware → convert model →
profile workload → orchestrate across CPU/GPU/NPU → generate inference code.

Run from the project root:
    source .venv/bin/activate
    python examples/sample_app.py

No hardware, SDK, or API keys required — mock mode simulates everything.
"""

import asyncio
import json
from pathlib import Path

from quad.adapters.factory import AdapterFactory
from quad.models.config import ServerConfig
from quad.tools.convert_model import convert_model_impl
from quad.tools.generate_code import generate_code_impl
from quad.tools.hardware_detect import hardware_detect_impl
from quad.tools.orchestrate_workload import orchestrate_workload_impl
from quad.tools.profile_workload import profile_workload_impl

# ── Helpers ───────────────────────────────────────────────────────────────────

def header(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")

def section(title: str) -> None:
    print(f"\n── {title} ──")

# ── Main workflow ─────────────────────────────────────────────────────────────

async def run_workflow(
    model_path: str = "resnet50.onnx",
    platform: str = "windows",
    target_sdk: str = "snpe",
    quantization: str = "int8",
):
    """
    Full QUAD workflow for a given model and platform.

    Steps:
      1. Detect hardware (chipset, CPU/GPU/NPU specs)
      2. Convert model (ONNX → SNPE DLC with INT8 quantization)
      3. Profile with detailed timing (latency, power, memory)
      4. Profile with HTP linting (per-op cycle counts, bottleneck detection)
      5. Orchestrate workload (allocate layers across CPU/GPU/NPU)
      6. Generate inference code (C++ for Windows)
    """
    factory = AdapterFactory(ServerConfig())

    header(f"QUAD Workflow — {model_path} on {platform.upper()}")
    print(f"  Model:    {model_path}")
    print(f"  Platform: {platform}")
    print(f"  SDK:      {target_sdk.upper()}")
    print(f"  Quant:    {quantization}")

    # ── Step 1: Hardware Detection ────────────────────────────────────────────
    section("Step 1: Hardware Detection")
    device = await hardware_detect_impl(platform, factory)

    print(f"  Chipset:  {device['chipset']}")
    print(f"  CPU:      {device['cpu_cores']} × {device['cpu_arch']} @ {device['cpu_freq_ghz']} GHz")
    print(f"  GPU:      {device['gpu_model']} ({device['gpu_tflops']} TFLOPS)")
    print(f"  NPU:      {device['npu_model']} ({device['npu_tops']} TOPS)")
    print(f"  RAM:      {device['ram_gb']} GB")
    print(f"  Runtimes: {', '.join(device['available_runtimes'])}")

    # ── Step 2: Model Conversion ──────────────────────────────────────────────
    section("Step 2: Model Conversion")
    conversion = await convert_model_impl(
        source_format="onnx",
        model_path=model_path,
        target_sdk=target_sdk,
        quantization=quantization,
        factory=factory,
        input_layout="nchw",      # ResNet uses PyTorch NCHW layout
        channel_order="rgb",
    )

    print(f"  Output:         {conversion['output_path']}")
    print(f"  Original size:  {conversion['original_size_mb']:.1f} MB")
    print(f"  Converted size: {conversion['model_size_mb']:.1f} MB")
    print(f"  Compression:    {conversion['compression_ratio']:.1f}×")
    print(f"  Quantization:   {conversion['quantization_applied']}")
    print(f"  Supported ops:  {conversion['supported_ops_pct']:.0f}%")
    if conversion['unsupported_ops']:
        print(f"  Fallback ops:   {', '.join(conversion['unsupported_ops'])}")
    if conversion['image_format_notes']:
        print(f"\n  ℹ️  Image format note:")
        print(f"     {conversion['image_format_notes'][0]}")
    if conversion['conversion_notes']:
        print(f"\n  ℹ️  Conversion note:")
        print(f"     {conversion['conversion_notes'][0]}")

    converted_model = conversion['output_path']

    # ── Step 3: Standard Profiling (detailed) ─────────────────────────────────
    section("Step 3: Profiling — Detailed (microsecond timing)")
    profile = await profile_workload_impl(
        model_path=converted_model,
        platform=platform,
        runtime="npu",
        duration_s=10,
        factory=factory,
        profiling_level="detailed",
    )

    lat = profile['latency']
    print(f"  Latency:    mean={lat['mean_ms']:.2f}ms  p95={lat['p95_ms']:.2f}ms  p99={lat['p99_ms']:.2f}ms")
    print(f"  Throughput: {profile['throughput_fps']:.0f} FPS")
    print(f"  Power:      {profile['power_mw']:.0f} mW")
    print(f"  Memory:     peak={profile['memory_peak_mb']:.0f} MB  avg={profile['memory_avg_mb']:.0f} MB")
    util = profile['utilization']
    print(f"  Utilization: NPU={util.get('npu', 0):.0f}%  GPU={util.get('gpu', 0):.0f}%  CPU={util.get('cpu', 0):.0f}%")

    # ── Step 4: HTP Linting Profile (cycle counts) ────────────────────────────
    section("Step 4: Profiling — HTP Linting (cycle-based per-op analysis)")
    linting = await profile_workload_impl(
        model_path=converted_model,
        platform=platform,
        runtime="npu",
        duration_s=5,
        factory=factory,
        profiling_level="linting",
    )

    print(f"  Total cycles:  {linting['linting_total_cycles']:,}")
    print(f"  Bottlenecks:   {linting['linting_bottleneck_count']} ops with low HTP parallelism")

    if linting['linting_layers']:
        # Show top 3 ops by cycle count
        top_ops = sorted(linting['linting_layers'], key=lambda x: x['total_cycles'], reverse=True)[:3]
        print(f"\n  Top ops by cycle count:")
        for op in top_ops:
            bottleneck_tag = " ⚠️ BOTTLENECK" if op['is_bottleneck'] else ""
            print(f"    [{op['index']:2d}] {op['name'][:45]:<45} "
                  f"{op['total_cycles']:>10,} cycles  "
                  f"overlap={op['overlap_ratio']:.0%}{bottleneck_tag}")

    if linting['linting_optimization_hints']:
        print(f"\n  Optimization hints:")
        for hint in linting['linting_optimization_hints']:
            print(f"    → {hint}")

    # ── Step 5: Workload Orchestration ────────────────────────────────────────
    section("Step 5: Workload Orchestration — balanced power mode")
    allocation = await orchestrate_workload_impl(
        model_path=converted_model,
        power_mode="balanced",
        factory=factory,
    )

    print(f"  Projected latency: {allocation['projected_latency_ms']:.2f} ms")
    print(f"  Projected power:   {allocation['projected_power_mw']:.0f} mW")
    print(f"  NPU utilization:   {allocation['npu_utilization_pct']:.0f}%")
    print(f"  GPU utilization:   {allocation['gpu_utilization_pct']:.0f}%")
    print(f"  CPU utilization:   {allocation['cpu_utilization_pct']:.0f}%")
    if allocation['fallback_layers']:
        print(f"  CPU fallback ops:  {', '.join(allocation['fallback_layers'][:3])}")

    # Show per-mode comparison
    print(f"\n  Power mode comparison:")
    for mode in ("performance", "balanced", "efficiency"):
        result = await orchestrate_workload_impl(converted_model, mode, factory)
        print(f"    {mode:<12} → {result['projected_latency_ms']:.2f}ms  "
              f"{result['projected_power_mw']:.0f}mW  "
              f"NPU={result['npu_utilization_pct']:.0f}%")

    # ── Step 6: Code Generation ───────────────────────────────────────────────
    section("Step 6: Code Generation — C++ inference app")
    codegen = await generate_code_impl(
        platform=platform,
        sdk="qnn",
        language="cpp",
        model_path=converted_model,
        template_dir="templates",
    )

    print(f"  Generated files:")
    for filename, content in codegen['source_files'].items():
        lines = len(content.splitlines())
        print(f"    📄 {filename} ({lines} lines)")

    if codegen.get('build_instructions'):
        print(f"\n  Build instructions:")
        for line in codegen['build_instructions'].splitlines()[:5]:
            if line.strip():
                print(f"    {line}")

    # ── Summary ───────────────────────────────────────────────────────────────
    header("Summary")
    print(f"  ✅ Detected:   {device['chipset']}")
    print(f"  ✅ Converted:  {model_path} → {converted_model}  ({conversion['compression_ratio']:.1f}× smaller)")
    print(f"  ✅ Profiled:   {lat['mean_ms']:.2f}ms mean latency  |  {linting['linting_total_cycles']:,} cycles")
    print(f"  ✅ Allocated:  {allocation['npu_utilization_pct']:.0f}% NPU  |  {allocation['projected_latency_ms']:.2f}ms projected")
    print(f"  ✅ Generated:  {', '.join(codegen['source_files'].keys())}")
    print()


# ── Multi-model comparison ────────────────────────────────────────────────────

async def compare_models():
    """Compare multiple models across platforms in mock mode."""
    factory = AdapterFactory(ServerConfig())

    header("Multi-Model Comparison (Mock Mode)")

    models = [
        ("mobilenetv2.onnx", "windows", "snpe", "int8"),
        ("resnet50.onnx",    "windows", "snpe", "int8"),
        ("yolov8n.onnx",     "android", "snpe", "int8"),
    ]

    print(f"\n  {'Model':<22} {'Platform':<10} {'Latency':>10} {'Throughput':>12} {'Power':>10} {'NPU':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    for model_path, platform, sdk, quant in models:
        conversion = await convert_model_impl("onnx", model_path, sdk, quant, factory)
        profile = await profile_workload_impl(
            conversion['output_path'], platform, "npu", 5, factory
        )
        alloc = await orchestrate_workload_impl(conversion['output_path'], "balanced", factory)

        print(f"  {model_path:<22} {platform:<10} "
              f"{profile['latency']['mean_ms']:>8.2f}ms "
              f"{profile['throughput_fps']:>10.0f}fps "
              f"{profile['power_mw']:>8.0f}mW "
              f"{alloc['npu_utilization_pct']:>6.0f}%")

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "compare":
        asyncio.run(compare_models())
    elif mode == "android":
        asyncio.run(run_workflow(
            model_path="mobilenetv2.onnx",
            platform="android",
            target_sdk="snpe",
            quantization="int8",
        ))
    elif mode == "linux":
        asyncio.run(run_workflow(
            model_path="yolov8n.onnx",
            platform="linux",
            target_sdk="snpe",
            quantization="int8",
        ))
    else:
        # Default: ResNet-50 on Windows AI PC
        asyncio.run(run_workflow(
            model_path="resnet50.onnx",
            platform="windows",
            target_sdk="snpe",
            quantization="int8",
        ))
