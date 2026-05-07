#!/usr/bin/env python3
"""QUAD Real-Hardware Sample Application — Snapdragon X Elite

Runs the full QUAD pipeline (the 5 MCP tools) on the **physical** machine
QUAD is currently installed on, and pairs every QUAD output with a
ground-truth measurement taken from the live system:

  * Hardware detection — cross-checks QUAD's profile against PowerShell
    Win32_Processor / Win32_VideoController / Get-PnpDevice queries
  * Conversion — runs the QUAD conversion (mock, since QAIRT SDK is not
    installed) and reports both the projected DLC size and the input
    ONNX size taken from the local file
  * Profiling — runs **real ONNX Runtime CPU inference** on
    MobileNetV2-1.0 (224×224×3) for `--iterations` runs on the Oryon CPU
    and reports the measured latency / throughput / memory; also prints
    the QUAD NPU projection so the developer can see the expected gain
    once QAIRT SDK is wired
  * Orchestration — QUAD allocates the model across CPU/GPU/NPU
  * Code generation — emits the Windows-on-Snapdragon C++ inference
    code that would drive QNN HTP

Power: read from the Windows battery API at the start and end of the
inference run; we report the percentage delta + an estimated draw using
the laptop's nominal battery capacity.

Run from the project root:
    python examples/sample_app_real_hw.py            # default: 200 iterations
    python examples/sample_app_real_hw.py --iterations 500 --warmup 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil

from quad.adapters.factory import AdapterFactory
from quad.models.config import ServerConfig
from quad.tools.convert_model import convert_model_impl
from quad.tools.generate_code import generate_code_impl
from quad.tools.hardware_detect import hardware_detect_impl
from quad.tools.orchestrate_workload import orchestrate_workload_impl
from quad.tools.profile_workload import profile_workload_impl

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "examples" / "models" / "mobilenetv2-12.onnx"


# ─── Helpers ──────────────────────────────────────────────────────────────────


def header(title: str) -> None:
    bar = "═" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def section(title: str) -> None:
    print(f"\n── {title} ──")


def kv(label: str, value: Any, *, width: int = 22) -> None:
    print(f"  {label:<{width}} {value}")


# ─── Real hardware probes (PowerShell + psutil) ───────────────────────────────


def _powershell(cmd: str) -> str:
    """Run a PowerShell command, return stripped stdout."""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


@dataclass
class HostHardware:
    """Ground-truth hardware probe of the local machine."""

    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    cpu_max_mhz: int
    gpu_name: str
    gpu_driver: str
    npu_name: str
    npu_present: bool
    ram_gb: float
    os_name: str
    os_arch: str
    on_battery: bool
    battery_pct: int


def probe_host_hardware() -> HostHardware:
    """Read true hardware data via Windows APIs, no mocking."""

    cpu_query = (
        "$c = Get-CimInstance Win32_Processor; "
        "Write-Output $c.Name; "
        "Write-Output $c.NumberOfCores; "
        "Write-Output $c.NumberOfLogicalProcessors; "
        "Write-Output $c.MaxClockSpeed"
    )
    cpu_lines = _powershell(cpu_query).splitlines()
    cpu_name, cpu_cores, cpu_threads, cpu_max_mhz = (
        cpu_lines[0],
        int(cpu_lines[1]),
        int(cpu_lines[2]),
        int(cpu_lines[3]),
    )

    gpu_query = (
        "$g = Get-CimInstance Win32_VideoController | Select-Object -First 1; "
        "Write-Output $g.Name; "
        "Write-Output $g.DriverVersion"
    )
    gpu_lines = _powershell(gpu_query).splitlines()
    gpu_name, gpu_driver = gpu_lines[0], gpu_lines[1] if len(gpu_lines) > 1 else "n/a"

    npu_query = (
        "$d = Get-PnpDevice -Status OK | Where-Object { "
        "$_.Class -eq 'ComputeAccelerator' -and "
        "$_.FriendlyName -match 'Hexagon|NPU|Snapdragon' } | Select-Object -First 1; "
        "if ($d) { Write-Output $d.FriendlyName } else { Write-Output 'none' }"
    )
    npu_name = _powershell(npu_query).strip() or "none"

    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    os_query = (
        "Write-Output (Get-CimInstance Win32_OperatingSystem).Caption; "
        "Write-Output ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)"
    )
    os_lines = _powershell(os_query).splitlines()
    os_name = os_lines[0] if os_lines else platform.platform()
    os_arch = os_lines[1] if len(os_lines) > 1 else platform.machine()

    bat_query = (
        "$b = Get-CimInstance Win32_Battery | Select-Object -First 1; "
        "if ($b) { Write-Output $b.EstimatedChargeRemaining; "
        "Write-Output $b.BatteryStatus } else { Write-Output 100; Write-Output 2 }"
    )
    bat_lines = _powershell(bat_query).splitlines()
    battery_pct = int(bat_lines[0]) if bat_lines else 100
    # BatteryStatus: 1 = discharging, 2 = on AC, 3+ = charging/charged
    on_battery = (int(bat_lines[1]) == 1) if len(bat_lines) > 1 else False

    return HostHardware(
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        cpu_max_mhz=cpu_max_mhz,
        gpu_name=gpu_name,
        gpu_driver=gpu_driver,
        npu_name=npu_name,
        npu_present=npu_name != "none",
        ram_gb=ram_gb,
        os_name=os_name,
        os_arch=os_arch,
        on_battery=on_battery,
        battery_pct=battery_pct,
    )


# ─── Real CPU inference benchmark ─────────────────────────────────────────────


@dataclass
class CpuBenchmarkResult:
    """Real ONNX Runtime CPU inference measurements."""

    runtime: str
    iterations: int
    warmup: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    throughput_fps: float
    memory_peak_mb: float
    memory_baseline_mb: float
    cpu_pct_during_run: float
    battery_pct_start: int
    battery_pct_end: int
    # MobileNetV2 has ~3.4 MMACs (multiply-accumulate) per inference
    mmacs_per_inference: float = 300.0  # MobileNetV2-1.0 ~= 300M MACs
    elapsed_total_s: float = 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def benchmark_cpu(
    model_path: Path,
    iterations: int = 200,
    warmup: int = 30,
) -> CpuBenchmarkResult:
    """Run real ONNX Runtime inference on the Oryon CPU and measure."""
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 0  # let ORT pick (uses all 12 Oryon cores)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    in_meta = sess.get_inputs()[0]
    in_name = in_meta.name
    in_shape = [1 if (isinstance(d, str) or d is None) else d for d in in_meta.shape]

    # Realistic input: random NCHW float32 image
    rng = np.random.default_rng(42)
    inp = rng.standard_normal(in_shape, dtype=np.float32)
    feed = {in_name: inp}

    proc = psutil.Process()
    baseline_mb = proc.memory_info().rss / (1024 * 1024)

    # Warmup (excluded from timing — first runs allocate, JIT, populate caches)
    for _ in range(warmup):
        sess.run(None, feed)

    # Probe battery before the timed run
    bat_query = (
        "$b = Get-CimInstance Win32_Battery | Select-Object -First 1; "
        "if ($b) { Write-Output $b.EstimatedChargeRemaining } else { Write-Output 100 }"
    )
    bat_start = int(_powershell(bat_query) or "100")

    proc.cpu_percent(interval=None)  # prime psutil's CPU sampler
    peak_mb = baseline_mb
    latencies: list[float] = []

    t0 = time.perf_counter()
    for _ in range(iterations):
        s = time.perf_counter()
        sess.run(None, feed)
        latencies.append((time.perf_counter() - s) * 1000.0)
        # Sample memory occasionally (cheap)
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        peak_mb = max(peak_mb, rss_mb)
    elapsed_total = time.perf_counter() - t0

    cpu_pct = proc.cpu_percent(interval=None)
    bat_end = int(_powershell(bat_query) or str(bat_start))

    return CpuBenchmarkResult(
        runtime="ONNX Runtime CPUExecutionProvider (Oryon)",
        iterations=iterations,
        warmup=warmup,
        mean_ms=mean(latencies),
        median_ms=median(latencies),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        min_ms=min(latencies),
        max_ms=max(latencies),
        stddev_ms=stdev(latencies) if len(latencies) > 1 else 0.0,
        throughput_fps=iterations / elapsed_total,
        memory_peak_mb=round(peak_mb, 1),
        memory_baseline_mb=round(baseline_mb, 1),
        cpu_pct_during_run=round(cpu_pct, 1),
        battery_pct_start=bat_start,
        battery_pct_end=bat_end,
        elapsed_total_s=round(elapsed_total, 3),
    )


# ─── Main pipeline ────────────────────────────────────────────────────────────


async def run(iterations: int, warmup: int, verbose: bool) -> dict[str, Any]:
    factory = AdapterFactory(ServerConfig())

    header("QUAD Sample App — Real Hardware Run (Snapdragon X Elite)")

    # ── 0. Real hardware probe ────────────────────────────────────────────────
    section("0. Host hardware (real, probed via Windows APIs)")
    host = probe_host_hardware()
    kv("OS:", f"{host.os_name} ({host.os_arch})")
    kv("CPU:", f"{host.cpu_name}")
    kv("Cores / threads:", f"{host.cpu_cores} cores / {host.cpu_threads} threads")
    kv("Max clock:", f"{host.cpu_max_mhz} MHz")
    kv("GPU:", f"{host.gpu_name} (driver {host.gpu_driver})")
    kv(
        "NPU:",
        f"{host.npu_name}  [{'PRESENT' if host.npu_present else 'not present'}]",
    )
    kv("RAM:", f"{host.ram_gb} GB")
    kv(
        "Power source:",
        f"{'battery' if host.on_battery else 'AC'} ({host.battery_pct}%)",
    )

    # ── 1. QUAD hardware_detect ───────────────────────────────────────────────
    section("1. QUAD hardware_detect (MCP tool)")
    qdev = await hardware_detect_impl("windows", factory)
    kv("Chipset:", qdev["chipset"])
    kv(
        "QUAD CPU:",
        f"{qdev['cpu_cores']} × {qdev['cpu_arch']} @ {qdev['cpu_freq_ghz']} GHz",
    )
    kv("QUAD GPU:", f"{qdev['gpu_model']} ({qdev['gpu_tflops']} TFLOPS)")
    kv("QUAD NPU:", f"{qdev['npu_model']} ({qdev['npu_tops']} TOPS)")
    kv("QUAD RAM:", f"{qdev['ram_gb']} GB")
    kv("Available runtimes:", ", ".join(qdev["available_runtimes"]))

    # ── 2. Conversion (mock — QAIRT SDK not installed) ────────────────────────
    section("2. QUAD convert_model (ONNX → SNPE DLC, INT8)")
    conv = await convert_model_impl(
        source_format="onnx",
        model_path=str(MODEL_PATH),
        target_sdk="snpe",
        quantization="int8",
        factory=factory,
        input_layout="nchw",
        channel_order="rgb",
    )
    kv("Source ONNX:", f"{MODEL_PATH.name} ({MODEL_PATH.stat().st_size/1024/1024:.1f} MB, real)")
    kv("Projected DLC:", f"{conv['model_size_mb']:.1f} MB ({conv['compression_ratio']:.1f}× smaller)")
    kv("Quantization:", conv["quantization_applied"])
    kv("Conversion time:", f"{conv['conversion_time_s']:.2f} s (projected)")
    if conv.get("conversion_notes"):
        kv("Notes:", conv["conversion_notes"][0])

    # ── 3. Profile — REAL CPU + projected NPU side-by-side ───────────────────
    section(f"3a. REAL CPU benchmark on Oryon ({iterations} iters, {warmup} warmup)")
    print("    Running real ONNX Runtime CPU inference on MobileNetV2-1.0 …")
    cpu = benchmark_cpu(MODEL_PATH, iterations=iterations, warmup=warmup)
    kv("Runtime:", cpu.runtime)
    kv("Iterations:", f"{cpu.iterations} ({cpu.warmup} warmup)")
    kv("Wall time:", f"{cpu.elapsed_total_s} s")
    kv(
        "Latency (ms):",
        f"mean={cpu.mean_ms:.2f}  median={cpu.median_ms:.2f}  "
        f"p95={cpu.p95_ms:.2f}  p99={cpu.p99_ms:.2f}  "
        f"min={cpu.min_ms:.2f}  max={cpu.max_ms:.2f}  "
        f"σ={cpu.stddev_ms:.2f}",
    )
    kv("Throughput:", f"{cpu.throughput_fps:.1f} FPS")
    kv(
        "Memory:",
        f"baseline={cpu.memory_baseline_mb} MB  peak={cpu.memory_peak_mb} MB  "
        f"Δ={cpu.memory_peak_mb - cpu.memory_baseline_mb:.1f} MB",
    )
    kv("CPU% during run:", f"{cpu.cpu_pct_during_run}%")
    bat_delta = cpu.battery_pct_start - cpu.battery_pct_end
    kv(
        "Battery delta:",
        f"{cpu.battery_pct_start}% → {cpu.battery_pct_end}% ({bat_delta:+d} pp)",
    )

    section("3b. QUAD NPU projection (HTP — projected, requires QAIRT SDK to measure)")
    npu_profile = await profile_workload_impl(
        model_path=str(MODEL_PATH),
        platform="windows",
        runtime="npu",
        duration_s=10,
        factory=factory,
        profiling_level="detailed",
    )
    nlat = npu_profile["latency"]
    kv("NPU runtime:", "Hexagon HTP (Snapdragon X Elite)")
    kv(
        "NPU latency (proj.):",
        f"mean={nlat['mean_ms']:.2f}ms  p95={nlat['p95_ms']:.2f}ms  "
        f"p99={nlat['p99_ms']:.2f}ms",
    )
    kv("NPU throughput:", f"{npu_profile['throughput_fps']} FPS")
    kv("NPU power:", f"{npu_profile['power_mw']} mW")
    kv("NPU memory:", f"peak={npu_profile['memory_peak_mb']} MB")
    kv("NPU utilization:", f"{npu_profile['utilization'].get('npu', 0)}%")
    proj_speedup = cpu.mean_ms / max(nlat["mean_ms"], 1e-6)
    kv("Projected speedup:", f"{proj_speedup:.1f}× faster than CPU (NPU vs Oryon)")

    # ── 4. Linting profile (per-op cycle counts) ─────────────────────────────
    section("4. QUAD profile_workload — HTP linting (cycle-level)")
    lint = await profile_workload_impl(
        model_path=str(MODEL_PATH),
        platform="windows",
        runtime="npu",
        duration_s=5,
        factory=factory,
        profiling_level="linting",
    )
    kv("Total cycles:", f"{lint['linting_total_cycles']:,}")
    kv("Bottleneck ops:", lint["linting_bottleneck_count"])
    if lint["linting_layers"]:
        top = sorted(
            lint["linting_layers"], key=lambda x: x["total_cycles"], reverse=True
        )[:3]
        print("    Top 3 ops by cycle count:")
        for op in top:
            tag = " ⚠ BOTTLENECK" if op["is_bottleneck"] else ""
            print(
                f"      [{op['index']:2d}] {op['name'][:40]:<40} "
                f"{op['total_cycles']:>10,} cycles  "
                f"overlap={op['overlap_ratio']:.0%}{tag}"
            )

    # ── 5. Orchestration ──────────────────────────────────────────────────────
    section("5. QUAD orchestrate_workload — power-mode comparison")
    print(
        f"    {'Mode':<14} {'Latency':>10} {'Power':>10} {'NPU%':>6} {'GPU%':>6} {'CPU%':>6}"
    )
    for mode in ("performance", "balanced", "efficiency"):
        a = await orchestrate_workload_impl(
            model_path=str(MODEL_PATH), power_mode=mode, factory=factory
        )
        print(
            f"    {mode:<14} "
            f"{a['projected_latency_ms']:>8.2f}ms "
            f"{a['projected_power_mw']:>8.0f}mW "
            f"{a['npu_utilization_pct']:>5.0f}% "
            f"{a['gpu_utilization_pct']:>5.0f}% "
            f"{a['cpu_utilization_pct']:>5.0f}%"
        )
    balanced = await orchestrate_workload_impl(str(MODEL_PATH), "balanced", factory)
    if balanced.get("fallback_layers"):
        kv("CPU fallback ops:", ", ".join(balanced["fallback_layers"][:3]))

    # ── 6. Code generation ────────────────────────────────────────────────────
    section("6. QUAD generate_code — Windows-on-Snapdragon C++ inference app")
    gen_dir = REPO_ROOT / "examples" / "generated" / "real_hw_demo"
    gen_dir.mkdir(parents=True, exist_ok=True)
    code = await generate_code_impl(
        platform="windows",
        sdk="qnn",
        language="cpp",
        model_path=str(MODEL_PATH),
        template_dir=str(REPO_ROOT / "templates"),
    )
    files_written = []
    for fname, content in code["source_files"].items():
        out = gen_dir / fname
        out.write_text(content)
        files_written.append((fname, len(content.splitlines()), out.stat().st_size))
    print("    Files written:")
    for fname, lines, size in files_written:
        print(f"      📄 {gen_dir.name}/{fname}  ({lines} lines, {size} bytes)")

    # ── Summary block ─────────────────────────────────────────────────────────
    header("Summary — Real Hardware Run")
    print(f"  Hardware:        Snapdragon X Elite X1E80100 (Oryon, Adreno X1-85, Hexagon NPU)")
    print(f"  Model:           MobileNetV2-1.0 ({MODEL_PATH.stat().st_size/1024/1024:.1f} MB)")
    print(f"  CPU mean lat.:   {cpu.mean_ms:.2f} ms  ({cpu.throughput_fps:.1f} FPS)  ← REAL on Oryon")
    print(f"  NPU proj. lat.:  {nlat['mean_ms']:.2f} ms  ({npu_profile['throughput_fps']} FPS)  ← QUAD projection")
    print(f"  Projected gain:  {proj_speedup:.1f}× faster on NPU")
    print(f"  Memory peak:     {cpu.memory_peak_mb} MB (real, CPU run)")
    print(f"  Power baseline:  battery {cpu.battery_pct_start}% → {cpu.battery_pct_end}% over {cpu.elapsed_total_s}s")
    print(f"  Generated code:  {gen_dir}")
    print()

    # Bundle all data for downstream report generation
    return {
        "host": asdict(host),
        "quad_device": qdev,
        "conversion": conv,
        "cpu_benchmark": asdict(cpu),
        "npu_projection": npu_profile,
        "linting": {
            "total_cycles": lint["linting_total_cycles"],
            "bottleneck_count": lint["linting_bottleneck_count"],
            "top_ops": sorted(
                lint["linting_layers"], key=lambda x: x["total_cycles"], reverse=True
            )[:5],
        },
        "orchestration": {
            mode: await orchestrate_workload_impl(str(MODEL_PATH), mode, factory)
            for mode in ("performance", "balanced", "efficiency")
        },
        "generated_files": [str(gen_dir / f) for f, _, _ in files_written],
        "model_path": str(MODEL_PATH),
        "model_size_mb": round(MODEL_PATH.stat().st_size / (1024 * 1024), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=200, help="CPU benchmark iterations (default: 200)")
    p.add_argument("--warmup", type=int, default=30, help="Warmup iterations (excluded from timing)")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    p.add_argument("--json-out", type=Path, help="Write run results as JSON")
    args = p.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
        print(f"Run: python -c \"import urllib.request; urllib.request.urlretrieve("
              f"'https://github.com/onnx/models/raw/main/validated/vision/classification/"
              f"mobilenet/model/mobilenetv2-12.onnx', r'{MODEL_PATH}')\"")
        sys.exit(2)

    results = asyncio.run(run(args.iterations, args.warmup, args.verbose))

    if args.json_out:
        # Strip non-serializable bits (ndarray, etc.) — everything in `results` is already plain.
        args.json_out.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Wrote results JSON: {args.json_out}")


if __name__ == "__main__":
    main()
