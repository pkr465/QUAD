"""MobilenetSSD benchmarking helpers — SNPE benchmark config and input list generation.

Based on SNPE "MobilenetSSD Benchmarking" documentation (80-63442-10 Rev AH).

Key MobilenetSSD-specific requirements vs standard benchmark:
  1. "CpuFallback": true  — DetectionOutput layer only runs on CPU
  2. "BufferTypes": ["ub_float", "ub_tf8"]  — test both float and INT8
  3. Output layer names in imagelist.txt first line:
       #Postprocessor/BatchMultiClassNonMaxSuppression add_6
  4. GPU/DSP are ~17-39x faster than CPU for MobilenetSSD
  5. Input resizing NOT possible (PriorBox layer folding prevents it)

Performance:
  Comparing Total Inference Time and Forward Propagate rows:
  GPU and DSP inference times are ~17-39x faster than CPU.
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from quad.benchmarks.config import (
    BenchmarkModelConfig,
    BenchmarkResults,
    BenchmarkTimingRow,
    SNPEBenchmarkConfig,
)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# Output layers for Tensorflow MobilenetSSD (as documented)
MOBILENET_SSD_OUTPUT_LAYERS = [
    "Postprocessor/BatchMultiClassNonMaxSuppression",
    "add_6",
]

# Default benchmark config values for MobilenetSSD
MOBILENET_SSD_DEFAULTS: dict[str, Any] = {
    "name": "mobilenet_ssd",
    "host_root_path": "mobilenet_ssd",
    "host_results_dir": "mobilenet_ssd/results",
    "device_path": "/data/local/tmp/snpebm",
    "runs": 2,
    "runtimes": ["GPU"],
    "measurements": ["timing"],
    "profiling_level": "detailed",           # detailed required for per-layer CSV data
    "buffer_types": ["ub_float", "ub_tf8"],  # MobilenetSSD requires both
    "cpu_fallback": True,                    # Required: DetectionOutput runs on CPU
}

# From SNPE docs: GPU and DSP are ~17-39x faster than CPU for MobilenetSSD
MOBILENET_SSD_GPU_DSP_SPEEDUP_RANGE = (17, 39)

# Note from docs: to get all timing information, profiling_level must be "detailed"
# Basic profiling only gives summary timing.
TIMING_NOTE = (
    "To get all timing information in the CSV, set ProfilingLevel to 'detailed'. "
    "Default is 'basic' which gives summary timing only."
)


# ══════════════════════════════════════════════════════════════════════════════
# Config Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_mobilenet_ssd_benchmark_config(
    dlc_path: str,
    input_list_path: str,
    image_data_dirs: list[str],
    device_serials: list[str],
    *,
    runtimes: list[str] | None = None,
    host_root_path: str = "mobilenet_ssd",
    host_results_dir: str = "mobilenet_ssd/results",
    device_path: str = "/data/local/tmp/snpebm",
    runs: int = 2,
    profiling_level: str = "detailed",
) -> SNPEBenchmarkConfig:
    """Build the benchmark JSON config for MobilenetSSD.

    Applies all MobilenetSSD-specific requirements:
    - CpuFallback: true
    - BufferTypes: ["ub_float", "ub_tf8"]
    - ProfilingLevel: detailed (to get full CSV data)

    Args:
        dlc_path: Path to converted mobilenet_ssd.dlc
        input_list_path: Path to imagelist.txt (with output layer header)
        image_data_dirs: Directories containing .rawtensor image files
        device_serials: ADB device serial numbers (e.g. ["454d40f3"])
        runtimes: List of runtimes to benchmark (default: ["GPU"])
        host_root_path: Working directory on host
        host_results_dir: Where to write CSV/JSON results
        device_path: Path on Android device
        runs: Number of benchmark runs
        profiling_level: "basic" or "detailed"

    Returns:
        SNPEBenchmarkConfig ready to write to JSON and pass to snpe_bench.py

    Example::
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/mobilenet_ssd.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=["454d40f3"],
        )
        cfg.write("/tmp/mobilenetssd.json")
    """
    model = BenchmarkModelConfig(
        name="mobilenet_ssd",
        dlc=dlc_path,
        input_list=input_list_path,
        data=image_data_dirs,
    )
    return SNPEBenchmarkConfig(
        name=MOBILENET_SSD_DEFAULTS["name"],
        host_root_path=host_root_path,
        host_results_dir=host_results_dir,
        device_path=device_path,
        devices=device_serials,
        model=model,
        runs=runs,
        runtimes=runtimes or MOBILENET_SSD_DEFAULTS["runtimes"],
        measurements=MOBILENET_SSD_DEFAULTS["measurements"],
        profiling_level=profiling_level,
        buffer_types=MOBILENET_SSD_DEFAULTS["buffer_types"],
        cpu_fallback=True,  # Always required for MobilenetSSD
    )


# ══════════════════════════════════════════════════════════════════════════════
# Input List Generator
# ══════════════════════════════════════════════════════════════════════════════

def build_mobilenet_ssd_input_list(
    image_raw_paths: list[str],
    output_path: str,
    output_layers: list[str] | None = None,
) -> str:
    """Write imagelist.txt for MobilenetSSD benchmark.

    MobilenetSSD has multiple output layers. snpe_bench.py needs them
    specified on the first line of the input list, prefixed with '#'
    and space-separated.

    Format:
        #Postprocessor/BatchMultiClassNonMaxSuppression add_6
        /tmp/images/0#.rawtensor
        /tmp/images/1#.rawtensor
        ...

    NOTE: If the model is retrained and output layers change, this first
    line must be updated to match the new output layer names.

    Args:
        image_raw_paths: Paths to .rawtensor image files
        output_path: Path to write imagelist.txt
        output_layers: Output layer names (defaults to documented layers)

    Returns:
        Path written to (same as output_path)
    """
    layers = output_layers or MOBILENET_SSD_OUTPUT_LAYERS
    lines = [f"#{' '.join(layers)}"]
    lines.extend(image_raw_paths)

    Path(output_path).write_text("\n".join(lines) + "\n")
    return output_path


def parse_input_list(path: str) -> tuple[list[str], list[str]]:
    """Parse an SNPE imagelist.txt file.

    Returns:
        (output_layers, image_paths)
        output_layers: list from the '#' header line (empty if absent)
        image_paths: list of image file paths
    """
    output_layers: list[str] = []
    image_paths: list[str] = []

    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # Output layer header
            header = line[1:].strip()
            if header:
                output_layers = header.split()
        else:
            image_paths.append(line)

    return output_layers, image_paths


# ══════════════════════════════════════════════════════════════════════════════
# snpe_bench.py Runner
# ══════════════════════════════════════════════════════════════════════════════

def find_snpe_bench(sdk_root: str | None = None) -> str | None:
    """Locate snpe_bench.py in the SNPE SDK or PATH.

    snpe_bench.py lives at $SNPE_ROOT/benchmarks/SNPE/snpe_bench.py
    """
    # Check PATH
    tool = shutil.which("snpe_bench.py") or shutil.which("snpe-bench")
    if tool:
        return tool

    # Check SDK root
    root = sdk_root or os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("SNPE_ROOT")
    if root:
        candidates = [
            Path(root) / "benchmarks" / "SNPE" / "snpe_bench.py",
            Path(root) / "benchmarks" / "snpe_bench.py",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    return None


def build_snpe_bench_cmd(
    config_path: str,
    *,
    sdk_root: str | None = None,
    generate_json: bool = False,
) -> list[str]:
    """Build the snpe_bench.py command line.

    Args:
        config_path: Path to the benchmark JSON config file
        sdk_root: SNPE SDK root (used to locate snpe_bench.py)
        generate_json: If True, add --generate_json to produce JSON output
                       in addition to CSV

    Returns:
        CLI argument list: ["python3", "snpe_bench.py", "-c", config_path, "-a"]
    """
    bench = find_snpe_bench(sdk_root)
    if bench is None:
        raise FileNotFoundError(
            "snpe_bench.py not found. Set SNPE_ROOT or QAIRT_SDK_ROOT, "
            "or add SDK benchmarks directory to PATH."
        )

    cmd = ["python3", bench, "-c", config_path, "-a"]
    if generate_json:
        cmd.append("--generate_json")
    return cmd


def run_benchmark(
    config_path: str,
    *,
    sdk_root: str | None = None,
    generate_json: bool = False,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess:
    """Run snpe_bench.py for the given config file.

    Args:
        config_path: Path to JSON config (e.g. /tmp/mobilenetssd.json)
        sdk_root: SNPE SDK root directory
        generate_json: Also generate JSON results file
        timeout: Max seconds to wait

    Returns:
        CompletedProcess with stdout/stderr

    Raises:
        FileNotFoundError: If snpe_bench.py is not found
        RuntimeError: If the benchmark exits with non-zero code
        TimeoutError: If the benchmark exceeds timeout
    """
    cmd = build_snpe_bench_cmd(config_path, sdk_root=sdk_root, generate_json=generate_json)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"snpe_bench.py timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(
            f"snpe_bench.py failed (exit {result.returncode}):\n{result.stderr[:1000]}"
        )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Results Parser
# ══════════════════════════════════════════════════════════════════════════════

def get_latest_results_dir(host_results_dir: str) -> str | None:
    """Return the path to the latest_results directory.

    snpe_bench.py creates timestamped subdirectories and a latest_results
    symlink pointing to the most recent run.
    """
    latest = Path(host_results_dir) / "latest_results"
    if latest.exists():
        return str(latest.resolve())

    # Fallback: find most recent timestamped directory
    candidates = sorted(
        [d for d in Path(host_results_dir).iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def parse_benchmark_csv(csv_path: str) -> BenchmarkResults:
    """Parse a snpe_bench.py CSV results file into structured data.

    CSV columns include: Runtime, Total Inference Time (us),
    Forward Propagate (us), and per-layer timings when profiling_level=detailed.
    All timing values are in microseconds.

    Args:
        csv_path: Path to the .csv results file

    Returns:
        BenchmarkResults with per-runtime timing rows
    """
    import csv

    path = Path(csv_path)
    rows: list[BenchmarkTimingRow] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            runtime = row.get("Runtime", row.get("runtime", ""))
            if not runtime:
                continue

            def _us(key: str) -> float:
                for k in (key, key.lower(), key.replace(" ", "_").lower()):
                    if k in row and row[k]:
                        try:
                            return float(row[k])
                        except ValueError:
                            pass
                return 0.0

            total = _us("Total Inference Time") or _us("total_inference_time")
            fwd = _us("Forward Propagate") or _us("forward_propagate")

            # Collect remaining numeric columns as layer timings
            layers: dict[str, float] = {}
            for k, v in row.items():
                if k in ("Runtime", "Total Inference Time", "Forward Propagate"):
                    continue
                try:
                    layers[k] = float(v)
                except (ValueError, TypeError):
                    pass

            rows.append(BenchmarkTimingRow(
                runtime=runtime,
                total_inference_us=total,
                forward_propagate_us=fwd,
                layers=layers,
            ))

    return BenchmarkResults(
        model_name=path.stem,
        run_dir=str(path.parent),
        rows=rows,
    )


def parse_benchmark_json(json_path: str) -> BenchmarkResults:
    """Parse a snpe_bench.py JSON results file (from --generate_json).

    JSON format mirrors the CSV but is structured as key-value pairs.
    """
    import json as json_module

    path = Path(json_path)
    data = json_module.loads(path.read_text())

    rows: list[BenchmarkTimingRow] = []
    # JSON structure varies — try common patterns
    results_data = data.get("results", data.get("Results", [data]))
    if isinstance(results_data, dict):
        results_data = [results_data]

    for item in results_data:
        if not isinstance(item, dict):
            continue
        runtime = item.get("Runtime", item.get("runtime", ""))
        if not runtime:
            continue
        total = float(item.get("Total Inference Time", item.get("total_inference_us", 0)) or 0)
        fwd = float(item.get("Forward Propagate", item.get("forward_propagate_us", 0)) or 0)
        rows.append(BenchmarkTimingRow(
            runtime=runtime,
            total_inference_us=total,
            forward_propagate_us=fwd,
        ))

    return BenchmarkResults(
        model_name=path.stem,
        run_dir=str(path.parent),
        rows=rows,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

MOBILENET_SSD_BENCHMARK_NOTES: dict[str, Any] = {
    "description": (
        "MobilenetSSD TensorFlow benchmarking via snpe_bench.py. "
        "Requires CpuFallback=true for GPU/DSP runs "
        "(DetectionOutput layer is CPU-only)."
    ),
    "required_config_fields": {
        "CpuFallback": {
            "value": True,
            "reason": (
                "DetectionOutput layer only runs on CPU. "
                "Without CpuFallback, GPU/DSP runs will fail or produce incorrect results."
            ),
        },
        "BufferTypes": {
            "value": ["ub_float", "ub_tf8"],
            "reason": "Test both float32 and INT8 quantized buffer modes.",
        },
        "ProfilingLevel": {
            "recommended": "detailed",
            "note": (
                "Default is 'basic'. Must be 'detailed' to get per-layer timing data "
                "in the CSV output file."
            ),
        },
    },
    "output_layers": {
        "names": MOBILENET_SSD_OUTPUT_LAYERS,
        "input_list_format": (
            "Add output layer names on first line of imagelist.txt, "
            "prefixed with '#', space-separated:\n"
            "  #Postprocessor/BatchMultiClassNonMaxSuppression add_6\n"
            "  /tmp/images/0#.rawtensor\n"
            "  /tmp/images/1#.rawtensor"
        ),
        "update_note": (
            "If the model is retrained and output layers change, "
            "the first line in imagelist.txt must be updated."
        ),
    },
    "performance": {
        "gpu_dsp_vs_cpu_speedup": "17-39x",
        "note": (
            "Comparing Total Inference Time: GPU and DSP are ~17-39x faster than CPU. "
            "Use GPU or DSP for production MobilenetSSD deployments."
        ),
    },
    "results": {
        "location": "HostResultsDir/YYYYMMDDHHMMSS/",
        "latest_link": "latest_results → most recent timestamped directory",
        "formats": ["CSV (default)", "JSON (--generate_json flag)"],
        "timing_unit": "microseconds",
        "run_command": "python3 snpe_bench.py -c /tmp/mobilenetssd.json -a",
        "json_run_command": "python3 snpe_bench.py -c /tmp/mobilenetssd.json -a --generate_json",
    },
    "example_files": {
        "dlc": "/tmp/mobilenetssd.dlc",
        "config": "/tmp/mobilenetssd.json",
        "input_list": "/tmp/imagelist.txt",
        "images_dir": "/tmp/images.rawtensor",
    },
}
