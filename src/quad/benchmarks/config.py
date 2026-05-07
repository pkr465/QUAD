"""SNPE Benchmark Configuration Models.

Based on SNPE Benchmarking documentation (80-63442-10 Rev AH, Apr 13 2026).

The snpe_bench.py tool takes a JSON config file describing:
- Model location (.dlc file)
- Device identifiers (adb serial numbers)
- Runtime targets (CPU, GPU, DSP, AIP)
- Measurement types (timing, etc.)
- Buffer types (ub_float, ub_tf8, etc.)

MobilenetSSD requires two additional fields versus a standard config:
  "CpuFallback": true   — required because DetectionOutput runs on CPU
  "BufferTypes": ["ub_float", "ub_tf8"]

Multi-output models (e.g. MobilenetSSD) also require output layer names
on the first line of the input list file, prefixed with '#':
  #Postprocessor/BatchMultiClassNonMaxSuppression add_6
  /tmp/images/0.rawtensor
  /tmp/images/1.rawtensor
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Runtime and Buffer Type Enums
# ══════════════════════════════════════════════════════════════════════════════

VALID_RUNTIMES = frozenset(["CPU", "GPU", "DSP", "AIP"])
VALID_BUFFER_TYPES = frozenset(["ub_float", "ub_tf8", "ub_float16"])
VALID_PROFILING_LEVELS = frozenset(["basic", "detailed", "linting", "qhas"])
VALID_MEASUREMENTS = frozenset(["timing", "power", "memory"])


# ══════════════════════════════════════════════════════════════════════════════
# Config Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkModelConfig:
    """Model section of the benchmark JSON config.

    Corresponds to the "Model" block in the benchmark JSON file.
    """
    name: str
    dlc: str                              # Path to .dlc model file
    input_list: str                       # Path to imagelist.txt
    data: list[str] = field(default_factory=list)  # Directories of raw input files

    def to_dict(self) -> dict[str, Any]:
        return {
            "Name": self.name,
            "Dlc": self.dlc,
            "InputList": self.input_list,
            "Data": self.data,
        }


@dataclass
class SNPEBenchmarkConfig:
    """Full SNPE benchmark JSON configuration.

    Maps directly to the fields consumed by snpe_bench.py.

    Standard fields are shared by all models. Model-specific extras
    (e.g. CpuFallback for MobilenetSSD) are represented as optional fields.

    Results are stored under HostResultsDir/YYYYMMDDHHMMSS/ with a
    latest_results symlink for convenience. Timing values are in microseconds.
    """
    name: str
    host_root_path: str                   # Working directory on host
    host_results_dir: str                 # Where CSV/JSON results are written
    device_path: str                      # Path on Android device (e.g. /data/local/tmp/snpebm)
    devices: list[str]                    # ADB serial numbers
    model: BenchmarkModelConfig
    runs: int = 2                         # Number of benchmark runs (averaged)
    runtimes: list[str] = field(default_factory=lambda: ["CPU"])
    measurements: list[str] = field(default_factory=lambda: ["timing"])
    profiling_level: str = "basic"        # basic (default) or detailed for per-layer timing
    buffer_types: list[str] = field(default_factory=list)
    cpu_fallback: bool = False            # Required for MobilenetSSD on GPU/DSP

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors = []
        for rt in self.runtimes:
            if rt not in VALID_RUNTIMES:
                errors.append(
                    f"Invalid runtime '{rt}'. Valid: {sorted(VALID_RUNTIMES)}"
                )
        for bt in self.buffer_types:
            if bt not in VALID_BUFFER_TYPES:
                errors.append(
                    f"Invalid buffer type '{bt}'. Valid: {sorted(VALID_BUFFER_TYPES)}"
                )
        if self.profiling_level not in VALID_PROFILING_LEVELS:
            errors.append(
                f"Invalid profiling_level '{self.profiling_level}'. "
                f"Valid: {sorted(VALID_PROFILING_LEVELS)}"
            )
        for m in self.measurements:
            if m not in VALID_MEASUREMENTS:
                errors.append(f"Invalid measurement '{m}'. Valid: {sorted(VALID_MEASUREMENTS)}")
        if self.runs < 1:
            errors.append("runs must be >= 1")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the dict format expected by snpe_bench.py."""
        d: dict[str, Any] = {
            "Name": self.name,
            "HostRootPath": self.host_root_path,
            "HostResultsDir": self.host_results_dir,
            "DevicePath": self.device_path,
            "Devices": self.devices,
            "Runs": self.runs,
            "Model": self.model.to_dict(),
            "Runtimes": self.runtimes,
            "Measurements": self.measurements,
            "ProfilingLevel": self.profiling_level,
        }
        if self.buffer_types:
            d["BufferTypes"] = self.buffer_types
        if self.cpu_fallback:
            d["CpuFallback"] = True
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string for writing to a config file."""
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: str) -> None:
        """Write config to a JSON file."""
        Path(path).write_text(self.to_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SNPEBenchmarkConfig":
        """Parse a benchmark config dict (as read from JSON file)."""
        model_data = data.get("Model", {})
        model = BenchmarkModelConfig(
            name=model_data.get("Name", ""),
            dlc=model_data.get("Dlc", ""),
            input_list=model_data.get("InputList", ""),
            data=model_data.get("Data", []),
        )
        return cls(
            name=data.get("Name", ""),
            host_root_path=data.get("HostRootPath", ""),
            host_results_dir=data.get("HostResultsDir", ""),
            device_path=data.get("DevicePath", "/data/local/tmp/snpebm"),
            devices=data.get("Devices", []),
            model=model,
            runs=data.get("Runs", 2),
            runtimes=data.get("Runtimes", ["CPU"]),
            measurements=data.get("Measurements", ["timing"]),
            profiling_level=data.get("ProfilingLevel", "basic").lower(),
            buffer_types=data.get("BufferTypes", []),
            cpu_fallback=data.get("CpuFallback", False),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SNPEBenchmarkConfig":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "SNPEBenchmarkConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark Result Models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkTimingRow:
    """One row from the CSV/JSON benchmark results file.

    All timing values are in microseconds (as specified in SNPE docs).
    The CSV contains: Runtime, Total Inference Time, Forward Propagate Time,
    and other per-layer metrics when profiling_level=detailed.
    """
    runtime: str
    total_inference_us: float          # "Total Inference Time" column
    forward_propagate_us: float        # "Forward Propagate" column
    layers: dict[str, float] = field(default_factory=dict)  # layer name → us

    @property
    def total_inference_ms(self) -> float:
        return self.total_inference_us / 1000.0

    @property
    def forward_propagate_ms(self) -> float:
        return self.forward_propagate_us / 1000.0

    def speedup_vs(self, other: "BenchmarkTimingRow") -> float:
        """Compute speedup ratio of this result vs another (higher = faster)."""
        if self.total_inference_us == 0:
            return 0.0
        return other.total_inference_us / self.total_inference_us


@dataclass
class BenchmarkResults:
    """All results from a benchmark run (from CSV or JSON output file)."""
    model_name: str
    run_dir: str                                    # Timestamped result directory
    rows: list[BenchmarkTimingRow] = field(default_factory=list)

    @property
    def cpu_row(self) -> Optional[BenchmarkTimingRow]:
        return next((r for r in self.rows if r.runtime.upper() == "CPU"), None)

    @property
    def gpu_row(self) -> Optional[BenchmarkTimingRow]:
        return next((r for r in self.rows if r.runtime.upper() == "GPU"), None)

    @property
    def dsp_row(self) -> Optional[BenchmarkTimingRow]:
        return next(
            (r for r in self.rows if r.runtime.upper() in ("DSP", "NPU", "HTP")), None
        )

    def gpu_vs_cpu_speedup(self) -> Optional[float]:
        """GPU speedup over CPU (None if either is missing)."""
        if self.cpu_row and self.gpu_row:
            return self.gpu_row.speedup_vs(self.cpu_row)
        return None

    def dsp_vs_cpu_speedup(self) -> Optional[float]:
        """DSP/NPU speedup over CPU (None if either is missing)."""
        if self.cpu_row and self.dsp_row:
            return self.dsp_row.speedup_vs(self.cpu_row)
        return None
