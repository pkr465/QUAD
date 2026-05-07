"""QUAD Benchmarks — SNPE benchmark config and result parsing."""

from quad.benchmarks.config import (
    BenchmarkModelConfig,
    BenchmarkResults,
    BenchmarkTimingRow,
    SNPEBenchmarkConfig,
    VALID_BUFFER_TYPES,
    VALID_MEASUREMENTS,
    VALID_PROFILING_LEVELS,
    VALID_RUNTIMES,
)
from quad.benchmarks.mobilenet_ssd import (
    MOBILENET_SSD_BENCHMARK_NOTES,
    MOBILENET_SSD_DEFAULTS,
    MOBILENET_SSD_GPU_DSP_SPEEDUP_RANGE,
    MOBILENET_SSD_OUTPUT_LAYERS,
    TIMING_NOTE,
    build_mobilenet_ssd_benchmark_config,
    build_mobilenet_ssd_input_list,
    build_snpe_bench_cmd,
    find_snpe_bench,
    get_latest_results_dir,
    parse_benchmark_csv,
    parse_benchmark_json,
    parse_input_list,
    run_benchmark,
)

__all__ = [
    # Config models
    "BenchmarkModelConfig",
    "BenchmarkResults",
    "BenchmarkTimingRow",
    "SNPEBenchmarkConfig",
    "VALID_BUFFER_TYPES",
    "VALID_MEASUREMENTS",
    "VALID_PROFILING_LEVELS",
    "VALID_RUNTIMES",
    # MobilenetSSD
    "MOBILENET_SSD_BENCHMARK_NOTES",
    "MOBILENET_SSD_DEFAULTS",
    "MOBILENET_SSD_GPU_DSP_SPEEDUP_RANGE",
    "MOBILENET_SSD_OUTPUT_LAYERS",
    "TIMING_NOTE",
    "build_mobilenet_ssd_benchmark_config",
    "build_mobilenet_ssd_input_list",
    "build_snpe_bench_cmd",
    "find_snpe_bench",
    "get_latest_results_dir",
    "parse_benchmark_csv",
    "parse_benchmark_json",
    "parse_input_list",
    "run_benchmark",
]
