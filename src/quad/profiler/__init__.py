"""QUAD Deep Profiler — Nsight-equivalent profiling depth for Qualcomm hardware."""

from quad.profiler.levels import ProfilingLevel
from quad.profiler.system import SystemProfiler
from quad.profiler.kernel import KernelProfiler
from quad.profiler.power_profiler import PowerProfiler
from quad.profiler.memory_profiler import MemoryProfiler
from quad.profiler.roofline import RooflineAnalysis
from quad.profiler.api import profile_model
from quad.profiler.diagview import (
    find_diagview,
    parse_diaglog_as_linting,
    run_diagview,
    run_diagview_chrometrace,
)
from quad.profiler.linting import (
    LintingProfile,
    LintingSubnetProfile,
    LintingOpMetrics,
    HTPResource,
    LINTING_PROFILE_NOTES,
    OP_SUBSTITUTIONS,
    parse_linting_output,
    analyze_bottlenecks,
    format_linting_report,
    build_linting_cli_args,
    build_diagview_chrometrace_args,
)
from quad.profiler.qhas import (
    QHASConfig,
    QHASWorkflow,
    QHAS_PROFILE_NOTES,
    QHAS_PROFILING_LEVEL,
    QHAS_LOG_FILENAME,
    QHAS_READER_LIB,
    SCHEMATIC_SUFFIX,
    get_schematic_path,
    get_profilelogs_dir,
    get_log_path,
    get_reader_lib_path,
    build_graph_prepare_qhas_args,
    build_net_run_qhas_args,
    build_profile_viewer_args,
)

__all__ = [
    "ProfilingLevel",
    "SystemProfiler",
    "KernelProfiler",
    "PowerProfiler",
    "MemoryProfiler",
    "RooflineAnalysis",
    "profile_model",
    # Diagview
    "find_diagview",
    "parse_diaglog_as_linting",
    "run_diagview",
    "run_diagview_chrometrace",
    # Linting
    "LintingProfile",
    "LintingSubnetProfile",
    "LintingOpMetrics",
    "HTPResource",
    "LINTING_PROFILE_NOTES",
    "OP_SUBSTITUTIONS",
    "parse_linting_output",
    "analyze_bottlenecks",
    "format_linting_report",
    "build_linting_cli_args",
    "build_diagview_chrometrace_args",
    # QHAS
    "QHASConfig",
    "QHASWorkflow",
    "QHAS_PROFILE_NOTES",
    "QHAS_PROFILING_LEVEL",
    "QHAS_LOG_FILENAME",
    "QHAS_READER_LIB",
    "SCHEMATIC_SUFFIX",
    "get_schematic_path",
    "get_profilelogs_dir",
    "get_log_path",
    "get_reader_lib_path",
    "build_graph_prepare_qhas_args",
    "build_net_run_qhas_args",
    "build_profile_viewer_args",
]
