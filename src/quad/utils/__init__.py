"""QUAD Utils package."""

from quad.utils.perf import (
    BURST_CPU_SLEEP_THRESHOLD_MS,
    BURST_DEFAULT_INACTIVITY_TIMEOUT_MS,
    BURST_DEFAULT_INACTIVITY_TIMEOUT_US,
    BurstModeConfig,
    DSPGuidance,
    GPU_OVERHEAD_MS_TYPICAL,
    GPU_SPEEDUP_FACTOR_LOW,
    GPU_SPEEDUP_FACTOR_HIGH,
    PerformanceProfile,
    ProfilingLevel,
    RuntimeRecommendation,
    build_platform_options,
    get_dsp_guidance,
    get_profiling_recommendation,
    recommend_runtime,
)
from quad.utils.snpe_logging import (
    SNPELogLevel,
    SNPELoggingConfig,
    get_logging_config,
)

__all__ = [
    "BURST_CPU_SLEEP_THRESHOLD_MS",
    "BURST_DEFAULT_INACTIVITY_TIMEOUT_MS",
    "BURST_DEFAULT_INACTIVITY_TIMEOUT_US",
    "BurstModeConfig",
    "DSPGuidance",
    "GPU_OVERHEAD_MS_TYPICAL",
    "GPU_SPEEDUP_FACTOR_HIGH",
    "GPU_SPEEDUP_FACTOR_LOW",
    "PerformanceProfile",
    "ProfilingLevel",
    "RuntimeRecommendation",
    "SNPELogLevel",
    "SNPELoggingConfig",
    "build_platform_options",
    "get_dsp_guidance",
    "get_logging_config",
    "get_profiling_recommendation",
    "recommend_runtime",
]
