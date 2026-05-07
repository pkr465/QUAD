"""QUAD Performance Utilities — profiles, runtime selection, DSP guidance, burst mode.

Documents rules from SNPE Performance Tips:

Performance Profiles (SNPEBuilder::setPerformanceProfile):
  - DEFAULT      → deprecated, same as BALANCED
  - BALANCED     → default power/perf tradeoff
  - HIGH_PERFORMANCE → maximize perf, higher power
  - POWER_SAVER  → more power saving, lower perf
  - SYSTEM_SETTINGS → SNPE leaves power settings alone (user manages)
  - BURST        → DSP/AIP only: fastest inference, more power than HIGH_PERFORMANCE

Burst Mode (DSP/AIP only — fastest inference):
  - Two features active simultaneously:
    1. Prevents CPUs entering deepest sleep during inference (<100ms threshold)
    2. Keeps DSP performance vote high for 300ms after last inference (default)
       Configurable: platform_options="inactivityTimeout:10000" (10ms in microseconds)
  - Use for latency-critical inference bursts
  - Power: higher than HIGH_PERFORMANCE

GPU vs CPU Decision Rule:
  - GPU typically 6-10x faster than CPU
  - BUT: GPU has 4-6ms constant overhead per execute()
  - For networks running <10ms on GPU, CPU may be faster
  - Use GPU for: >10ms networks, high throughput
  - Use CPU for: <10ms networks, latency-critical, GPU busy with rendering

DSP Performance Rules:
  - Input preprocessing (color space, scaling, crop, mean subtract) must
    be done BEFORE passing to SNPE — DSP preprocessing layers are not optimized
  - DSP runs 8-bit quantized math — not all networks are suitable
  - Init cache STRONGLY RECOMMENDED for DSP V68+:
    much longer init times, but init cache reduces this dramatically

Profiling in Production:
  - Disable profiling in production (overhead adds to inference time)
  - Only enable for development/debugging
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# Performance Profiles
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceProfile(str, Enum):
    """SNPE performance profiles (SNPEBuilder::setPerformanceProfile).

    Mapped to snpe-net-run --perf_profile values.
    """
    BALANCED = "balanced"
    HIGH_PERFORMANCE = "high_performance"
    POWER_SAVER = "power_saver"
    SYSTEM_SETTINGS = "system_settings"
    # Burst is a common alias for HIGH_PERFORMANCE used in snpe-net-run
    BURST = "burst"
    # Deprecated aliases (kept for backward compat)
    DEFAULT = "default"  # Deprecated — same as BALANCED

    @property
    def snpe_enum_name(self) -> str:
        """Maps to zdl::DlSystem::PerformanceProfile_t enum name."""
        _map = {
            "balanced": "BALANCED",
            "high_performance": "HIGH_PERFORMANCE",
            "power_saver": "POWER_SAVER",
            "system_settings": "SYSTEM_SETTINGS",
            "burst": "BURST",
            "default": "DEFAULT",
        }
        return _map.get(self.value, self.value.upper())

    @property
    def is_deprecated(self) -> bool:
        return self == PerformanceProfile.DEFAULT

    @property
    def power_level(self) -> str:
        """Qualitative power consumption level."""
        return {
            "balanced": "moderate",
            "high_performance": "high",
            "burst": "high",
            "power_saver": "low",
            "system_settings": "user-controlled",
            "default": "moderate (deprecated)",
        }.get(self.value, "unknown")

    @property
    def description(self) -> str:
        return {
            "balanced": "Default power/performance tradeoff. DEFAULT is deprecated alias.",
            "high_performance": "Maximum performance at the expense of power. Use for latency-critical apps.",
            "burst": "Alias for HIGH_PERFORMANCE in snpe-net-run.",
            "power_saver": "More power saving than BALANCED, may reduce performance.",
            "system_settings": "SNPE does not modify power/performance settings. User manages via external APIs.",
            "default": "Deprecated. Same as BALANCED.",
        }.get(self.value, "")


# ══════════════════════════════════════════════════════════════════════════════
# GPU vs CPU Decision
# ══════════════════════════════════════════════════════════════════════════════

# GPU has a constant per-execute overhead (from Performance Tips docs)
GPU_OVERHEAD_MS_LOW = 4.0
GPU_OVERHEAD_MS_HIGH = 6.0
GPU_OVERHEAD_MS_TYPICAL = 5.0

# GPU speedup range vs CPU (from docs: "6X-10X speed increase")
GPU_SPEEDUP_FACTOR_LOW = 6.0
GPU_SPEEDUP_FACTOR_HIGH = 10.0


@dataclass
class RuntimeRecommendation:
    """Recommendation for which runtime to use."""
    recommended_runtime: str        # "npu", "gpu", "cpu"
    reason: str
    estimated_latency_ms: float
    alternatives: list[str]
    warnings: list[str]


def recommend_runtime(
    cpu_latency_ms: float,
    npu_available: bool = True,
    gpu_available: bool = True,
    gpu_utilization_pct: float = 0.0,
) -> RuntimeRecommendation:
    """Recommend the best runtime based on performance characteristics.

    Applies the rules from SNPE Performance Tips:
    1. NPU (DSP/HTP) is typically best for AI workloads if available
    2. GPU is 6-10x faster than CPU but has 4-6ms constant overhead
    3. For networks <10ms on GPU, CPU may be faster due to GPU overhead
    4. Avoid GPU if utilization is high (e.g. gaming)

    Args:
        cpu_latency_ms: Expected inference time on CPU (ms)
        npu_available: Whether NPU/DSP is available
        gpu_available: Whether GPU is available
        gpu_utilization_pct: Current GPU utilization (0-100%)

    Returns:
        RuntimeRecommendation with reasoning.
    """
    warnings: list[str] = []

    # NPU is almost always best for AI workloads
    if npu_available:
        npu_estimate = cpu_latency_ms / GPU_SPEEDUP_FACTOR_HIGH
        return RuntimeRecommendation(
            recommended_runtime="npu",
            reason="NPU/DSP provides best AI performance with lowest power consumption.",
            estimated_latency_ms=npu_estimate,
            alternatives=["gpu", "cpu"],
            warnings=[],
        )

    # GPU decision: apply overhead rule
    if gpu_available:
        gpu_busy_warning = (
            f"GPU is at {gpu_utilization_pct:.0f}% utilization — "
            "consider CPU to avoid competing with rendering."
        ) if gpu_utilization_pct > 50 else ""

        gpu_speedup = (GPU_SPEEDUP_FACTOR_LOW + GPU_SPEEDUP_FACTOR_HIGH) / 2  # ~8x
        gpu_raw_ms = cpu_latency_ms / gpu_speedup
        gpu_total_ms = gpu_raw_ms + GPU_OVERHEAD_MS_TYPICAL  # Add constant overhead

        if gpu_busy_warning:
            warnings.append(gpu_busy_warning)

        # Key rule: GPU is only beneficial if total GPU time < CPU time
        # Rule of thumb: if GPU network time < 10ms, CPU might win
        if gpu_total_ms < cpu_latency_ms:
            if gpu_raw_ms < 10.0:
                warnings.append(
                    f"Network runs in ~{gpu_raw_ms:.1f}ms on GPU (< 10ms threshold). "
                    f"CPU may be faster than GPU due to {GPU_OVERHEAD_MS_TYPICAL}ms GPU overhead. "
                    f"Benchmark both options."
                )
            return RuntimeRecommendation(
                recommended_runtime="gpu" if gpu_utilization_pct < 50 else "cpu",
                reason=(
                    f"GPU ({gpu_total_ms:.1f}ms) faster than CPU ({cpu_latency_ms:.1f}ms). "
                    f"GPU provides {gpu_speedup:.0f}x speedup minus {GPU_OVERHEAD_MS_TYPICAL}ms overhead."
                ),
                estimated_latency_ms=gpu_total_ms,
                alternatives=["cpu"],
                warnings=warnings,
            )

    # Default to CPU
    return RuntimeRecommendation(
        recommended_runtime="cpu",
        reason="CPU is the baseline runtime. Consider GPU or NPU for better performance.",
        estimated_latency_ms=cpu_latency_ms,
        alternatives=["gpu", "npu"],
        warnings=warnings,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DSP Performance Guidance
# ══════════════════════════════════════════════════════════════════════════════

# DSP V68+ has significantly longer init times — init cache strongly recommended
DSP_INIT_CACHE_REQUIRED_VERSION = "v68"
DSP_QUANTIZATION_BITS = 8  # DSP runs 8-bit quantized math


@dataclass
class DSPGuidance:
    """Performance guidance for using DSP runtime."""
    use_init_cache: bool
    init_cache_reason: str
    preprocess_before_snpe: bool
    preprocess_reason: str
    quantization_warning: str | None
    estimated_speedup_vs_cpu: str


def get_dsp_guidance(
    hexagon_version: str,
    model_has_preprocessing_layers: bool = False,
    model_is_quantized: bool = True,
) -> DSPGuidance:
    """Get DSP-specific performance guidance.

    Based on SNPE Performance Tips for DSP runtime.

    Args:
        hexagon_version: e.g. "v68", "v73", "v75"
        model_has_preprocessing_layers: Whether the model includes input
            preprocessing (color space, scaling, crop, mean subtract).
            These are NOT optimized on DSP — do them before SNPE.
        model_is_quantized: Whether model is INT8 quantized.
    """
    version_num = int(hexagon_version.lstrip("v"))

    # Init cache strongly recommended for V68+
    use_cache = version_num >= 68
    cache_reason = (
        f"DSP {hexagon_version} has significantly longer init times. "
        "Init cache greatly reduces subsequent initialization time "
        "and improves execution times via data locality."
    ) if use_cache else (
        f"DSP {hexagon_version} init times are acceptable without caching."
    )

    # Preprocessing warning
    preprocess_reason = (
        "Input preprocessing layers (color space conversion, scaling, "
        "crop, mean subtract) are NOT optimized on DSP. "
        "Perform these operations before calling SNPE execute()."
    ) if model_has_preprocessing_layers else (
        "No preprocessing layers detected. If adding preprocessing, "
        "do it before passing input to SNPE."
    )

    # Quantization check
    quant_warning = None if model_is_quantized else (
        "DSP runs 8-bit quantized math. FP32 model may have accuracy loss "
        "or may not be suitable for DSP. Quantize before deploying to DSP."
    )

    return DSPGuidance(
        use_init_cache=use_cache,
        init_cache_reason=cache_reason,
        preprocess_before_snpe=model_has_preprocessing_layers,
        preprocess_reason=preprocess_reason,
        quantization_warning=quant_warning,
        estimated_speedup_vs_cpu="comparable to GPU (6-10x) for quantized models",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Production Profiling Guide
# ══════════════════════════════════════════════════════════════════════════════

class ProfilingLevel(str, Enum):
    """SNPE profiling levels (Snpe_SNPEBuilder_SetProfilingLevel)."""
    OFF = "off"           # Production: no overhead
    BASIC = "basic"       # Minimal info, low overhead
    MODERATE = "moderate" # Per-layer timing, moderate overhead
    DETAILED = "detailed" # Full profiling, higher overhead
    LINTING = "linting"   # Detailed with layer analysis

    @property
    def is_production_safe(self) -> bool:
        """True if safe to use in production (minimal overhead)."""
        return self in (ProfilingLevel.OFF, ProfilingLevel.BASIC)

    @property
    def snpe_net_run_flag(self) -> str:
        """Flag for snpe-net-run --profiling_level."""
        return self.value


def get_profiling_recommendation(is_production: bool) -> ProfilingLevel:
    """Return the recommended profiling level.

    From SNPE Performance Tips:
    - Production: disable profiling (overhead adds to inference time)
    - Development: use MODERATE or DETAILED
    """
    return ProfilingLevel.OFF if is_production else ProfilingLevel.DETAILED


# ══════════════════════════════════════════════════════════════════════════════
# Burst Mode Configuration (DSP and AIP only)
# ══════════════════════════════════════════════════════════════════════════════
#
# Burst mode = fastest inference on DSP/AIP at expense of more power.
# Two features:
#   1. Prevents CPU deep sleep during inference (if inference < 100ms)
#   2. Keeps DSP performance vote high for inactivityTimeout after last inference
#      Default: 300ms (300,000 µs). Configurable via platform options.
#
# Platform option syntax: "inactivityTimeout:VALUE_IN_MICROSECONDS"
#   e.g. 10ms = "inactivityTimeout:10000"
#   e.g. 300ms = "inactivityTimeout:300000"  (default)
#   e.g. 500ms = "inactivityTimeout:500000"
#
# Usage:
#   snpe-net-run ... --perf_profile burst --platform_options "inactivityTimeout:10000"
#   C++: snpeBuilder.setPerformanceProfile(PerformanceProfile_t::BURST)
#         + platformConfig.setPlatformOptions("inactivityTimeout:10000")

# Burst mode constants
BURST_CPU_SLEEP_THRESHOLD_MS = 100.0  # CPUs can deep-sleep if inference > 100ms
BURST_DEFAULT_INACTIVITY_TIMEOUT_MS = 300.0  # Default DSP vote hold after last inference
BURST_DEFAULT_INACTIVITY_TIMEOUT_US = 300_000  # In microseconds (platform option unit)


@dataclass
class BurstModeConfig:
    """Configuration for burst mode on DSP/AIP runtimes.

    Burst mode achieves fastest inference by:
    1. Preventing CPU deep sleep during inference (<100ms threshold)
    2. Holding DSP performance vote for inactivity_timeout_ms after inference

    The inactivity timeout controls power vs responsiveness tradeoff:
    - Short timeout (e.g. 10ms): less power, brief pause before DSP vote drops
    - Long timeout (e.g. 300ms default): more power, instant response to burst
    """
    inactivity_timeout_ms: float = BURST_DEFAULT_INACTIVITY_TIMEOUT_MS

    @property
    def inactivity_timeout_us(self) -> int:
        """Timeout in microseconds (platform option unit)."""
        return int(self.inactivity_timeout_ms * 1000)

    @property
    def platform_option_string(self) -> str:
        """Returns platform option string for SNPEBuilder or snpe-net-run.

        Format: "inactivityTimeout:VALUE_IN_MICROSECONDS"
        Combined with PD type if needed:
          "unsignedPD:ON;inactivityTimeout:10000"
        """
        return f"inactivityTimeout:{self.inactivity_timeout_us}"

    def combined_with_pd(self, pd_type: str = "unsigned") -> str:
        """Combine with PD type into single platform options string.

        Args:
            pd_type: "unsigned" or "signed"

        Returns:
            e.g. "unsignedPD:ON;inactivityTimeout:10000"
        """
        pd_option = "unsignedPD:OFF" if pd_type == "signed" else "unsignedPD:ON"
        return f"{pd_option};{self.platform_option_string}"

    def is_non_default_timeout(self) -> bool:
        """True if the timeout differs from the 300ms default."""
        return self.inactivity_timeout_ms != BURST_DEFAULT_INACTIVITY_TIMEOUT_MS

    def __repr__(self) -> str:
        return (
            f"BurstModeConfig(timeout={self.inactivity_timeout_ms:.0f}ms, "
            f"option='{self.platform_option_string}')"
        )


def build_platform_options(
    pd_type: str = "unsigned",
    burst_config: BurstModeConfig | None = None,
    extra_options: dict[str, str] | None = None,
) -> str:
    """Build a complete platform options string for SNPE.

    Combines PD type, burst mode timeout, and any extra options
    into the semicolon-separated format expected by SNPE.

    Args:
        pd_type: "unsigned" (default) or "signed"
        burst_config: Optional burst mode config (sets inactivityTimeout)
        extra_options: Additional key:value pairs to include

    Returns:
        Platform options string, e.g.:
            "unsignedPD:ON;inactivityTimeout:10000"

    Examples:
        build_platform_options()
        # → "unsignedPD:ON"

        build_platform_options(burst_config=BurstModeConfig(10))
        # → "unsignedPD:ON;inactivityTimeout:10000"

        build_platform_options(pd_type="signed", burst_config=BurstModeConfig(300))
        # → "unsignedPD:OFF;inactivityTimeout:300000"
    """
    parts: list[str] = []

    # PD type
    parts.append("unsignedPD:OFF" if pd_type == "signed" else "unsignedPD:ON")

    # Burst mode inactivity timeout (only if non-default or explicitly provided)
    if burst_config is not None:
        parts.append(burst_config.platform_option_string)

    # Extra options
    if extra_options:
        for k, v in extra_options.items():
            parts.append(f"{k}:{v}")

    return ";".join(parts)
