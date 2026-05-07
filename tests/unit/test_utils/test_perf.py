"""Tests for SNPE performance utilities."""

from __future__ import annotations

import pytest

from quad.utils.perf import (
    DSP_INIT_CACHE_REQUIRED_VERSION,
    GPU_OVERHEAD_MS_HIGH,
    GPU_OVERHEAD_MS_LOW,
    GPU_OVERHEAD_MS_TYPICAL,
    GPU_SPEEDUP_FACTOR_HIGH,
    GPU_SPEEDUP_FACTOR_LOW,
    DSPGuidance,
    PerformanceProfile,
    ProfilingLevel,
    RuntimeRecommendation,
    get_dsp_guidance,
    get_profiling_recommendation,
    recommend_runtime,
)


class TestPerformanceProfile:
    def test_all_profiles_exist(self) -> None:
        assert PerformanceProfile.BALANCED
        assert PerformanceProfile.HIGH_PERFORMANCE
        assert PerformanceProfile.POWER_SAVER
        assert PerformanceProfile.SYSTEM_SETTINGS
        assert PerformanceProfile.BURST

    def test_default_is_deprecated(self) -> None:
        assert PerformanceProfile.DEFAULT.is_deprecated is True

    def test_others_not_deprecated(self) -> None:
        for p in (PerformanceProfile.BALANCED, PerformanceProfile.HIGH_PERFORMANCE,
                  PerformanceProfile.BURST, PerformanceProfile.POWER_SAVER):
            assert p.is_deprecated is False

    def test_snpe_enum_names(self) -> None:
        assert PerformanceProfile.HIGH_PERFORMANCE.snpe_enum_name == "HIGH_PERFORMANCE"
        assert PerformanceProfile.BALANCED.snpe_enum_name == "BALANCED"
        assert PerformanceProfile.POWER_SAVER.snpe_enum_name == "POWER_SAVER"

    def test_power_levels(self) -> None:
        assert "high" in PerformanceProfile.HIGH_PERFORMANCE.power_level
        assert "low" in PerformanceProfile.POWER_SAVER.power_level
        assert "moderate" in PerformanceProfile.BALANCED.power_level

    def test_profile_has_description(self) -> None:
        for p in PerformanceProfile:
            assert len(p.description) > 0


class TestGPUConstants:
    def test_gpu_overhead_range(self) -> None:
        """GPU overhead 4-6ms per SNPE docs."""
        assert GPU_OVERHEAD_MS_LOW == 4.0
        assert GPU_OVERHEAD_MS_HIGH == 6.0
        assert GPU_OVERHEAD_MS_LOW < GPU_OVERHEAD_MS_TYPICAL < GPU_OVERHEAD_MS_HIGH

    def test_gpu_speedup_range(self) -> None:
        """GPU 6-10x faster than CPU per SNPE docs."""
        assert GPU_SPEEDUP_FACTOR_LOW == 6.0
        assert GPU_SPEEDUP_FACTOR_HIGH == 10.0


class TestRuntimeRecommendation:
    def test_npu_preferred_when_available(self) -> None:
        rec = recommend_runtime(cpu_latency_ms=50.0, npu_available=True)
        assert rec.recommended_runtime == "npu"

    def test_gpu_recommended_for_long_networks(self) -> None:
        """Long network (100ms CPU) → GPU is beneficial despite overhead."""
        rec = recommend_runtime(
            cpu_latency_ms=100.0,
            npu_available=False,
            gpu_available=True,
        )
        assert rec.recommended_runtime == "gpu"
        assert rec.estimated_latency_ms < 100.0  # GPU faster

    def test_cpu_for_very_short_networks(self) -> None:
        """Short network: GPU overhead (5ms) may eliminate GPU advantage.
        CPU 10ms / 8x speedup = 1.25ms + 5ms overhead = 6.25ms GPU
        GPU is still faster here, but warning should mention threshold.
        """
        rec = recommend_runtime(
            cpu_latency_ms=10.0,
            npu_available=False,
            gpu_available=True,
        )
        # GPU might still win but should warn about overhead threshold
        if rec.recommended_runtime == "gpu":
            assert any("10ms" in w or "overhead" in w for w in rec.warnings)

    def test_cpu_when_gpu_busy(self) -> None:
        """If GPU is at >50% utilization (e.g. gaming), recommend CPU."""
        rec = recommend_runtime(
            cpu_latency_ms=50.0,
            npu_available=False,
            gpu_available=True,
            gpu_utilization_pct=80.0,
        )
        assert rec.recommended_runtime == "cpu"
        assert any("utilization" in w.lower() or "gaming" in w.lower() or "80" in w
                   for w in rec.warnings)

    def test_estimated_latency_always_positive(self) -> None:
        for latency in (5.0, 50.0, 200.0):
            rec = recommend_runtime(latency, npu_available=True)
            assert rec.estimated_latency_ms > 0

    def test_alternatives_not_empty(self) -> None:
        rec = recommend_runtime(50.0, npu_available=True)
        assert len(rec.alternatives) > 0

    def test_reason_always_provided(self) -> None:
        rec = recommend_runtime(50.0, npu_available=True)
        assert len(rec.reason) > 0


class TestDSPGuidance:
    def test_v68_plus_recommends_init_cache(self) -> None:
        """DSP V68+ init times are much longer — init cache strongly recommended."""
        for v in ("v68", "v69", "v73", "v75", "v79"):
            guidance = get_dsp_guidance(v)
            assert guidance.use_init_cache is True, f"{v} should recommend init cache"

    def test_pre_v68_no_init_cache_requirement(self) -> None:
        for v in ("v65", "v66"):
            guidance = get_dsp_guidance(v)
            assert guidance.use_init_cache is False

    def test_preprocessing_warning_when_layers_present(self) -> None:
        guidance = get_dsp_guidance("v73", model_has_preprocessing_layers=True)
        assert guidance.preprocess_before_snpe is True
        assert "NOT optimized" in guidance.preprocess_reason or "not optimized" in guidance.preprocess_reason.lower()

    def test_quantization_warning_for_unquantized_model(self) -> None:
        guidance = get_dsp_guidance("v73", model_is_quantized=False)
        assert guidance.quantization_warning is not None
        assert "8-bit" in guidance.quantization_warning

    def test_quantized_model_no_warning(self) -> None:
        guidance = get_dsp_guidance("v73", model_is_quantized=True)
        assert guidance.quantization_warning is None

    def test_init_cache_constant_version(self) -> None:
        assert DSP_INIT_CACHE_REQUIRED_VERSION == "v68"


class TestProfilingLevel:
    def test_off_is_production_safe(self) -> None:
        assert ProfilingLevel.OFF.is_production_safe is True

    def test_basic_is_production_safe(self) -> None:
        assert ProfilingLevel.BASIC.is_production_safe is True

    def test_moderate_not_production_safe(self) -> None:
        assert ProfilingLevel.MODERATE.is_production_safe is False

    def test_detailed_not_production_safe(self) -> None:
        assert ProfilingLevel.DETAILED.is_production_safe is False

    def test_snpe_net_run_flag(self) -> None:
        assert ProfilingLevel.OFF.snpe_net_run_flag == "off"
        assert ProfilingLevel.DETAILED.snpe_net_run_flag == "detailed"


class TestProfilingRecommendation:
    def test_production_returns_off(self) -> None:
        assert get_profiling_recommendation(is_production=True) == ProfilingLevel.OFF

    def test_development_returns_detailed(self) -> None:
        assert get_profiling_recommendation(is_production=False) == ProfilingLevel.DETAILED


class TestStdCopyInTemplates:
    """Verify std::copy() is used (not iterator loops) in C++ template."""

    def test_cpp_template_uses_std_copy(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("inference.cpp.j2")
        rendered = t.render(model_path="model.dlc", runtime="cpu")
        # Must use std::copy, not manual iterator loops
        assert "std::copy(" in rendered
        # The slow iterator pattern should NOT be the primary copy method
        assert rendered.count("std::copy(") >= 2

    def test_c_template_uses_fread(self) -> None:
        """C template uses fread() for bulk data — no element-by-element loop."""
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        t = env.get_template("inference.c.j2")
        rendered = t.render(model_path="model.dlc", runtime="cpu")
        assert "fread(" in rendered
