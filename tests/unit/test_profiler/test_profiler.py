"""Tests for QUAD Deep Profiler components."""

import time

import pytest

from quad.profiler.system import SystemProfiler, SystemTrace, TraceEvent
from quad.profiler.kernel import KernelProfiler, KernelReport, KernelMetrics
from quad.profiler.roofline import RooflineAnalysis, RooflineResult
from quad.profiler.power_profiler import PowerProfiler, PowerTrace, PowerSample, BatteryImpact
from quad.profiler.memory_profiler import MemoryProfiler, MemoryReport, AllocationEvent
from quad.profiler.api import profile_model, ProfileSummary


# =============================================================================
# SystemProfiler Tests
# =============================================================================


class TestSystemProfiler:
    """Tests for the system-level profiler."""

    def test_start_stop(self):
        profiler = SystemProfiler(mock=True)
        profiler.start()
        time.sleep(0.01)
        trace = profiler.stop()

        assert isinstance(trace, SystemTrace)
        assert trace.total_duration_ms > 0
        assert len(trace.events) > 0

    def test_context_manager(self):
        with SystemProfiler(mock=True) as profiler:
            time.sleep(0.01)

        trace = profiler.trace
        assert trace is not None
        assert isinstance(trace, SystemTrace)
        assert len(trace.events) > 0

    def test_trace_events_have_required_fields(self):
        profiler = SystemProfiler(mock=True)
        profiler.start()
        trace = profiler.stop()

        for event in trace.events:
            assert isinstance(event, TraceEvent)
            assert event.timestamp_us >= 0
            assert event.duration_us > 0
            assert event.device in ("cpu", "gpu", "npu", "dma")
            assert event.name != ""
            assert event.category in ("compute", "transfer", "sync", "idle")

    def test_idle_percentages(self):
        profiler = SystemProfiler(mock=True)
        profiler.start()
        trace = profiler.stop()

        assert "cpu" in trace.idle_pct
        assert "gpu" in trace.idle_pct
        assert "npu" in trace.idle_pct
        assert "dma" in trace.idle_pct

        for device, pct in trace.idle_pct.items():
            assert 0 <= pct <= 100

    def test_dma_stall(self):
        profiler = SystemProfiler(mock=True)
        profiler.start()
        trace = profiler.stop()

        assert trace.dma_stall_ms >= 0

    def test_events_sorted_by_timestamp(self):
        profiler = SystemProfiler(mock=True)
        profiler.start()
        trace = profiler.stop()

        timestamps = [e.timestamp_us for e in trace.events]
        assert timestamps == sorted(timestamps)

    def test_stop_without_start_raises(self):
        profiler = SystemProfiler(mock=True)
        with pytest.raises(RuntimeError):
            profiler.stop()

    def test_device_events_filter(self):
        profiler = SystemProfiler(mock=True)
        profiler.start()
        trace = profiler.stop()

        npu_events = trace.device_events("npu")
        assert all(e.device == "npu" for e in npu_events)

    def test_timeline_summary(self):
        profiler = SystemProfiler(mock=True)
        profiler.start()
        trace = profiler.stop()

        summary = trace.timeline_summary()
        assert isinstance(summary, dict)
        # At least some device should have time
        assert sum(summary.values()) > 0


# =============================================================================
# KernelProfiler Tests
# =============================================================================


class TestKernelProfiler:
    """Tests for the kernel-level profiler."""

    def test_profile_returns_report(self):
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()

        assert isinstance(report, KernelReport)
        assert len(report.kernels) > 0

    def test_kernel_metrics_fields(self):
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()

        for kernel in report.kernels:
            assert isinstance(kernel, KernelMetrics)
            assert kernel.name != ""
            assert kernel.op_type != ""
            assert kernel.latency_us > 0
            assert 0 <= kernel.compute_utilization_pct <= 100
            assert kernel.memory_bandwidth_gb_s > 0
            assert kernel.arithmetic_intensity > 0
            assert kernel.bottleneck in ("compute", "memory", "latency")

    def test_top_kernels(self):
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()

        top5 = report.top_kernels(5)
        assert len(top5) <= 5
        # Verify sorted descending by latency
        for i in range(len(top5) - 1):
            assert top5[i].latency_us >= top5[i + 1].latency_us

    def test_top_kernels_custom_n(self):
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()

        top3 = report.top_kernels(3)
        assert len(top3) <= 3

    def test_total_latency(self):
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()

        expected = sum(k.latency_us for k in report.kernels)
        assert abs(report.total_latency_us - expected) < 0.01

    def test_bottleneck_summary(self):
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()

        summary = report.bottleneck_summary()
        assert "compute" in summary
        assert "memory" in summary
        assert "latency" in summary
        assert sum(summary.values()) == len(report.kernels)

    def test_different_devices(self):
        for device in ("npu", "gpu", "cpu"):
            profiler = KernelProfiler(mock=True, device=device)
            report = profiler.profile()
            assert len(report.kernels) > 0


# =============================================================================
# RooflineAnalysis Tests
# =============================================================================


class TestRooflineAnalysis:
    """Tests for the roofline model."""

    def test_basic_analysis(self):
        roofline = RooflineAnalysis(device_peak_tops=73.0, device_bandwidth_gb_s=68.0)
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()

        result = roofline.analyze(report.kernels)

        assert isinstance(result, RooflineResult)
        assert result.peak_tops == 73.0
        assert result.peak_bandwidth_gb_s == 68.0
        assert result.ridge_point > 0
        assert result.achieved_tops >= 0
        assert 0 <= result.achieved_pct <= 100
        assert result.diagnosis in ("compute-bound", "memory-bound")
        assert result.recommendation != ""

    def test_ridge_point_calculation(self):
        roofline = RooflineAnalysis(device_peak_tops=73.0, device_bandwidth_gb_s=68.0)
        # Ridge = (73 * 1000) / 68 = ~1073.5 ops/byte
        expected_ridge = (73.0 * 1000.0) / 68.0
        assert abs(roofline.ridge_point - expected_ridge) < 0.01

    def test_empty_kernels(self):
        roofline = RooflineAnalysis()
        result = roofline.analyze([])

        assert result.achieved_tops == 0.0
        assert result.achieved_pct == 0.0
        assert result.diagnosis == "memory-bound"

    def test_custom_device_specs(self):
        roofline = RooflineAnalysis(device_peak_tops=10.0, device_bandwidth_gb_s=30.0)
        assert roofline.device_peak_tops == 10.0
        assert roofline.device_bandwidth_gb_s == 30.0

    def test_recommendation_not_empty(self):
        roofline = RooflineAnalysis()
        profiler = KernelProfiler(mock=True)
        report = profiler.profile()
        result = roofline.analyze(report.kernels)

        assert len(result.recommendation) > 10


# =============================================================================
# PowerProfiler Tests
# =============================================================================


class TestPowerProfiler:
    """Tests for the power profiler."""

    def test_start_stop(self):
        profiler = PowerProfiler(mock=True)
        profiler.start()
        time.sleep(0.02)
        trace = profiler.stop()

        assert isinstance(trace, PowerTrace)
        assert len(trace.samples) > 0
        assert trace.avg_power_mw > 0
        assert trace.peak_power_mw >= trace.avg_power_mw
        assert trace.energy_mj > 0

    def test_context_manager(self):
        with PowerProfiler(mock=True) as profiler:
            time.sleep(0.02)

        trace = profiler.trace
        assert trace is not None
        assert len(trace.samples) > 0

    def test_power_samples_fields(self):
        profiler = PowerProfiler(mock=True)
        profiler.start()
        time.sleep(0.01)
        trace = profiler.stop()

        for sample in trace.samples:
            assert isinstance(sample, PowerSample)
            assert sample.timestamp_ms >= 0
            assert sample.npu_mw > 0
            assert sample.gpu_mw > 0
            assert sample.cpu_mw > 0
            assert sample.total_mw > 0
            # Total should be sum of components
            expected_total = sample.npu_mw + sample.gpu_mw + sample.cpu_mw
            assert abs(sample.total_mw - expected_total) < 0.2

    def test_breakdown_pct(self):
        profiler = PowerProfiler(mock=True)
        profiler.start()
        time.sleep(0.01)
        trace = profiler.stop()

        assert "npu" in trace.breakdown_pct
        assert "gpu" in trace.breakdown_pct
        assert "cpu" in trace.breakdown_pct

        total_pct = sum(trace.breakdown_pct.values())
        assert abs(total_pct - 100.0) < 1.0  # Should sum to ~100%

    def test_battery_impact(self):
        profiler = PowerProfiler(mock=True)
        profiler.start()
        time.sleep(0.01)
        trace = profiler.stop()

        impact = trace.battery_impact(battery_mah=5000, voltage=3.85)

        assert isinstance(impact, BatteryImpact)
        assert impact.hours_at_workload > 0
        assert impact.drain_pct_per_hour > 0
        assert impact.battery_mah == 5000
        assert impact.voltage == 3.85

    def test_thermal_headroom(self):
        profiler = PowerProfiler(mock=True)
        profiler.start()
        time.sleep(0.01)
        trace = profiler.stop()

        assert 0 <= trace.thermal_headroom_pct <= 100

    def test_stop_without_start_raises(self):
        profiler = PowerProfiler(mock=True)
        with pytest.raises(RuntimeError):
            profiler.stop()


# =============================================================================
# MemoryProfiler Tests
# =============================================================================


class TestMemoryProfiler:
    """Tests for the memory profiler."""

    def test_profile_returns_report(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        assert isinstance(report, MemoryReport)
        assert report.peak_mb > 0
        assert report.avg_mb > 0
        assert len(report.allocations) > 0

    def test_allocation_events_fields(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        for alloc in report.allocations:
            assert isinstance(alloc, AllocationEvent)
            assert alloc.timestamp_ms >= 0
            assert alloc.size_mb > 0
            assert alloc.device in ("vtcm", "ddr", "system")
            assert alloc.name != ""

    def test_vtcm_utilization(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        assert 0 <= report.vtcm_utilization_pct <= 100

    def test_ddr_bandwidth(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        assert report.ddr_bandwidth_gb_s > 0

    def test_fragmentation(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        assert 0 <= report.fragmentation_pct <= 100

    def test_reuse_efficiency(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        assert 0 <= report.reuse_efficiency_pct <= 100

    def test_vtcm_allocations_filter(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        vtcm_allocs = report.vtcm_allocations()
        assert all(a.device == "vtcm" for a in vtcm_allocs)

    def test_ddr_allocations_filter(self):
        profiler = MemoryProfiler(mock=True)
        report = profiler.profile()

        ddr_allocs = report.ddr_allocations()
        assert all(a.device == "ddr" for a in ddr_allocs)


# =============================================================================
# profile_model API Tests
# =============================================================================


class TestProfileModelAPI:
    """Tests for the high-level profile_model API."""

    def test_system_level(self):
        summary = profile_model("dummy_model.onnx", level="system")

        assert isinstance(summary, ProfileSummary)
        assert summary.system_trace is not None
        assert summary.kernel_report is None
        assert summary.roofline is None
        assert summary.power_trace is None
        assert summary.memory_report is None

    def test_kernel_level(self):
        summary = profile_model("dummy_model.onnx", level="kernel")

        assert isinstance(summary, ProfileSummary)
        assert summary.system_trace is not None
        assert summary.kernel_report is not None
        assert summary.roofline is not None
        assert summary.power_trace is None
        assert summary.memory_report is None

    def test_deep_level(self):
        summary = profile_model("dummy_model.onnx", level="deep")

        assert isinstance(summary, ProfileSummary)
        assert summary.system_trace is not None
        assert summary.kernel_report is not None
        assert summary.roofline is not None
        assert summary.power_trace is not None
        assert summary.memory_report is not None

    def test_recommendations_generated(self):
        summary = profile_model("dummy_model.onnx", level="deep")

        assert len(summary.recommendations) > 0
        for rec in summary.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 10

    def test_profile_duration_tracked(self):
        summary = profile_model("dummy_model.onnx", level="system")

        assert summary.profile_duration_ms > 0

    def test_different_devices(self):
        for device in ("npu", "gpu", "cpu"):
            summary = profile_model("dummy_model.onnx", level="kernel", device=device)
            assert summary.kernel_report is not None

    def test_repr(self):
        summary = profile_model("dummy_model.onnx", level="deep")
        repr_str = repr(summary)
        assert "ProfileSummary" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestProfilerIntegration:
    """Integration tests combining multiple profiler components."""

    def test_kernel_to_roofline_pipeline(self):
        """Kernel profiler output feeds directly into roofline analysis."""
        kernel_profiler = KernelProfiler(mock=True)
        report = kernel_profiler.profile()

        roofline = RooflineAnalysis(device_peak_tops=73.0, device_bandwidth_gb_s=68.0)
        result = roofline.analyze(report.kernels)

        assert result.diagnosis in ("compute-bound", "memory-bound")
        assert result.achieved_pct > 0

    def test_full_profiling_session(self):
        """Simulate a complete profiling session."""
        # 1. System trace
        with SystemProfiler(mock=True) as sys_prof:
            time.sleep(0.01)
        sys_trace = sys_prof.trace

        # 2. Kernel profiling
        kernel_prof = KernelProfiler(mock=True, device="npu")
        kernel_report = kernel_prof.profile()

        # 3. Power measurement
        with PowerProfiler(mock=True) as power_prof:
            time.sleep(0.01)
        power_trace = power_prof.trace

        # 4. Memory analysis
        mem_prof = MemoryProfiler(mock=True)
        mem_report = mem_prof.profile()

        # 5. Roofline
        roofline = RooflineAnalysis()
        roofline_result = roofline.analyze(kernel_report.kernels)

        # Verify all components produced valid results
        assert sys_trace.event_count > 0
        assert kernel_report.total_latency_us > 0
        assert power_trace.energy_mj > 0
        assert mem_report.peak_mb > 0
        assert roofline_result.recommendation != ""

    def test_battery_impact_realistic(self):
        """Battery impact numbers should be realistic for mobile."""
        with PowerProfiler(mock=True) as prof:
            time.sleep(0.02)

        trace = prof.trace
        impact = trace.battery_impact(battery_mah=5000, voltage=3.85)

        # A 5000mAh battery at ~4-8W should last roughly 2-5 hours
        assert 1.0 < impact.hours_at_workload < 20.0
        assert 5.0 < impact.drain_pct_per_hour < 100.0
