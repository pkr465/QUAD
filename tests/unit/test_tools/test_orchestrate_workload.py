"""Tests for the orchestrate_workload tool — focuses on edge-case handling.

The end-to-end happy paths are covered by tests/e2e/test_full_pipeline.py;
this file targets the failure modes the gap analysis flagged (T2.4 —
empty layers, degenerate allocations, etc.).
"""

from __future__ import annotations

import pytest

from quad.adapters.factory import AdapterFactory
from quad.exceptions import InvalidProfileError
from quad.models.config import ServerConfig
from quad.models.profiling import (
    LatencyStats,
    LayerProfile,
    ProfilingReport,
)
from quad.tools.orchestrate_workload import _profile_with_layers, orchestrate_workload_impl


def _make_report(*, layers: list[LayerProfile] | None = None) -> ProfilingReport:
    """Build a minimal ProfilingReport for testing."""
    from quad.models.device import DeviceProfile

    return ProfilingReport(
        latency=LatencyStats(mean_ms=10.0, p50_ms=9.5, p95_ms=15.0, p99_ms=20.0, min_ms=8.0, max_ms=25.0),
        throughput_fps=100.0,
        power_mw=1500.0,
        memory_peak_mb=80.0,
        memory_avg_mb=60.0,
        utilization={"cpu": 30.0, "gpu": 10.0, "npu": 60.0},
        layers=layers if layers is not None else [],
        device=DeviceProfile(
            chipset="test",
            platform="windows",
            cpu_cores=8,
            cpu_arch="ARM64",
            cpu_freq_ghz=3.0,
            gpu_model="test-gpu",
            gpu_tflops=2.0,
            npu_model="test-npu",
            npu_tops=20.0,
            ram_gb=16.0,
            sdk_path="",
            sdk_version="",
            available_runtimes=["cpu", "gpu", "npu"],
        ),
        runtime_used="npu",
        duration_s=5.0,
        profiling_level="detailed",
    )


class _FakeAdapter:
    """Adapter stub that returns scripted profiles per call."""

    def __init__(self, reports: list[ProfilingReport], supported_ops: list[str] | None = None):
        self._reports = list(reports)
        self._supported_ops = supported_ops or ["conv", "relu", "add", "matmul"]
        self.profile_calls: list[str] = []

    async def profile(self, request):  # type: ignore[no-untyped-def]
        level = getattr(request, "profiling_level", "detailed")
        self.profile_calls.append(level)
        if not self._reports:
            return _make_report(layers=[])
        return self._reports.pop(0)

    async def get_supported_ops(self) -> list[str]:
        return self._supported_ops


# ─── _profile_with_layers ────────────────────────────────────────────────────


class TestProfileWithLayers:
    @pytest.mark.asyncio
    async def test_returns_first_profile_when_layers_present(self) -> None:
        layers = [LayerProfile(name="conv1", op_type="conv", runtime="npu", latency_ms=2.0, memory_mb=5.0)]
        adapter = _FakeAdapter([_make_report(layers=layers)])
        result = await _profile_with_layers(adapter, "model.dlc")
        assert result.layers == layers
        # Should NOT have re-profiled
        assert len(adapter.profile_calls) == 1

    @pytest.mark.asyncio
    async def test_reprofiles_when_first_returns_no_layers(self) -> None:
        # First call returns empty layers (linting profile), second call
        # (forced detailed) returns the real layer list.
        first = _make_report(layers=[])
        second_layers = [
            LayerProfile(name="conv1", op_type="conv", runtime="npu", latency_ms=2.0, memory_mb=5.0),
            LayerProfile(name="relu1", op_type="relu", runtime="npu", latency_ms=0.5, memory_mb=1.0),
        ]
        second = _make_report(layers=second_layers)
        adapter = _FakeAdapter([first, second])

        result = await _profile_with_layers(adapter, "model.dlc")

        assert result.layers == second_layers
        # Should have re-profiled in 'detailed' mode
        assert len(adapter.profile_calls) == 2
        assert adapter.profile_calls[1] == "detailed"


# ─── orchestrate_workload_impl ───────────────────────────────────────────────


class TestOrchestrateEmptyLayers:
    """T2.4: orchestrate_workload should not produce a degenerate
    allocation when the upstream profile has no layers."""

    @pytest.mark.asyncio
    async def test_raises_invalid_profile_when_layers_persistently_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If even the re-profile returns no layers, raise a clear error."""
        empty_report = _make_report(layers=[])
        adapter = _FakeAdapter([empty_report, empty_report])

        # Patch the factory to return our fake adapter
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        monkeypatch.setattr(factory, "get_adapter", lambda sdk="auto": adapter)

        with pytest.raises(InvalidProfileError) as exc:
            await orchestrate_workload_impl("model.dlc", "balanced", factory)

        assert "no per-layer data" in str(exc.value)
        assert "detailed" in str(exc.value)

    @pytest.mark.asyncio
    async def test_succeeds_when_reprofile_recovers_layers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty layers on first profile + populated on second should still allocate."""
        layers = [
            LayerProfile(name="conv1", op_type="conv", runtime="npu", latency_ms=2.0, memory_mb=5.0),
            LayerProfile(name="relu1", op_type="relu", runtime="npu", latency_ms=0.5, memory_mb=1.0),
        ]
        adapter = _FakeAdapter([_make_report(layers=[]), _make_report(layers=layers)])

        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        monkeypatch.setattr(factory, "get_adapter", lambda sdk="auto": adapter)

        result = await orchestrate_workload_impl("model.dlc", "performance", factory)

        assert "allocation" in result
        assert result["allocation"]["conv1"] == "npu"
        assert result["allocation"]["relu1"] == "npu"
        assert result["npu_utilization_pct"] == 100.0


class TestOrchestratePowerModes:
    """Smoke tests covering each of the three power modes."""

    @pytest.mark.asyncio
    async def test_performance_mode_pushes_everything_to_npu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        layers = [
            LayerProfile(name="conv1", op_type="conv", runtime="npu", latency_ms=2.0, memory_mb=5.0),
            LayerProfile(name="conv2", op_type="conv", runtime="npu", latency_ms=0.1, memory_mb=1.0),
        ]
        adapter = _FakeAdapter([_make_report(layers=layers)])
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        monkeypatch.setattr(factory, "get_adapter", lambda sdk="auto": adapter)

        result = await orchestrate_workload_impl("m.dlc", "performance", factory)
        assert result["allocation"]["conv1"] == "npu"
        assert result["allocation"]["conv2"] == "npu"

    @pytest.mark.asyncio
    async def test_efficiency_mode_keeps_light_layers_on_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # mean_ms=10.0 across 2 layers → avg_latency=5.0
        # conv1 (8.0 ms) > avg → NPU
        # conv2 (0.1 ms) < avg → CPU
        layers = [
            LayerProfile(name="conv1", op_type="conv", runtime="npu", latency_ms=8.0, memory_mb=5.0),
            LayerProfile(name="conv2", op_type="conv", runtime="npu", latency_ms=0.1, memory_mb=1.0),
        ]
        adapter = _FakeAdapter([_make_report(layers=layers)])
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        monkeypatch.setattr(factory, "get_adapter", lambda sdk="auto": adapter)

        result = await orchestrate_workload_impl("m.dlc", "efficiency", factory)
        assert result["allocation"]["conv1"] == "npu"
        assert result["allocation"]["conv2"] == "cpu"

    @pytest.mark.asyncio
    async def test_unsupported_op_falls_back_to_cpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        layers = [
            LayerProfile(name="x1", op_type="custom_op", runtime="npu", latency_ms=2.0, memory_mb=5.0),
        ]
        adapter = _FakeAdapter([_make_report(layers=layers)], supported_ops=["conv", "relu"])
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        monkeypatch.setattr(factory, "get_adapter", lambda sdk="auto": adapter)

        result = await orchestrate_workload_impl("m.dlc", "balanced", factory)
        assert result["allocation"]["x1"] == "cpu"
        assert "x1" in result["fallback_layers"]
