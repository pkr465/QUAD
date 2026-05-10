"""Tests for profiler/host_power.py + profiler/host_utilization.py +
profiler/rss_sampler.py."""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

from quad.profiler.host_power import (
    SrumEnergyRow,
    estimate_host_power_mw,
    srumutil_available,
    _parse_srum_xml,
)
from quad.profiler.host_utilization import (
    cpu_percent_blocking,
    gpu_utilization_from_chrometrace,
    npu_utilization_from_cycles,
)
from quad.profiler.rss_sampler import RSSReport, run_with_rss_sampling


# ── host_power ──────────────────────────────────────────────────────────────


def test_estimate_power_idle_floor():
    """At zero utilisation we should still draw the idle-floor mW."""
    p = estimate_host_power_mw(cpu_pct=0, gpu_pct=0, npu_pct=0)
    assert p >= 1500.0
    assert p == 1500.0  # exactly the X Elite idle floor


def test_estimate_power_full_cpu_only():
    """100 % CPU on X Elite ≈ idle_floor + 18 W."""
    p = estimate_host_power_mw(cpu_pct=100, gpu_pct=0, npu_pct=0)
    assert p == pytest.approx(1500 + 18000, abs=1)


def test_estimate_power_clamps_above_100_pct():
    p_clamp = estimate_host_power_mw(cpu_pct=300, gpu_pct=0, npu_pct=0)
    p_max = estimate_host_power_mw(cpu_pct=100, gpu_pct=0, npu_pct=0)
    assert p_clamp == p_max


def test_estimate_power_negative_treated_as_zero():
    p = estimate_host_power_mw(cpu_pct=-50, gpu_pct=0, npu_pct=0)
    assert p == 1500.0


def test_estimate_power_custom_profile():
    custom = {"cpu_max_mw": 1000, "gpu_max_mw": 1000, "npu_max_mw": 1000, "idle_floor_mw": 500}
    p = estimate_host_power_mw(cpu_pct=50, gpu_pct=50, npu_pct=50, profile=custom)
    assert p == pytest.approx(500 + 0.5 * 1000 * 3, abs=1)


def test_srumutil_available_matches_platform():
    if os.name == "nt":
        # powercfg ships with every Win10+ host.
        import shutil
        assert srumutil_available() is bool(shutil.which("powercfg"))
    else:
        assert srumutil_available() is False


def test_parse_srum_xml_robust_to_empty(tmp_path):
    """Empty / malformed XML returns [] rather than raising."""
    junk = tmp_path / "junk.xml"
    junk.write_text("not xml")
    assert _parse_srum_xml(str(junk)) == []
    missing = tmp_path / "missing.xml"
    assert _parse_srum_xml(str(missing)) == []


def test_parse_srum_xml_extracts_energy_rows(tmp_path):
    """Synthetic SRUM-like XML — verify we extract Energy Estimation rows."""
    xml = """<?xml version='1.0'?>
<Root xmlns="urn:schemas-microsoft-com:srum">
  <EnergyEstimation>
    <AppId>!!Microsoft.PowerShell_pwsh.exe</AppId>
    <EnergyLoss>10000</EnergyLoss>
    <CpuTimeStateMS>1500</CpuTimeStateMS>
  </EnergyEstimation>
  <EnergyEstimation>
    <AppId>!!python.exe</AppId>
    <EnergyLoss>5000</EnergyLoss>
    <CpuTimeStateMS>800</CpuTimeStateMS>
  </EnergyEstimation>
</Root>"""
    p = tmp_path / "srum.xml"
    p.write_text(xml)
    rows = _parse_srum_xml(str(p))
    assert len(rows) == 2
    # Sorted by descending energy.
    assert rows[0].energy_uj == 10000
    assert rows[0].cpu_active_ms == 1500
    assert "pwsh" in rows[0].app
    assert rows[1].energy_uj == 5000


def test_parse_srum_xml_keyword_filter(tmp_path):
    xml = """<?xml version='1.0'?><Root>
  <EnergyEstimation><AppId>foo.exe</AppId><EnergyLoss>10</EnergyLoss></EnergyEstimation>
  <EnergyEstimation><AppId>python.exe</AppId><EnergyLoss>20</EnergyLoss></EnergyEstimation>
</Root>"""
    p = tmp_path / "srum.xml"
    p.write_text(xml)
    rows = _parse_srum_xml(str(p), keyword="python")
    assert len(rows) == 1
    assert "python" in rows[0].app


# ── host_utilization ───────────────────────────────────────────────────────


def test_cpu_percent_blocking_returns_float_in_range():
    pct = cpu_percent_blocking(interval_s=0.05)
    assert isinstance(pct, float)
    assert 0.0 <= pct <= 100.0


def test_npu_util_zero_inputs_zero_output():
    assert npu_utilization_from_cycles(total_cycles=0, wall_time_us=1000.0) == 0.0
    assert npu_utilization_from_cycles(total_cycles=1000, wall_time_us=0.0) == 0.0


def test_npu_util_v73_clock_arithmetic():
    """1.5 GHz Hexagon V73, 1.5 M cycles in 1 ms wall-time → 100 %."""
    pct = npu_utilization_from_cycles(
        total_cycles=1_500_000,
        wall_time_us=1_000.0,
        hexagon_arch="V73",
    )
    assert pct == pytest.approx(100.0, rel=1e-3)


def test_npu_util_unknown_arch_returns_zero():
    pct = npu_utilization_from_cycles(
        total_cycles=1_000_000, wall_time_us=1000.0, hexagon_arch="V99",
    )
    assert pct == 0.0


def test_npu_util_clamps_at_100():
    """Bogus inputs should never report >100 %."""
    pct = npu_utilization_from_cycles(
        total_cycles=999_999_999, wall_time_us=10.0, hexagon_arch="V73",
    )
    assert 0.0 <= pct <= 100.0


def test_gpu_chrometrace_missing_returns_zero():
    assert gpu_utilization_from_chrometrace("/no/such/path.json") == 0.0


def test_gpu_chrometrace_basic(tmp_path):
    """5 ms of GPU events over a 10 ms span → 50 %."""
    import json
    trace = {
        "traceEvents": [
            {"name": "fwd",  "cat": "GPU", "ph": "X", "ts": 0,    "dur": 5000},
            {"name": "free", "cat": "CPU", "ph": "X", "ts": 5000, "dur": 5000},
            {"name": "end",  "cat": "GPU", "ph": "X", "ts": 9000, "dur": 1000},
        ]
    }
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))
    pct = gpu_utilization_from_chrometrace(str(p))
    # 5000 us GPU + 1000 us GPU = 6000 us across span 0..10000 us = 60 %.
    assert pct == pytest.approx(60.0, rel=1e-3)


def test_gpu_chrometrace_malformed_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{")
    assert gpu_utilization_from_chrometrace(str(p)) == 0.0


# ── rss_sampler ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rss_sampling_short_subprocess():
    """A ~50 ms python -c sleep produces at least one RSS sample."""
    py = sys.executable
    completed, rss = await run_with_rss_sampling(
        [py, "-c", "import time; time.sleep(0.5)"],
        timeout=10.0,
        sample_interval_s=0.05,
    )
    assert completed.returncode == 0
    if rss.available:
        assert rss.peak_mb > 0
        assert rss.mean_mb > 0
        assert rss.samples >= 1
    else:
        # On boxes without psutil we still want the subprocess to succeed
        assert rss.reason


@pytest.mark.asyncio
async def test_rss_sampling_captures_stdout():
    py = sys.executable
    completed, _ = await run_with_rss_sampling(
        [py, "-c", "print('hello-rss')"],
        timeout=10.0,
        sample_interval_s=0.05,
    )
    assert "hello-rss" in completed.stdout


@pytest.mark.asyncio
async def test_rss_sampling_timeout_raises():
    py = sys.executable
    with pytest.raises(TimeoutError):
        await run_with_rss_sampling(
            [py, "-c", "import time; time.sleep(60)"],
            timeout=0.5,
            sample_interval_s=0.05,
        )


def test_rss_report_defaults():
    r = RSSReport()
    assert r.peak_mb == 0
    assert r.mean_mb == 0
    assert r.available is True
