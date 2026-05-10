"""Host-side power estimation — no QPM3 required.

Two paths:

1. **Estimated** (default, no extra hardware) —
   ``estimate_host_power_mw()`` combines CPU/GPU/NPU utilisation with
   per-component TDPs derived from the current SoC's power class.
   Honest about what it is: an *estimate*, tagged
   ``estimated:host_thermal_model`` in measurement_notes.

2. **Measured** (Windows ARM64, requires powercfg.exe — built-in) —
   ``read_srum_energy_estimation_uj()`` reads the most recent
   ``Energy Estimation`` row from Windows' System Resource Usage
   Monitor via ``powercfg /srumutil``. Per-process accuracy is
   coarse (~30-second granularity) but it's the highest-fidelity
   no-extra-tooling answer available on Snapdragon X Elite.

Anything more precise (per-frame, per-IP) needs QPM3 or Snapdragon
Profiler — tracked as the deferred Phase-4 work in the gap analysis.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


# Per-component thermal-design-power coefficients (mW). Tuned for the
# Snapdragon X Elite reference platform; over-estimates on lower SKUs
# and under-estimates on plug-in workstations — call it ±25 %.
_TDP_PROFILE_W_X_ELITE = {
    "cpu_max_mw":  18_000,   # 12 Oryon cores at burst
    "gpu_max_mw":   8_000,   # Adreno X1-85 boost
    "npu_max_mw":   6_000,   # Hexagon V73 sustained
    "idle_floor_mw": 1_500,  # display, RAM, package leakage
}


def estimate_host_power_mw(
    *,
    cpu_pct: float = 0.0,
    gpu_pct: float = 0.0,
    npu_pct: float = 0.0,
    profile: dict[str, float] | None = None,
) -> float:
    """Estimate the package power draw given utilisation percentages.

    Args:
        cpu_pct, gpu_pct, npu_pct: utilisation 0-100.
        profile: optional override of the TDP coefficients.

    Returns:
        Estimated power in mW. Always >= the idle floor of the profile.
    """
    p = profile or _TDP_PROFILE_W_X_ELITE
    cpu = max(0.0, min(cpu_pct, 100.0)) / 100.0
    gpu = max(0.0, min(gpu_pct, 100.0)) / 100.0
    npu = max(0.0, min(npu_pct, 100.0)) / 100.0
    dynamic = (
        cpu * p["cpu_max_mw"]
        + gpu * p["gpu_max_mw"]
        + npu * p["npu_max_mw"]
    )
    return round(p["idle_floor_mw"] + dynamic, 0)


# ── Windows SRUM (per-process energy estimation) ───────────────────────────


@dataclass
class SrumEnergyRow:
    """One row from `powercfg /srumutil --xml` Energy Estimation block."""
    app: str
    energy_uj: int
    cpu_active_ms: int
    network_kb: int


def srumutil_available() -> bool:
    """True iff ``powercfg.exe`` exists on PATH (Windows-only)."""
    return os.name == "nt" and shutil.which("powercfg") is not None


def read_srum_energy_estimation_uj(
    *,
    timeout: float = 30.0,
    keyword: str | None = None,
) -> list[SrumEnergyRow]:
    """Run ``powercfg /srumutil`` and return the Energy Estimation rows.

    SRUM updates ~every 30 seconds and tracks per-process kernel power
    counters. Granularity is too coarse for sub-second profiling, but
    perfect for the 5-second runs ``snpe-net-run`` performs by default.

    Args:
        timeout: Subprocess timeout (powercfg can be slow on first run).
        keyword: Optional case-insensitive substring filter on app name.

    Returns:
        Rows sorted by descending energy (most-power-hungry first).
        Empty list if powercfg isn't available or the call fails.
    """
    if not srumutil_available():
        return []
    out_dir = tempfile.mkdtemp(prefix="quad_srum_")
    out_xml = os.path.join(out_dir, "srum.xml")
    try:
        # /srumutil dumps the SRUM database. /xml chooses XML; otherwise
        # /csv is also available — XML is more robust to parse.
        proc = subprocess.run(
            ["powercfg", "/srumutil", "/output", out_xml, "/xml"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0 or not os.path.exists(out_xml):
            return []
        return _parse_srum_xml(out_xml, keyword=keyword)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    finally:
        try:
            os.unlink(out_xml)
            os.rmdir(out_dir)
        except OSError:
            pass


def _parse_srum_xml(xml_path: str, *, keyword: str | None = None) -> list[SrumEnergyRow]:
    """Parse Energy Estimation rows from a powercfg /srumutil XML dump.

    Robust to:
      - missing files (returns []),
      - unparseable XML (returns []),
      - default xmlns on the root (we strip the namespace before lookups).
    """
    import xml.etree.ElementTree as ET

    rows: list[SrumEnergyRow] = []
    try:
        tree = ET.parse(xml_path)
    except (FileNotFoundError, OSError, ET.ParseError):
        return []

    root = tree.getroot()

    def _local(child_tag_name: str, parent: ET.Element) -> str | None:
        """Find the first descendant whose local-name matches, namespace-agnostic."""
        for el in parent.iter():
            if el.tag.split("}")[-1] == child_tag_name:
                if el is not parent:
                    return (el.text or "").strip()
        return None

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag not in ("EnergyEstimation", "EnergyUsage"):
            continue
        app = (_local("AppId", elem) or elem.attrib.get("AppId") or "").strip()
        if keyword and keyword.lower() not in app.lower():
            continue
        try:
            energy = int(_local("EnergyLoss", elem) or elem.attrib.get("EnergyLoss") or "0")
        except (TypeError, ValueError):
            energy = 0
        try:
            cpu_ms = int(_local("CpuTimeStateMS", elem) or "0")
        except (TypeError, ValueError):
            cpu_ms = 0
        try:
            net_kb = int(_local("NetworkKB", elem) or "0")
        except (TypeError, ValueError):
            net_kb = 0
        if app:
            rows.append(SrumEnergyRow(
                app=app, energy_uj=energy,
                cpu_active_ms=cpu_ms, network_kb=net_kb,
            ))
    rows.sort(key=lambda r: r.energy_uj, reverse=True)
    return rows


__all__ = [
    "SrumEnergyRow",
    "estimate_host_power_mw",
    "read_srum_energy_estimation_uj",
    "srumutil_available",
]
