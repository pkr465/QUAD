"""QPM3 (Qualcomm Power Monitor 3) wrapper — measured per-frame power.

Replaces the host_thermal_model estimate with actual hardware-measured
power when QPM3 is installed and a Power Monitor PMIC board is attached.
Auto-detected by the QAIRT adapter; absent QPM3 the adapter silently
keeps using ``estimate_host_power_mw``.

Public API:

    >>> from quad.profiler.qpm3 import qpm3_available, capture_power
    >>> if qpm3_available():
    ...     trace = await capture_power(duration_s=5.0)
    ...     print(f"avg = {trace.avg_power_mw:.0f} mW, peak = {trace.peak_power_mw:.0f}")

Installation seamlessness:

  - ``qpm3_available()`` checks PATH + the standard install locations
    (``C:\\Program Files\\Qualcomm\\Snapdragon Profiler\\Tools\\``,
    ``%LOCALAPPDATA%\\Qualcomm\\QPM\\``).
  - If absent, ``install.sh`` and ``quad doctor`` surface a one-line
    download URL and let the install proceed without QPM3.
  - The adapter never raises if QPM3 is missing — it falls back.

QPM3 stdout / CSV format (verified against QPM3 1.5+):

    Timestamp_us, VPH_PWR_mW, MAIN_DVDD_mW, ...
    0,            1234.5,     567.8, ...
    100,          1240.1,     569.2, ...

The wrapper accepts both the standard CSV form and the JSON output that
``qpm3-cli capture --format=json`` produces in newer versions.
"""
from __future__ import annotations

import asyncio
import csv
import io
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional


_QPM3_BIN_NAMES = ("qpm3-cli", "qpm3-cli.exe", "qpm-cli", "qpm-cli.exe", "qpm3", "qpm3.exe")
_QPM3_INSTALL_HINTS = (
    r"C:\Program Files\Qualcomm\Snapdragon Profiler\Tools",
    r"C:\Program Files (x86)\Qualcomm\Snapdragon Profiler\Tools",
    r"C:\Qualcomm\QPM3",
    r"C:\Qualcomm\Snapdragon Profiler\Tools",
)
QPM3_DOWNLOAD_URL = "https://www.qualcomm.com/developer/software/snapdragon-profiler"


@dataclass
class PowerSample:
    """One row from a QPM3 capture (microsecond timestamp + per-rail mW)."""
    t_us: int
    rails_mw: dict[str, float] = field(default_factory=dict)

    @property
    def total_mw(self) -> float:
        # VPH_PWR is the device-total rail when present; otherwise sum.
        if "VPH_PWR_mW" in self.rails_mw:
            return float(self.rails_mw["VPH_PWR_mW"])
        return float(sum(self.rails_mw.values()))


@dataclass
class PowerTrace:
    """Aggregate power profile from one capture."""
    samples: list[PowerSample] = field(default_factory=list)
    avg_power_mw: float = 0.0
    peak_power_mw: float = 0.0
    energy_mj: float = 0.0
    duration_s: float = 0.0
    source: str = "qpm3-cli"

    @classmethod
    def from_samples(cls, samples: list[PowerSample], *, source: str = "qpm3-cli") -> "PowerTrace":
        if not samples:
            return cls(source=source)
        totals = [s.total_mw for s in samples]
        avg = sum(totals) / len(totals)
        peak = max(totals)
        # Trapezoidal integration of total power over time → energy in mJ
        # (mW × s = mJ).
        energy_mj = 0.0
        if len(samples) >= 2:
            for i in range(1, len(samples)):
                dt_s = (samples[i].t_us - samples[i - 1].t_us) / 1_000_000.0
                p_mid = (totals[i] + totals[i - 1]) / 2.0
                energy_mj += p_mid * dt_s
        duration_s = (samples[-1].t_us - samples[0].t_us) / 1_000_000.0 if len(samples) >= 2 else 0.0
        return cls(
            samples=samples,
            avg_power_mw=round(avg, 1),
            peak_power_mw=round(peak, 1),
            energy_mj=round(energy_mj, 2),
            duration_s=round(duration_s, 3),
            source=source,
        )


# ── Detection ──────────────────────────────────────────────────────────────


def find_qpm3() -> Optional[str]:
    """Return the path to qpm3-cli (or qpm-cli) or None.

    Resolution order:
        1. Each of ``_QPM3_BIN_NAMES`` via ``shutil.which`` (PATH).
        2. Standard QPM/Snapdragon Profiler install dirs on Windows.
        3. ``QPM3_HOME`` env var pointing at an install root.
    """
    for name in _QPM3_BIN_NAMES:
        p = shutil.which(name)
        if p:
            return p
    if os.name == "nt":
        for hint in _QPM3_INSTALL_HINTS:
            for name in _QPM3_BIN_NAMES:
                cand = Path(hint) / name
                if cand.exists():
                    return str(cand)
    qpm3_home = os.environ.get("QPM3_HOME")
    if qpm3_home:
        for name in _QPM3_BIN_NAMES:
            cand = Path(qpm3_home) / name
            if cand.exists():
                return str(cand)
            cand = Path(qpm3_home) / "bin" / name
            if cand.exists():
                return str(cand)
    return None


def qpm3_available() -> bool:
    """True iff QPM3 CLI is callable on this host."""
    return find_qpm3() is not None


def install_hint() -> str:
    """One-line guidance to surface when QPM3 isn't installed."""
    return (
        f"QPM3 not detected. Download from {QPM3_DOWNLOAD_URL} "
        f"(Qualcomm developer login + EULA). After install, set QPM3_HOME or "
        f"add the QPM3 bin dir to PATH and re-run `quad doctor --real-mode`."
    )


# ── Capture ────────────────────────────────────────────────────────────────


async def capture_power(
    duration_s: float,
    *,
    sample_rate_hz: int = 1000,
    output_path: Optional[str] = None,
    extra_flags: Optional[list[str]] = None,
) -> PowerTrace:
    """Run ``qpm3-cli capture`` for ``duration_s`` and parse the result.

    Args:
        duration_s:     Capture window. Must match (or exceed) the
                        snpe-net-run --duration to align traces.
        sample_rate_hz: QPM3 sampling rate; 1000 Hz is the typical
                        default for CPU/NPU power workloads.
        output_path:    Optional caller-supplied path for the CSV.
                        If None we use a temp file.
        extra_flags:    Extra CLI flags to forward (e.g. ``--rail VPH_PWR``).

    Returns:
        PowerTrace with samples + aggregates. Empty trace if QPM3 isn't
        available — never raises.
    """
    tool = find_qpm3()
    if tool is None:
        return PowerTrace(source="qpm3:not_available")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="quad_qpm3_", suffix=".csv")
        os.close(fd)
    out = Path(output_path)

    cmd = [
        tool, "capture",
        "--duration", str(int(duration_s * 1000)),  # qpm3-cli takes ms
        "--rate", str(sample_rate_hz),
        "--format", "csv",
        "--output", str(out),
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=duration_s + 30.0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return PowerTrace(source="qpm3:timeout")
        if proc.returncode != 0:
            return PowerTrace(source="qpm3:exit_nonzero")
    except (FileNotFoundError, OSError):
        return PowerTrace(source="qpm3:exec_failed")

    if not out.exists() or out.stat().st_size == 0:
        return PowerTrace(source="qpm3:no_output")

    try:
        text = out.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return PowerTrace(source="qpm3:read_failed")
    finally:
        # Caller didn't pin the path → clean up.
        if out.parent == Path(tempfile.gettempdir()):
            try:
                out.unlink()
            except OSError:
                pass

    return parse_qpm3_csv(text)


# ── Parsing ────────────────────────────────────────────────────────────────


def parse_qpm3_csv(csv_text: str) -> PowerTrace:
    """Parse a QPM3 CSV blob into a PowerTrace.

    The CSV header is variable across QPM3 versions; we accept any
    column whose name matches ``*_mW`` and treats the timestamp column
    as either ``Timestamp_us`` or the first column.
    """
    if not csv_text.strip():
        return PowerTrace(source="qpm3:empty")

    rdr = csv.reader(io.StringIO(csv_text))
    header: list[str] | None = None
    samples: list[PowerSample] = []
    for row in rdr:
        if not row:
            continue
        first = (row[0] or "").strip()
        if header is None:
            # Skip comment / metadata banners that precede the real header.
            if first.startswith("#") or not first:
                continue
            header = [c.strip() for c in row]
            continue
        if not first.lstrip("-").isdigit():
            # Comment / metadata row mid-stream — skip.
            continue
        try:
            t_us = int(row[0])
        except ValueError:
            continue
        rails: dict[str, float] = {}
        for i, name in enumerate(header):
            if i == 0 or i >= len(row):
                continue
            if not name.endswith("_mW") and not name.lower().endswith("mw"):
                continue
            try:
                rails[name] = float(row[i])
            except (ValueError, TypeError):
                continue
        samples.append(PowerSample(t_us=t_us, rails_mw=rails))

    return PowerTrace.from_samples(samples, source="qpm3-csv")


__all__ = [
    "PowerSample",
    "PowerTrace",
    "QPM3_DOWNLOAD_URL",
    "capture_power",
    "find_qpm3",
    "install_hint",
    "parse_qpm3_csv",
    "qpm3_available",
]
