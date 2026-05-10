"""Snapdragon Profiler ``sdptrace`` wrapper — multi-IP system trace.

Captures full Perfetto / chrometrace JSON traces around an inference
window. Used to populate the ``utilization`` block (CPU + GPU + NPU
contention details) and to feed the existing
``gpu_utilization_from_chrometrace()`` parser.

Same seamless-setup contract as ``qpm3.py``: the adapter calls
``capture_trace_around()`` *iff* ``sdptrace_available()`` is true; if
the tool isn't installed we silently degrade to the CPU/NPU arithmetic
path. ``install.sh`` and ``quad doctor`` surface a download URL so the
user knows where to get it without breaking the install flow.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional


_SDPTRACE_BIN_NAMES = ("sdptrace", "sdptrace.exe", "sdptrace-cli", "sdptrace-cli.exe")
_SDPTRACE_INSTALL_HINTS = (
    r"C:\Program Files\Qualcomm\Snapdragon Profiler\Tools",
    r"C:\Program Files (x86)\Qualcomm\Snapdragon Profiler\Tools",
    r"C:\Qualcomm\Snapdragon Profiler\Tools",
    "/opt/qcom/snapdragon-profiler/tools",
)
SDPTRACE_DOWNLOAD_URL = "https://www.qualcomm.com/developer/software/snapdragon-profiler"


@dataclass
class TraceCapture:
    """Result of one sdptrace capture."""
    trace_path: Optional[Path] = None
    duration_s: float = 0.0
    success: bool = False
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.success and self.trace_path is not None and self.trace_path.exists()


# ── Detection ──────────────────────────────────────────────────────────────


def find_sdptrace() -> Optional[str]:
    """Return the path to ``sdptrace`` or None."""
    for name in _SDPTRACE_BIN_NAMES:
        p = shutil.which(name)
        if p:
            return p
    for hint in _SDPTRACE_INSTALL_HINTS:
        for name in _SDPTRACE_BIN_NAMES:
            cand = Path(hint) / name
            if cand.exists():
                return str(cand)
    sdp_home = os.environ.get("SNAPDRAGON_PROFILER_HOME")
    if sdp_home:
        for name in _SDPTRACE_BIN_NAMES:
            cand = Path(sdp_home) / name
            if cand.exists():
                return str(cand)
            cand = Path(sdp_home) / "Tools" / name
            if cand.exists():
                return str(cand)
    return None


def sdptrace_available() -> bool:
    """True iff sdptrace is callable on this host."""
    return find_sdptrace() is not None


def install_hint() -> str:
    """One-line guidance to surface when sdptrace isn't installed."""
    return (
        f"Snapdragon Profiler not detected. Download from "
        f"{SDPTRACE_DOWNLOAD_URL} (free; Qualcomm developer login). "
        f"After install, add Tools/ to PATH or set SNAPDRAGON_PROFILER_HOME, "
        f"then re-run `quad doctor --real-mode`."
    )


# ── Capture ────────────────────────────────────────────────────────────────


async def capture_trace(
    duration_s: float,
    *,
    output_path: Optional[str] = None,
    targets: Optional[list[str]] = None,
) -> TraceCapture:
    """Capture an ``sdptrace`` window of length ``duration_s``.

    Args:
        duration_s:  Capture window in seconds.
        output_path: Optional explicit chrometrace JSON output path. If
                     None, uses a temp file (caller manages cleanup).
        targets:     Optional list of capture targets passed to
                     ``--targets`` (e.g. ``cpu,gpu,npu``).

    Returns:
        TraceCapture with the path to the chrometrace JSON. Always
        non-raising; ``success=False`` indicates the tool isn't
        available or the capture failed.
    """
    tool = find_sdptrace()
    if tool is None:
        return TraceCapture(reason="sdptrace:not_available")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="quad_sdptrace_", suffix=".json")
        os.close(fd)
    out = Path(output_path)

    cmd = [
        tool, "capture",
        "--duration", str(int(duration_s * 1000)),
        "--format", "chrometrace",
        "--output", str(out),
    ]
    if targets:
        cmd += ["--targets", ",".join(targets)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=duration_s + 60.0)
        except asyncio.TimeoutError:
            proc.kill()
            return TraceCapture(reason="sdptrace:timeout")
        if proc.returncode != 0:
            return TraceCapture(reason="sdptrace:exit_nonzero")
    except (FileNotFoundError, OSError):
        return TraceCapture(reason="sdptrace:exec_failed")

    if not out.exists() or out.stat().st_size == 0:
        return TraceCapture(reason="sdptrace:no_output")

    return TraceCapture(
        trace_path=out,
        duration_s=duration_s,
        success=True,
    )


@asynccontextmanager
async def capture_trace_around(
    duration_s: float,
    *,
    output_path: Optional[str] = None,
    targets: Optional[list[str]] = None,
) -> AsyncIterator[TraceCapture]:
    """Async context manager that captures concurrently with the body.

    Usage::

        async with capture_trace_around(5.0) as trace:
            await run_inference(...)
        # `trace.trace_path` is the chrometrace JSON.

    If sdptrace isn't installed the context manager yields a
    ``TraceCapture`` with ``success=False`` and the body still runs —
    callers must check ``trace.available``.
    """
    tool = find_sdptrace()
    if tool is None:
        yield TraceCapture(reason="sdptrace:not_available")
        return

    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="quad_sdptrace_", suffix=".json")
        os.close(fd)
    out = Path(output_path)

    cmd = [
        tool, "capture",
        "--duration", str(int(duration_s * 1000)),
        "--format", "chrometrace",
        "--output", str(out),
    ]
    if targets:
        cmd += ["--targets", ",".join(targets)]

    capture = TraceCapture(trace_path=out, duration_s=duration_s)
    proc: asyncio.subprocess.Process | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, OSError):
        capture.reason = "sdptrace:exec_failed"
        yield capture
        return

    try:
        yield capture
    finally:
        if proc is not None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=duration_s + 60.0)
            except asyncio.TimeoutError:
                proc.kill()
                capture.reason = "sdptrace:timeout"
            else:
                if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
                    capture.success = True
                else:
                    capture.reason = "sdptrace:exit_nonzero"


__all__ = [
    "SDPTRACE_DOWNLOAD_URL",
    "TraceCapture",
    "capture_trace",
    "capture_trace_around",
    "find_sdptrace",
    "install_hint",
    "sdptrace_available",
]
