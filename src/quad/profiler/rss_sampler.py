"""Per-process RSS / working-set sampler.

Wraps a subprocess call so that while the child runs we periodically poll
``psutil.Process.memory_info()`` and record peak + mean RSS. The result
plus the original CompletedProcess is returned to the caller.

Used by the QAIRT adapter to populate ``ProfilingReport.memory_peak_mb``
and ``memory_avg_mb`` honestly, replacing the previous
``not_measured:requires_per_proc_rss_capture`` placeholder.
"""
from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover - declared in [real] extras
    psutil = None


@dataclasses.dataclass
class RSSReport:
    """Aggregate RSS stats from one subprocess run."""
    samples: int = 0
    peak_mb: float = 0.0
    mean_mb: float = 0.0
    available: bool = True
    reason: str = ""


async def run_with_rss_sampling(
    cmd: list[str],
    *,
    timeout: float = 300.0,
    sample_interval_s: float = 0.1,
) -> tuple[subprocess.CompletedProcess, RSSReport]:
    """Run ``cmd`` and sample its RSS at ``sample_interval_s`` intervals.

    Returns a ``CompletedProcess`` (compatible with the existing
    ``_run_command`` callers) plus an ``RSSReport``. If psutil isn't
    importable or the process exits before the first sample lands,
    ``RSSReport.available`` is False and ``reason`` explains why.
    """
    if psutil is None:
        # Run without sampling so the caller's flow still works.
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        completed = subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )
        return completed, RSSReport(available=False, reason="psutil not installed")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    samples: list[float] = []
    sampler_done = asyncio.Event()

    async def _sampler() -> None:
        try:
            ps = psutil.Process(proc.pid)
        except (psutil.NoSuchProcess, ProcessLookupError):
            return
        while not sampler_done.is_set():
            try:
                # ``memory_info().rss`` is the resident set on POSIX, the
                # working set on Windows — both are the right number for
                # "how much physical memory does this process hold."
                rss = ps.memory_info().rss / (1024 * 1024)
                samples.append(rss)
            except (psutil.NoSuchProcess, ProcessLookupError):
                break
            try:
                await asyncio.wait_for(sampler_done.wait(), timeout=sample_interval_s)
            except asyncio.TimeoutError:
                continue

    sampler_task = asyncio.create_task(_sampler())
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        sampler_done.set()
        await sampler_task
        raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    finally:
        sampler_done.set()
        # Give the sampler one tick to drain.
        try:
            await asyncio.wait_for(sampler_task, timeout=1.0)
        except asyncio.TimeoutError:
            sampler_task.cancel()

    completed = subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )

    if samples:
        peak = max(samples)
        mean = sum(samples) / len(samples)
        report = RSSReport(samples=len(samples), peak_mb=peak, mean_mb=mean, available=True)
    else:
        report = RSSReport(available=False, reason="process exited before first sample")
    return completed, report


__all__ = ["RSSReport", "run_with_rss_sampling"]
