"""snpe-diagview wrapper — parse and convert SNPE profiling log files.

snpe-diagview reads SNPEDiag_*.bin files produced by snpe-net-run and outputs:
  - Human-readable profiling text (all levels)
  - Chrometrace JSON files (--chrometrace, for Linting level only)

Usage::
    from quad.profiler.diagview import run_diagview, run_diagview_chrometrace

    # Parse a .bin log to text
    text = run_diagview("linting_output/SNPEDiag_0.bin")

    # Export a Linting chrometrace
    run_diagview_chrometrace("SNPEDiag_0.bin", output_prefix="trace")
    # → trace_subnet0.json, trace_subnet1.json, ...
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


DIAGVIEW_TOOL = "snpe-diagview"


def find_diagview() -> Optional[str]:
    """Return the path to snpe-diagview, or None if not found."""
    return shutil.which(DIAGVIEW_TOOL)


def run_diagview(
    diaglog_path: str,
    *,
    timeout: float = 60.0,
) -> str:
    """Run snpe-diagview on a .bin log file and return the text output.

    Args:
        diaglog_path: Path to SNPEDiag_*.bin file from snpe-net-run output
        timeout: Maximum seconds to wait for the command

    Returns:
        Full stdout text from snpe-diagview

    Raises:
        FileNotFoundError: If snpe-diagview is not in PATH
        RuntimeError: If snpe-diagview returns non-zero exit code
        TimeoutError: If the command exceeds timeout
    """
    tool = find_diagview()
    if tool is None:
        raise FileNotFoundError(
            f"'{DIAGVIEW_TOOL}' not found in PATH. "
            "Ensure QAIRT SDK is installed and sourced: source activate_qairt.sh"
        )

    cmd = [tool, "--input_log", diaglog_path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"snpe-diagview timed out after {timeout}s on {diaglog_path}"
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"snpe-diagview failed (exit {result.returncode}):\n{result.stderr[:500]}"
        )

    return result.stdout


def run_diagview_chrometrace(
    diaglog_path: str,
    output_prefix: str = "linting_trace",
    *,
    timeout: float = 60.0,
) -> list[str]:
    """Run snpe-diagview --chrometrace to export per-HTP-subnet JSON files.

    For a network with N HTP subnets, snpe-diagview produces N separate
    chrometrace JSON files. Non-HTP subnets are not exported.

    Args:
        diaglog_path: Path to SNPEDiag_*.bin file
        output_prefix: Filename prefix for output JSON files
        timeout: Maximum seconds to wait

    Returns:
        List of paths to generated chrometrace JSON files

    Raises:
        FileNotFoundError: If snpe-diagview not found
        RuntimeError: If command fails
    """
    tool = find_diagview()
    if tool is None:
        raise FileNotFoundError(
            f"'{DIAGVIEW_TOOL}' not found in PATH."
        )

    cmd = [
        tool,
        "--input_log", diaglog_path,
        "--chrometrace",
        "--output", output_prefix,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"snpe-diagview chrometrace timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(
            f"snpe-diagview --chrometrace failed (exit {result.returncode}):\n"
            f"{result.stderr[:500]}"
        )

    # Discover generated files (snpe-diagview appends subnet index to prefix)
    parent = Path(output_prefix).parent
    stem = Path(output_prefix).name
    generated = sorted(parent.glob(f"{stem}*.json"))
    return [str(p) for p in generated]


def parse_diaglog_as_linting(diaglog_path: str, **kwargs) -> "LintingProfile":  # type: ignore[name-defined]
    """Run snpe-diagview on a .bin and parse the output as a LintingProfile.

    Convenience wrapper combining run_diagview() and parse_linting_output().

    Args:
        diaglog_path: Path to SNPEDiag_*.bin
        **kwargs: Forwarded to run_diagview() (e.g. timeout=)

    Returns:
        Parsed LintingProfile
    """
    from quad.profiler.linting import parse_linting_output

    text = run_diagview(diaglog_path, **kwargs)
    return parse_linting_output(text)
