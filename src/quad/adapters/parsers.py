"""Pure-function parsers for QAIRT/SNPE CLI tool stdout/stderr/CSV.

Kept deliberately free of side effects so they can be unit-tested with
hand-crafted fixtures without launching the SDK. The QAIRT adapter
imports these and converts the dicts into Pydantic models.

Parsers in this module:

  - parse_snpe_net_run_stdout(text) -> dict
        Total / Average inference time, runtime banner, error tags.

  - parse_snpe_diagview_csv(csv_text) -> dict
        Total Inference Time + Forward Propagate from snpe-diagview's
        CSV output. The CSV is the authoritative source of detailed
        metrics; snpe-net-run alone only emits the binary diaglog.

  - parse_snpe_diagview_layers(csv_text) -> list[dict]
        Per-layer rows from the "Model Layer Times" CSV section.

  - parse_qnn_platform_validator(stdout) -> dict
        Backend/runtime availability + chipset / NPU / GPU strings.

  - parse_qairt_converter_stdout(stdout) -> dict
        Conversion summary: status, output_path, supported_ops_pct,
        unsupported_ops, warnings.

All parsers return ``{"_parsed": False, "reason": "..."}`` rather than
raising on unrecognised input, so callers can degrade to mock outputs
without losing the diagnostic.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any


# ── snpe-net-run stdout ─────────────────────────────────────────────────────


# Multiple historical / per-runtime phrasings. Keep the most-specific
# patterns first; the first match wins.
_RX_NET_RUN_LATENCY = (
    re.compile(r"Total Inference Time(?:\s*\[NetRun\])?\s*[:=]\s*([\d.]+)\s*ms", re.I),
    re.compile(r"Total Inference Time\s*[:=]\s*([\d.]+)\s*us", re.I),       # microseconds
    re.compile(r"Average\s+Total\s+Inference\s+Time\s*[:=]\s*([\d.]+)", re.I),
    re.compile(r"Average\s+Inference\s+Time\s*[:=]\s*([\d.]+)\s*ms?", re.I),
    re.compile(r"\binference\s+time\s*[:=]\s*([\d.]+)\s*ms\b", re.I),
    re.compile(r"\bExecute\s+time\s*[:=]\s*([\d.]+)\s*ms", re.I),
)
_RX_NET_RUN_RUNTIME = re.compile(
    r"\bUsing\s+(?:the\s+)?(?P<rt>CPU|GPU|DSP|HTP|AIP)\s+runtime", re.I
)
_RX_NET_RUN_PERF_PROFILE = re.compile(
    r"\bperf[_\-]profile\s*[:=]\s*(?P<v>\w+)", re.I
)
_RX_NET_RUN_FORWARD = re.compile(
    r"Forward\s+Propa(?:gate|gat\s+\(?)\s*[:=]\s*([\d.]+)\s*(us|ms)", re.I
)
_RX_NET_RUN_OUTPUTS = re.compile(
    r"Saved\s+(\d+)\s+output", re.I
)
_RX_NET_RUN_ERROR = re.compile(
    r"\b(?:ERROR|FATAL|Status\s*[:=]\s*\w*FAIL\w*|transportStatus:\s*\d+)\b"
)


def parse_snpe_net_run_stdout(text: str) -> dict[str, Any]:
    """Parse snpe-net-run combined stdout+stderr.

    Returns a dict with these keys (any of which may be None / 0 / []
    when the corresponding line wasn't seen):

        latency_ms        : float    — total inference time in ms
        forward_ms        : float    — accelerator-only execute time
        runtime           : str|None — cpu / gpu / npu (mapped from DSP/HTP)
        perf_profile      : str|None
        n_outputs         : int      — count of saved output buffers
        errors            : list[str]— any FAIL/ERROR lines
        _parsed           : bool     — True iff at least latency or runtime was found
    """
    out: dict[str, Any] = {
        "latency_ms": 0.0,
        "forward_ms": 0.0,
        "runtime": None,
        "perf_profile": None,
        "n_outputs": 0,
        "errors": [],
        "_parsed": False,
    }
    if not text:
        return out

    for rx in _RX_NET_RUN_LATENCY:
        m = rx.search(text)
        if m:
            try:
                v = float(m.group(1))
            except ValueError:
                continue
            # Convert microseconds to ms when the regex came from a "us" pattern.
            if "us" in rx.pattern.lower() and "\\bms" not in rx.pattern.lower():
                v = v / 1000.0
            out["latency_ms"] = v
            out["_parsed"] = True
            break

    m = _RX_NET_RUN_FORWARD.search(text)
    if m:
        v = float(m.group(1))
        if m.group(2).lower() == "us":
            v = v / 1000.0
        out["forward_ms"] = v

    m = _RX_NET_RUN_RUNTIME.search(text)
    if m:
        rt = m.group("rt").lower()
        out["runtime"] = "npu" if rt in ("dsp", "htp", "aip") else rt
        out["_parsed"] = True

    m = _RX_NET_RUN_PERF_PROFILE.search(text)
    if m:
        out["perf_profile"] = m.group("v").lower()

    m = _RX_NET_RUN_OUTPUTS.search(text)
    if m:
        out["n_outputs"] = int(m.group(1))

    out["errors"] = [
        line.strip()
        for line in text.splitlines()
        if _RX_NET_RUN_ERROR.search(line)
    ][:5]
    return out


# ── snpe-diagview CSV (authoritative latency + per-layer source) ────────────


# Snippet of the CSV produced by snpe-diagview --input_log <file>:
#
#   Section Title
#   Init,Create Network(s),De-Init,RPC Init Time,...
#   12345,9876,234,5678,...
#
#   Section Title
#   Total Inference Time,Forward Propagate,Misc Accelerator,...
#   12345,9876,...
#
#   Section Title
#   Layer Names
#   conv1, conv2, ...
#
#   Section Title
#   Layer Times
#   conv1,123,234,345,DSP
#   conv2,...
#
# Sections are separated by blank lines or "section" comment rows.

_DIAGVIEW_LATENCY_KEYS = {
    "total inference time": "total_inference_us",
    "total inference time [netrun]": "total_inference_us",
    "forward propagate": "forward_propagate_us",
    "forward propogate": "forward_propagate_us",
    "rpc execute": "rpc_execute_us",
    "rpc init time": "rpc_init_us",
    "snpe accelerator": "snpe_accelerator_us",
    "accelerator": "accelerator_us",
    "misc accelerator": "misc_accelerator_us",
    "init": "init_us",
    "de-init": "deinit_us",
    "create network(s)": "create_network_us",
}


def parse_snpe_diagview_csv(csv_text: str) -> dict[str, Any]:
    """Parse the high-level latency block from snpe-diagview CSV output.

    Returns a dict with timing fields in microseconds plus derived ms
    fields. Values default to 0.0 when the row is absent.

    Also returns ``mean_latency_ms``: the most useful single number for
    the caller — equals total_inference_us / 1000 if present, else
    forward_propagate_us / 1000, else 0.
    """
    out: dict[str, Any] = {k: 0.0 for k in _DIAGVIEW_LATENCY_KEYS.values()}
    out["mean_latency_ms"] = 0.0
    out["_parsed"] = False
    if not csv_text or not csv_text.strip():
        return out

    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        return out

    # Build a header→row index. snpe-diagview emits header rows like
    # "Total Inference Time, Forward Propagate, ..." followed by a
    # numeric row. We pair adjacent rows where the first row contains
    # a known latency key (case-insensitive).
    for i in range(len(rows) - 1):
        header = [c.strip().lower() for c in rows[i] if c is not None]
        if not header or not any(c in _DIAGVIEW_LATENCY_KEYS for c in header):
            continue
        values_row = rows[i + 1]
        for col_idx, col in enumerate(header):
            field = _DIAGVIEW_LATENCY_KEYS.get(col)
            if not field:
                continue
            if col_idx >= len(values_row):
                continue
            raw = values_row[col_idx].strip()
            try:
                out[field] = float(raw)
                out["_parsed"] = True
            except (ValueError, TypeError):
                continue

    # Pick the single best mean-latency answer.
    if out["total_inference_us"] > 0:
        out["mean_latency_ms"] = out["total_inference_us"] / 1000.0
    elif out["forward_propagate_us"] > 0:
        out["mean_latency_ms"] = out["forward_propagate_us"] / 1000.0

    return out


def parse_snpe_diagview_layers(csv_text: str) -> list[dict[str, Any]]:
    """Parse the per-layer "Model Layer Times" section of a diagview CSV.

    Layer rows follow this shape (after a "Layer Times"-style header):

        layer_name, average_us, min_us, max_us, runtime

    Returns a list of dicts:

        {"name": str, "avg_us": float, "min_us": float,
         "max_us": float, "runtime": str}
    """
    out: list[dict[str, Any]] = []
    if not csv_text:
        return out
    rows = list(csv.reader(io.StringIO(csv_text)))
    in_section = False
    for row in rows:
        if not row:
            in_section = False
            continue
        first = (row[0] or "").strip().lower()
        # Header detection: a row starting with "Layer Times" (or a
        # row whose entries are exactly the layer-time column labels)
        # opens the section.
        if first.startswith("layer times") or "layer times" in first:
            in_section = True
            continue
        if first in ("layer", "name"):
            in_section = True
            continue
        if not in_section:
            continue
        # Section ends when we hit a non-numeric second column.
        if len(row) < 4:
            continue
        try:
            avg_us = float(row[1])
            min_us = float(row[2])
            max_us = float(row[3])
        except ValueError:
            in_section = False
            continue
        runtime = (row[4] if len(row) >= 5 else "").strip().lower() or "npu"
        if runtime in ("dsp", "htp", "aip"):
            runtime = "npu"
        out.append({
            "name": row[0].strip(),
            "avg_us": avg_us,
            "min_us": min_us,
            "max_us": max_us,
            "runtime": runtime,
        })
    return out


# ── qnn-platform-validator stdout ───────────────────────────────────────────


_RX_PV_BACKEND_BLOCK = re.compile(
    r"Backend\s*=\s*(?P<be>CPU|GPU|DSP|HTP|SAVER)\b", re.I
)
_RX_PV_HARDWARE = re.compile(
    r"Backend\s+Hardware\s*[:=]\s*(?P<v>Supported|Not\s*Supported|Not\s*Found)", re.I
)
# In-block "Library Version :" / "Core Version :" — these live inside
# the {...} body of the Results Summary and carry the canonical value.
# We deliberately do NOT match the verbose pre-summary lines here so
# per-backend slicing isn't polluted by adjacent backends' verbose log.
_RX_PV_LIB_VERSION_IN_BLOCK = re.compile(
    r"^\s*Library\s+Version\s*[:=]\s*(?P<v>[^\r\n}]+?)\s*$", re.I | re.M
)
_RX_PV_CORE_VERSION_IN_BLOCK = re.compile(
    r"^\s*Core\s+Version\s*[:=]\s*(?P<v>[^\r\n}]+?)\s*$", re.I | re.M
)
_RX_PV_HEXAGON = re.compile(r"\bHexagon\s+Architecture\s+(V?\d+)\b", re.I)
_RX_PV_ADRENO = re.compile(r"\bAdreno(?:\(TM\))?\s+([\w\-]+)", re.I)


def parse_qnn_platform_validator(stdout: str) -> dict[str, Any]:
    """Parse qnn-platform-validator stdout into a structured dict.

    Returns:
        runtimes:    list of "cpu" / "gpu" / "npu" available
        chipset:     str | None
        npu_arch:    e.g. "V73" if Hexagon Architecture V73 was reported
        gpu_model:   e.g. "X1-85"
        per_backend: { "DSP": {"hardware": "Supported", "lib_version": "...", "core_version": "..."}, ... }
        _parsed:     bool
    """
    result: dict[str, Any] = {
        "runtimes": [],
        "chipset": None,
        "npu_arch": None,
        "gpu_model": None,
        "per_backend": {},
        "_parsed": False,
    }
    if not stdout:
        return result

    # Walk per-backend blocks. The validator emits something like:
    #     Backend = DSP
    #     {
    #       Backend Hardware  : Supported
    #       Library Version   : Not Found
    #       Core Version      : Hexagon Architecture V73
    #     }
    # We match each Backend = X line, then capture the next ~12 lines.
    blocks = re.split(r"^\*+\s*Results Summary\s*\*+\s*$", stdout, flags=re.M)
    if len(blocks) > 1:
        # Walk only the "Results Summary" portion(s) for per-backend blocks.
        blocks = blocks[1:]
    else:
        blocks = [stdout]

    for block in blocks:
        for m_be in _RX_PV_BACKEND_BLOCK.finditer(block):
            be = m_be.group("be").upper()
            # Slice the block from this Backend = match to the next one.
            start = m_be.end()
            next_match = _RX_PV_BACKEND_BLOCK.search(block, start)
            sub = block[start:next_match.start()] if next_match else block[start:]

            # Restrict matching to the {...} body so adjacent backends'
            # verbose pre-summary lines (which live above the next
            # `Backend = ...`) don't bleed into this entry.
            brace_open = sub.find("{")
            brace_close = sub.find("}", brace_open + 1) if brace_open >= 0 else -1
            if 0 <= brace_open < brace_close:
                body = sub[brace_open + 1:brace_close]
            else:
                body = sub  # No braces — fall back to the whole slice.

            entry: dict[str, Any] = {}
            m = _RX_PV_HARDWARE.search(body)
            if m:
                entry["hardware"] = m.group("v").strip()
            m = _RX_PV_LIB_VERSION_IN_BLOCK.search(body)
            if m:
                entry["lib_version"] = m.group("v").strip()
            m = _RX_PV_CORE_VERSION_IN_BLOCK.search(body)
            if m:
                entry["core_version"] = m.group("v").strip()
            result["per_backend"][be] = entry
            result["_parsed"] = True

            if entry.get("hardware", "").lower().startswith("supported"):
                rt = "npu" if be in ("DSP", "HTP") else be.lower()
                if rt not in result["runtimes"]:
                    result["runtimes"].append(rt)

    m = _RX_PV_HEXAGON.search(stdout)
    if m:
        result["npu_arch"] = m.group(1).upper()
    m = _RX_PV_ADRENO.search(stdout)
    if m:
        result["gpu_model"] = m.group(1).strip()

    return result


# ── qairt-converter stdout ─────────────────────────────────────────────────


_RX_CONV_OK = re.compile(
    r"\b(Conversion\s+complete|Successfully\s+converted|Saved\s+to)\b", re.I
)
_RX_CONV_OUTPUT = re.compile(
    r"(?:Output|Saved\s+to)\s*[:=]\s*([^\r\n]+\.(?:bin|dlc))", re.I
)
_RX_CONV_SUPPORTED_PCT = re.compile(
    r"Supported\s+ops?\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.I
)
_RX_CONV_UNSUPPORTED_LIST = re.compile(
    r"Unsupported\s+ops?\s*[:=]\s*(?P<list>[^\r\n]+)", re.I
)
_RX_CONV_WARNING = re.compile(r"^\s*\[?WARNING\]?[:\s](.+)$", re.I | re.M)
_RX_CONV_ERROR = re.compile(r"^\s*\[?(?:ERROR|FATAL)\]?[:\s](.+)$", re.I | re.M)


def parse_qairt_converter_stdout(stdout: str) -> dict[str, Any]:
    """Parse qairt-converter stdout for conversion summary.

    Returns:
        success            : bool   — heuristic from "Conversion complete" or no ERROR
        output_path        : str|None
        supported_ops      : int    — count of ops the backend supports
        total_ops          : int    — total ops in the model
        supported_ops_pct  : float  — supported / total * 100, or 0
        unsupported_ops    : list[str]
        warnings           : list[str]
        errors             : list[str]
        _parsed            : bool
    """
    out: dict[str, Any] = {
        "success": False,
        "output_path": None,
        "supported_ops": 0,
        "total_ops": 0,
        "supported_ops_pct": 0.0,
        "unsupported_ops": [],
        "warnings": [],
        "errors": [],
        "_parsed": False,
    }
    if not stdout:
        return out

    if _RX_CONV_OK.search(stdout):
        out["success"] = True
        out["_parsed"] = True

    m = _RX_CONV_OUTPUT.search(stdout)
    if m:
        out["output_path"] = m.group(1).strip().strip('"')
        out["_parsed"] = True

    m = _RX_CONV_SUPPORTED_PCT.search(stdout)
    if m:
        sup, tot = int(m.group(1)), int(m.group(2))
        out["supported_ops"] = sup
        out["total_ops"] = tot
        if tot > 0:
            out["supported_ops_pct"] = round(sup / tot * 100.0, 1)
        out["_parsed"] = True

    m = _RX_CONV_UNSUPPORTED_LIST.search(stdout)
    if m:
        out["unsupported_ops"] = [
            op.strip() for op in m.group("list").split(",") if op.strip()
        ]
        out["_parsed"] = True

    out["warnings"] = [m.group(1).strip() for m in _RX_CONV_WARNING.finditer(stdout)][:10]
    errs = [m.group(1).strip() for m in _RX_CONV_ERROR.finditer(stdout)][:10]
    out["errors"] = errs
    if errs:
        out["success"] = False
    return out
