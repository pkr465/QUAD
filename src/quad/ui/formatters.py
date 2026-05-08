"""Rich markdown formatters for QUAD MCP tool outputs.

Used by:
  * MCP tool responses (returned alongside the data dict)
  * The ``quad`` CLI commands (rendered to terminal)
  * Claude Code skill files in ``.claude/skills/``

Every formatter returns a single markdown string. They use Unicode
glyphs sparingly — primarily for status indicators (✓ / ✗ / ⚠) — so
that the output renders cleanly in both terminals (with UTF-8) and
chat-style markdown viewers.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


# ─── Primitives ────────────────────────────────────────────────────────────


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    align: Sequence[str] | None = None,
) -> str:
    """Render a markdown table.

    Args:
        headers: column headers
        rows: list of rows; each row is a list of cells
        align: per-column alignment ('l' / 'r' / 'c'), default left
    """
    if not headers:
        return ""
    align = align or ["l"] * len(headers)
    align_map = {"l": ":---", "r": "---:", "c": ":---:"}
    sep_row = [align_map.get(a, ":---") for a in align]

    def _fmt_cell(c: Any) -> str:
        if isinstance(c, float):
            return f"{c:.2f}"
        return str(c)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep_row) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(c) for c in row) + " |")
    return "\n".join(lines)


def format_utilization_bar(value: float, *, width: int = 20, label: str = "") -> str:
    """Render a unicode horizontal bar for a 0-100% value.

    Examples:
        >>> format_utilization_bar(75.0, label="NPU")
        '█████████████████░░░  75% NPU'
    """
    pct = max(0.0, min(100.0, value))
    filled = int(round(pct / 100.0 * width))
    bar = "█" * filled + "░" * (width - filled)
    suffix = f"  {pct:.0f}%"
    if label:
        suffix += f" {label}"
    return f"`{bar}` {suffix}"


def _icon(status: str) -> str:
    return {"pass": "✓", "warn": "⚠", "fail": "✗"}.get(status, "·")


# ─── Hardware / Device ─────────────────────────────────────────────────────


def format_device(device: dict[str, Any]) -> str:
    """Format a DeviceProfile dict (from ``hardware_detect``) as markdown.

    Accepts the dict shape that the MCP tool returns:
      chipset, platform, cpu_cores, cpu_arch, cpu_freq_ghz,
      gpu_model, gpu_tflops, npu_model, npu_tops, ram_gb,
      sdk_path, sdk_version, available_runtimes
    """
    chipset = device.get("chipset", "unknown")
    platform = device.get("platform", "unknown")

    rows = [
        ["**CPU**", f"{device.get('cpu_cores', 0)} × {device.get('cpu_arch', '?')} @ {device.get('cpu_freq_ghz', 0)} GHz"],
        ["**GPU**", f"{device.get('gpu_model', '?')} ({device.get('gpu_tflops', 0)} TFLOPS)"],
        ["**NPU**", f"{device.get('npu_model', '?')} ({device.get('npu_tops', 0)} TOPS)"],
        ["**RAM**", f"{device.get('ram_gb', 0)} GB"],
        ["**Runtimes**", ", ".join(device.get("available_runtimes", []) or ["—"])],
        ["**SDK**", f"{device.get('sdk_version', '?')} @ {device.get('sdk_path', '?') or '(not configured)'}"],
    ]
    return (
        f"### Hardware: {chipset}\n"
        f"_{platform.title()} platform_\n\n"
        + format_table(["", "Detail"], rows)
    )


# ─── Profiling report ──────────────────────────────────────────────────────


def format_profile(profile: dict[str, Any]) -> str:
    """Format a ProfilingReport dict as markdown."""
    lat = profile.get("latency", {})
    util = profile.get("utilization", {})
    runtime = profile.get("runtime_used", "?")
    level = profile.get("profiling_level", "detailed")

    parts: list[str] = []
    parts.append(f"### Profile — runtime: **{runtime.upper()}** · level: `{level}`\n")

    # Latency table
    parts.append("**Latency**\n")
    parts.append(format_table(
        ["Statistic", "Value (ms)"],
        [
            ["Mean", lat.get("mean_ms", 0)],
            ["Median (p50)", lat.get("p50_ms", 0)],
            ["p95", lat.get("p95_ms", 0)],
            ["p99", lat.get("p99_ms", 0)],
            ["Min", lat.get("min_ms", 0)],
            ["Max", lat.get("max_ms", 0)],
        ],
        align=["l", "r"],
    ))
    parts.append("")

    # Throughput / Power / Memory one-liner
    parts.append(
        f"**Throughput:** {profile.get('throughput_fps', 0):.0f} FPS  ·  "
        f"**Power:** {profile.get('power_mw', 0):.0f} mW  ·  "
        f"**Memory:** peak {profile.get('memory_peak_mb', 0):.0f} MB / "
        f"avg {profile.get('memory_avg_mb', 0):.0f} MB"
    )
    parts.append("")

    # Utilisation bars
    parts.append("**Compute utilisation**")
    for key in ("npu", "gpu", "cpu"):
        pct = float(util.get(key, 0) or 0)
        parts.append(format_utilization_bar(pct, label=key.upper()))
    parts.append("")

    # Linting bottlenecks if present
    linting_layers = profile.get("linting_layers")
    if linting_layers:
        bottlenecks = [layer for layer in linting_layers if layer.get("is_bottleneck")]
        if bottlenecks:
            parts.append(f"⚠ **{len(bottlenecks)} bottleneck op(s) detected**")
            top = sorted(bottlenecks, key=lambda x: x.get("total_cycles", 0), reverse=True)[:3]
            parts.append(format_table(
                ["#", "Op", "Cycles", "Overlap", "Hint"],
                [
                    [
                        op.get("index", "?"),
                        op.get("name", "?"),
                        f"{op.get('total_cycles', 0):,}",
                        f"{int(op.get('overlap_ratio', 0) * 100)}%",
                        (op.get("optimization_hint") or "")[:60],
                    ]
                    for op in top
                ],
                align=["r", "l", "r", "r", "l"],
            ))

    return "\n".join(parts)


# ─── Conversion result ─────────────────────────────────────────────────────


def format_conversion(result: dict[str, Any]) -> str:
    """Format a ConversionResult dict as markdown."""
    parts: list[str] = []
    parts.append(
        f"### Model converted: `{result.get('output_path', '?')}`\n"
    )

    rows = [
        ["Original size", f"{result.get('original_size_mb', 0):.1f} MB"],
        ["Output size", f"{result.get('model_size_mb', 0):.1f} MB"],
        ["Compression", f"{result.get('compression_ratio', 1):.2f}× smaller"],
        ["Quantization", result.get("quantization_applied", "fp32")],
        ["Supported ops", f"{result.get('supported_ops_pct', 100):.0f}%"],
        ["Conversion time", f"{result.get('conversion_time_s', 0):.2f} s"],
        ["Target SDK", result.get("target_sdk", "?")],
    ]
    parts.append(format_table(["", ""], rows))

    fallback_ops = result.get("unsupported_ops", []) or []
    if fallback_ops:
        parts.append("")
        parts.append(
            f"⚠ **{len(fallback_ops)} op(s) will fall back to CPU**: "
            + ", ".join(f"`{op}`" for op in fallback_ops[:6])
            + ("…" if len(fallback_ops) > 6 else "")
        )

    notes = result.get("conversion_notes", []) or []
    if notes:
        parts.append("")
        parts.append("**Conversion notes:**")
        for note in notes[:4]:
            parts.append(f"- {note}")

    image_notes = result.get("image_format_notes", []) or []
    if image_notes:
        parts.append("")
        parts.append("**Image format guidance:**")
        for note in image_notes[:3]:
            parts.append(f"- {note}")

    warnings = result.get("warnings", []) or []
    if warnings:
        parts.append("")
        parts.append(f"**Warnings ({len(warnings)}):**")
        for w in warnings[:3]:
            parts.append(f"- {w}")

    return "\n".join(parts)


# ─── Allocation map ────────────────────────────────────────────────────────


def format_allocation(alloc: dict[str, Any]) -> str:
    """Format an AllocationMap dict (from ``orchestrate_workload``)."""
    parts: list[str] = []
    parts.append(f"### Allocation — `{alloc.get('power_mode', '?')}` power mode\n")

    rows = [
        ["Projected latency", f"{alloc.get('projected_latency_ms', 0):.2f} ms"],
        ["Projected power", f"{alloc.get('projected_power_mw', 0):.0f} mW"],
        ["Projected memory", f"{alloc.get('projected_memory_mb', 0):.0f} MB"],
    ]
    parts.append(format_table(["", ""], rows))
    parts.append("")

    parts.append("**Compute distribution**")
    for key in ("npu", "gpu", "cpu"):
        pct = float(alloc.get(f"{key}_utilization_pct", 0) or 0)
        parts.append(format_utilization_bar(pct, label=key.upper()))
    parts.append("")

    fallback = alloc.get("fallback_layers", []) or []
    if fallback:
        parts.append(
            f"⚠ **{len(fallback)} layer(s) fall back to CPU** "
            f"(unsupported on NPU): "
            + ", ".join(f"`{l}`" for l in fallback[:5])
            + ("…" if len(fallback) > 5 else "")
        )

    return "\n".join(parts)


# ─── Doctor report ─────────────────────────────────────────────────────────


def format_doctor(checks: list[dict[str, str]] | list[Any]) -> str:
    """Format a list of CheckResult records as markdown.

    Accepts either dicts with name/status/message keys OR objects with
    those attributes.
    """
    parts: list[str] = ["### Doctor report\n"]

    rows = []
    pass_count = warn_count = fail_count = 0
    for check in checks:
        if isinstance(check, dict):
            name = check.get("name", "?")
            status = check.get("status", "?")
            message = check.get("message", "")
        else:
            name = getattr(check, "name", "?")
            status = getattr(check, "status", "?")
            message = getattr(check, "message", "")
        if status == "pass":
            pass_count += 1
        elif status == "warn":
            warn_count += 1
        elif status == "fail":
            fail_count += 1
        rows.append([_icon(status), name, message[:100]])

    parts.append(format_table(
        ["", "Check", "Detail"],
        rows,
        align=["c", "l", "l"],
    ))
    parts.append("")
    parts.append(
        f"**Summary:** {pass_count} passed · {warn_count} warnings · {fail_count} errors"
    )
    return "\n".join(parts)


# ─── Coverage report ───────────────────────────────────────────────────────


def format_coverage(coverage: dict[str, Any]) -> str:
    """Format a CoverageReport (or dict-of-reports keyed by target)."""
    if "target" in coverage and "coverage_pct" in coverage:
        # Single-target report
        return _format_single_coverage(coverage)
    # Multi-target dict
    parts = ["### Op-coverage by target\n"]
    rows = []
    for target, report in coverage.items():
        if not isinstance(report, dict):
            continue
        rows.append([
            target,
            report.get("total_ops", 0),
            report.get("supported_ops", 0),
            f"{report.get('coverage_pct', 0):.1f}%",
            "✓" if report.get("is_fully_covered") else "⚠",
        ])
    parts.append(format_table(
        ["Target", "Total ops", "Supported", "Coverage", ""],
        rows,
        align=["l", "r", "r", "r", "c"],
    ))
    return "\n".join(parts)


def _format_single_coverage(report: dict[str, Any]) -> str:
    target = report.get("target", "?")
    pct = report.get("coverage_pct", 0)
    icon = "✓" if report.get("is_fully_covered") else "⚠"
    parts = [
        f"### Op coverage on **{target}** {icon}\n",
        f"`{format_utilization_bar(pct, label='supported').strip('`').strip()}`",
        "",
        f"**{report.get('supported_ops', 0)} of {report.get('total_ops', 0)} ops supported.**",
    ]
    unsupported = report.get("unsupported_ops", []) or []
    if unsupported:
        parts.append("")
        parts.append(f"**Unsupported ops ({len(unsupported)}):**")
        for u in unsupported[:8]:
            parts.append(f"- `{u.get('op_type', '?')}` ({u.get('name', '?')})")
        if len(unsupported) > 8:
            parts.append(f"- _… {len(unsupported) - 8} more_")
    rec = report.get("fallback_recommendation", "")
    if rec:
        parts.append("")
        parts.append(f"_{rec}_")
    return "\n".join(parts)


# ─── SDK status (sdk_manager output) ───────────────────────────────────────


def format_sdk_status(info: dict[str, Any] | None) -> str:
    """Format the SDKInfo dict from ``sdk_manager.resolve_sdk_root()``."""
    if info is None:
        return (
            "### SDK status\n\n"
            "**No QAIRT/SNPE SDK detected.** Running in mock mode.\n\n"
            "To enable real-hardware mode:\n"
            "1. Download QAIRT from <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>\n"
            "2. `quad sdk install ~/Downloads/qairt-X.Y.Z.zip`\n"
            "3. `quad mode` (should report READY)"
        )

    rows = [
        ["Flavor", info.get("flavor", "?")],
        ["Version", info.get("version", "?")],
        ["Root", f"`{info.get('root', '?')}`"],
        ["Bin", f"`{info.get('bin_dir', '(none)')}`"],
        ["Source", info.get("source", "?")],
        [
            "Tools",
            f"qairt-converter: {'✓' if info.get('has_qairt_converter') else '✗'}, "
            f"snpe-net-run: {'✓' if info.get('has_snpe_net_run') else '✗'}",
        ],
    ]
    return "### Active SDK\n\n" + format_table(["", ""], rows)
