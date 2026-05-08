"""QUAD Suggestions engine — actionable recommendations.

Given a model + target + profile, surfaces concrete next-step
recommendations. Used by the MCP tool wrappers and the Claude Code
skills to make QUAD feel like an experienced ML-systems engineer
rather than a passive tool.

Categories:
  * **Quantization** — pick INT8 / INT4 / FP16 based on size + accuracy
  * **Runtime** — pick NPU / GPU / CPU based on op coverage + latency
  * **Power mode** — performance / balanced / efficiency for the use case
  * **Optimisation** — surface bottleneck-fix opportunities from linting
  * **Architecture** — model-family-specific tips (e.g. "MobileNetV2 on
    HTP works best with INT8; INT4 trips on the SE module")

Each suggestion has:
  * A short ``title`` (under 80 chars; renders as the heading)
  * A longer ``rationale`` (1-2 sentences, why this matters)
  * Optional ``command`` to run next
  * ``severity`` — info / recommend / warning / critical
  * ``confidence`` — high / medium / low
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal


Severity = Literal["info", "recommend", "warning", "critical"]
Confidence = Literal["high", "medium", "low"]


@dataclass
class Suggestion:
    title: str
    rationale: str
    severity: Severity = "recommend"
    confidence: Confidence = "high"
    command: str | None = None
    category: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        icon = {
            "info": "ℹ️",
            "recommend": "💡",
            "warning": "⚠️",
            "critical": "🛑",
        }[self.severity]
        parts = [f"{icon} **{self.title}**", f"  {self.rationale}"]
        if self.command:
            parts.append(f"  ```\n  {self.command}\n  ```")
        return "\n".join(parts)


# ─── Suggestion generators ───────────────────────────────────────────────────


def suggest_quantization(
    *,
    model_size_mb: float,
    quantization: str = "fp32",
    target_memory_budget_mb: float | None = None,
) -> list[Suggestion]:
    """Suggest a quantization level for the given model + target."""
    out: list[Suggestion] = []

    if quantization == "fp32":
        if model_size_mb > 50:
            out.append(Suggestion(
                title="Quantize to INT8 to shrink ≥4× and unlock NPU",
                rationale=(
                    f"Model is {model_size_mb:.0f} MB at FP32. INT8 typically "
                    "achieves 4× compression with <1% accuracy drop on classification "
                    "and detection models. Required for HTP NPU execution."
                ),
                command='quantization="int8"',
                category="quantization",
            ))
        else:
            out.append(Suggestion(
                title="Consider INT8 quantization for NPU deployment",
                rationale=(
                    f"Model is {model_size_mb:.0f} MB. INT8 enables HTP NPU "
                    "execution (typically 4-10× faster than CPU at lower power)."
                ),
                command='quantization="int8"',
                severity="info",
                category="quantization",
            ))

    if target_memory_budget_mb and model_size_mb > target_memory_budget_mb:
        if quantization in ("fp32", "int8"):
            out.append(Suggestion(
                title="INT4 quantization may be required for tight memory budget",
                rationale=(
                    f"Model is {model_size_mb:.0f} MB but target budget is "
                    f"{target_memory_budget_mb:.0f} MB. INT4 (via AIMET) gives "
                    "8× compression at the cost of 1-3% accuracy. Provide real "
                    "calibration data for best results."
                ),
                command='quantization="int4", use_aimet=True, calibration_data=...',
                severity="warning",
                category="quantization",
            ))

    return out


def suggest_runtime(
    *,
    coverage_pct: float,
    npu_compatible_ops: int,
    total_ops: int,
    has_npu: bool,
    has_gpu: bool,
) -> list[Suggestion]:
    """Pick the right runtime for a model based on op coverage."""
    out: list[Suggestion] = []

    if has_npu and coverage_pct >= 95:
        out.append(Suggestion(
            title=f"Run on NPU — {coverage_pct:.0f}% op coverage, ideal for this model",
            rationale=(
                f"{npu_compatible_ops} of {total_ops} ops are NPU-compatible. "
                "Expect 4-10× throughput vs CPU at <50% the power."
            ),
            command='runtime="npu"',
            category="runtime",
        ))
    elif has_npu and 80 <= coverage_pct < 95:
        out.append(Suggestion(
            title=f"NPU + small CPU fallback — {coverage_pct:.0f}% op coverage",
            rationale=(
                f"{total_ops - npu_compatible_ops} op(s) will fall back to CPU. "
                "Still typically faster than full-CPU inference; check the "
                "fallback list for hot ops that warrant a custom UDO."
            ),
            command='runtime="auto"',
            severity="recommend",
            category="runtime",
        ))
    elif has_npu and coverage_pct < 80:
        out.append(Suggestion(
            title="GPU may be a better default than NPU for this model",
            rationale=(
                f"Only {coverage_pct:.0f}% of ops are NPU-compatible, which means "
                "many CPU/NPU transfers per inference. GPU has broader op coverage "
                "and avoids the transfer cost."
            ),
            command='runtime="gpu"' if has_gpu else 'runtime="cpu"',
            severity="warning",
            category="runtime",
        ))
    elif not has_npu:
        out.append(Suggestion(
            title="No NPU detected — use GPU or CPU",
            rationale=(
                "QUAD will route through Adreno GPU or Oryon CPU. NPU "
                "performance benefits aren't available on this machine."
            ),
            command='runtime="gpu"' if has_gpu else 'runtime="cpu"',
            severity="info",
            category="runtime",
        ))

    return out


def suggest_power_mode(
    *,
    use_case: str = "interactive",
    on_battery: bool = False,
    target_fps: float | None = None,
) -> list[Suggestion]:
    """Pick a power mode based on use case."""
    out: list[Suggestion] = []

    if use_case in ("realtime", "camera", "video"):
        out.append(Suggestion(
            title="Use `performance` power mode for real-time workloads",
            rationale=(
                "Camera/video pipelines need consistent sub-frame latency. "
                "Performance mode pins the NPU at max clock and disables "
                "thermal throttling. Higher power draw is acceptable here."
            ),
            command='power_mode="performance"',
            category="power",
        ))
    elif use_case in ("batch", "offline", "training-data"):
        out.append(Suggestion(
            title="Use `efficiency` power mode for batch workloads",
            rationale=(
                "Batch inference doesn't need millisecond latency. "
                "Efficiency mode trades 2-3× higher latency for 50% lower "
                "power, which matters for long-running jobs."
            ),
            command='power_mode="efficiency"',
            category="power",
        ))
    else:  # interactive / default
        out.append(Suggestion(
            title="Use `balanced` power mode for interactive workloads",
            rationale=(
                "Most chat/UI/single-image-classification flows benefit "
                "from balanced mode — good latency without thermal stress."
            ),
            command='power_mode="balanced"',
            severity="info",
            category="power",
        ))

    if on_battery and use_case in ("realtime", "camera", "video"):
        out.append(Suggestion(
            title="On battery: consider efficiency mode for sustained sessions",
            rationale=(
                "Performance mode at full NPU clock can drain a battery in "
                "30-60 minutes. If the user keeps the app open, balanced or "
                "efficiency mode is much friendlier."
            ),
            severity="warning",
            category="power",
        ))

    return out


def suggest_optimisations(
    *,
    bottlenecks: list[dict[str, Any]],
    profiling_level: str = "detailed",
) -> list[Suggestion]:
    """Surface optimisation ideas from a linting profile.

    Each bottleneck dict is expected to have ``name``, ``op_type``,
    ``overlap_ratio``, and optionally ``optimization_hint``.
    """
    out: list[Suggestion] = []
    for b in bottlenecks[:5]:
        op = b.get("op_type") or "op"
        name = b.get("name") or "?"
        overlap = b.get("overlap_ratio", 0)
        hint = b.get("optimization_hint")
        if hint:
            out.append(Suggestion(
                title=f"Bottleneck `{name}`: {hint[:60]}",
                rationale=(
                    f"Op overlap is {overlap * 100:.0f}% — the HTP cores are "
                    f"stalling on memory or scalar dependencies for most of "
                    f"this op's runtime. {hint}"
                ),
                severity="warning",
                category="optimisation",
            ))
        else:
            out.append(Suggestion(
                title=f"Investigate `{name}` ({op}) — only {overlap * 100:.0f}% overlap",
                rationale=(
                    "Low resource overlap suggests the op is scalar-bound or "
                    "memory-bound. Consider replacing with an HTP-friendly "
                    "equivalent — Sub→Conv, Div→Mul, PReLU→ReLU+FP16 are common "
                    "wins."
                ),
                severity="info",
                category="optimisation",
            ))

    if profiling_level not in ("linting", "qhas") and not bottlenecks:
        out.append(Suggestion(
            title="Run linting profile to find HTP bottlenecks",
            rationale=(
                "Detailed profiling shows ms timings; linting profiling "
                "shows per-op cycle counts and resource overlap. Use it to "
                "find ops with low utilisation that bottleneck the pipeline."
            ),
            command='profile_workload(model, profiling_level="linting")',
            severity="info",
            category="optimisation",
        ))

    return out


def suggest_for_workflow(
    *,
    profile: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    conversion: dict[str, Any] | None = None,
    use_case: str = "interactive",
    on_battery: bool = False,
) -> list[Suggestion]:
    """One-stop helper: combine all suggestion sources for a workflow.

    Returns a flat ordered list — most-important first.
    """
    out: list[Suggestion] = []

    if conversion:
        out.extend(suggest_quantization(
            model_size_mb=conversion.get("original_size_mb", 0),
            quantization=conversion.get("quantization_applied", "fp32"),
        ))

    if coverage:
        # coverage may be a single report or a multi-target dict
        report = coverage if "coverage_pct" in coverage else next(
            (v for v in coverage.values() if isinstance(v, dict)), None
        )
        if report:
            out.extend(suggest_runtime(
                coverage_pct=report.get("coverage_pct", 100),
                npu_compatible_ops=report.get("supported_ops", 0),
                total_ops=report.get("total_ops", 1),
                has_npu=True,
                has_gpu=True,
            ))

    out.extend(suggest_power_mode(use_case=use_case, on_battery=on_battery))

    if profile:
        bottlenecks = []
        if profile.get("linting_layers"):
            bottlenecks = [
                layer for layer in profile["linting_layers"]
                if layer.get("is_bottleneck")
            ]
        out.extend(suggest_optimisations(
            bottlenecks=bottlenecks,
            profiling_level=profile.get("profiling_level", "detailed"),
        ))

    return out
