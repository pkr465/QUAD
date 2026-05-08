"""Tips system — contextual one-liners surfaced with MCP tool responses.

Where ``suggestions.py`` produces multi-sentence actionable
recommendations based on real data, ``tips.py`` is for *general*
guidance that's helpful regardless of input. Tips are short
(under ~120 chars), grouped by context, and tagged with severity.

Usage:

    from quad.tips import get_tips_for, format_tips_markdown

    tips = get_tips_for("convert_model", n=2)
    print(format_tips_markdown(tips))

The tip catalogue is module-level data — easy to extend without
touching the engine.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Literal

TipLevel = Literal["info", "tip", "warning"]


@dataclass(frozen=True)
class Tip:
    text: str
    context: str  # 'detect' / 'convert' / 'profile' / 'orchestrate' / 'codegen' / 'general'
    level: TipLevel = "tip"
    link: str | None = None

    def to_markdown(self) -> str:
        icon = {"info": "ℹ️", "tip": "💡", "warning": "⚠️"}[self.level]
        link_part = f" [{self.link}]({self.link})" if self.link else ""
        return f"{icon} {self.text}{link_part}"


# ─── Catalogue ──────────────────────────────────────────────────────────────


_CATALOGUE: tuple[Tip, ...] = (
    # --- General -----------------------------------------------------------
    Tip(
        text="Run `quad doctor --real-mode` to verify the SDK + tools are wired correctly.",
        context="general",
    ),
    Tip(
        text="Use `quad mode` to see whether real-hardware mode is READY on this machine.",
        context="general",
    ),
    Tip(
        text="Set `QUAD_STRICT_REAL=1` in CI to fail fast when the SDK is missing instead of silently mocking.",
        context="general",
    ),
    Tip(
        text="The full gap analysis lives at `docs/GAP_ANALYSIS.md` — useful when planning real-hardware work.",
        context="general",
        link="docs/GAP_ANALYSIS.md",
    ),

    # --- detect ------------------------------------------------------------
    Tip(
        text="`quad detect --refresh` re-probes local hardware (skips the discovery cache).",
        context="detect",
    ),
    Tip(
        text="On Snapdragon X-series Copilot+ PCs, the NPU appears as a `ComputeAccelerator` PnP device.",
        context="detect",
        level="info",
    ),
    Tip(
        text="If `quad detect` reports an x86_64 host on a known Snapdragon laptop, you may be running an emulated shell. Try cmd.exe or pwsh.",
        context="detect",
        level="warning",
    ),

    # --- convert -----------------------------------------------------------
    Tip(
        text="Pass real `calibration_data` to INT8/INT4 conversion. Random noise produces wrong quantization scales.",
        context="convert",
        level="warning",
    ),
    Tip(
        text="ONNX models with symbolic batch dimensions need `input_dimensions=...` to bind a concrete shape.",
        context="convert",
    ),
    Tip(
        text="PyTorch models: export to ONNX first via `torch.onnx.export`, then convert. QUAD doesn't read .pt directly.",
        context="convert",
    ),
    Tip(
        text="HTP NPU input layout is NHWC. PyTorch defaults are NCHW — use `input_layout='nchw'` so QUAD inserts the transpose.",
        context="convert",
        level="info",
    ),
    Tip(
        text="INT4 requires per-channel symmetric scheme via AIMET. Per-tensor INT4 is unusable for almost any model.",
        context="convert",
        level="warning",
    ),

    # --- profile -----------------------------------------------------------
    Tip(
        text="Profile in `linting` mode to find HTP bottlenecks (cycle-level resource overlap per op).",
        context="profile",
    ),
    Tip(
        text="QHAS profiling produces a chrometrace JSON viewable in chrome://tracing.",
        context="profile",
        level="info",
    ),
    Tip(
        text="High p99/mean ratio (>2x) usually means thermal throttling. Run a 30-second pre-warm before profiling.",
        context="profile",
        level="info",
    ),
    Tip(
        text="Profile with `--perf_profile burst` for camera-like spike loads, `high_performance` for sustained loads.",
        context="profile",
    ),

    # --- orchestrate -------------------------------------------------------
    Tip(
        text="Linting/QHAS profiles don't have ms-per-layer timings — orchestrate auto-reprofiles in `detailed` mode for you.",
        context="orchestrate",
        level="info",
    ),
    Tip(
        text="Performance vs balanced mode usually differ in DSP clock cap, not core assignment.",
        context="orchestrate",
        level="info",
    ),
    Tip(
        text="Custom UDOs (User-Defined Operations) let you implement an unsupported op on HTP — see `templates/snpe/udo/`.",
        context="orchestrate",
    ),

    # --- codegen -----------------------------------------------------------
    Tip(
        text="Set `QUAD_VALIDATE_CPP_SYNTAX=1` to invoke `gcc -fsyntax-only` on generated C++ output.",
        context="codegen",
    ),
    Tip(
        text="Set `QUAD_STRICT_TODOS=1` to reject any generated code containing TODO/FIXME markers.",
        context="codegen",
        level="warning",
    ),
    Tip(
        text="Generated C++ links against the QAIRT SDK headers — make sure CMake's `QAIRT_SDK_ROOT` is set.",
        context="codegen",
        level="info",
    ),

    # --- serve -------------------------------------------------------------
    Tip(
        text="`quad serve` requires the [real] extras: `pip install -e .[real]` for fastapi + uvicorn.",
        context="serve",
        level="info",
    ),
    Tip(
        text="The HTTP server's /infer endpoint expects base64-encoded ndarray bytes + shape + dtype.",
        context="serve",
    ),
    Tip(
        text="Use /metrics to track p99 latency under load. Sustained p99 > 2x mean is a good thermal-throttling alarm.",
        context="serve",
    ),
)


# ─── Public API ─────────────────────────────────────────────────────────────


def get_tips_for(
    context: str,
    n: int = 2,
    *,
    level: TipLevel | None = None,
    seed: int | None = None,
) -> list[Tip]:
    """Return up to ``n`` tips relevant to a context.

    Args:
        context: 'detect' | 'convert' | 'profile' | 'orchestrate' |
                 'codegen' | 'serve' | 'general'
        n: max number of tips to return (default 2 to avoid noise)
        level: optionally filter by severity
        seed: pin the RNG for reproducible output (useful in tests)

    Returns:
        List of tips, possibly empty. The list is shuffled so we don't
        always show the same tips first.
    """
    pool = [t for t in _CATALOGUE if t.context == context]
    if level:
        pool = [t for t in pool if t.level == level]

    if not pool:
        # Fall back to general tips
        pool = [t for t in _CATALOGUE if t.context == "general"]

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def get_general_tips(n: int = 1, *, seed: int | None = None) -> list[Tip]:
    return get_tips_for("general", n=n, seed=seed)


def format_tips_markdown(tips: Iterable[Tip]) -> str:
    """Render a list of tips as a markdown bulleted list."""
    items = [f"- {tip.to_markdown()}" for tip in tips]
    if not items:
        return ""
    return "**Tips:**\n" + "\n".join(items)


def all_contexts() -> list[str]:
    """Return all context strings present in the catalogue."""
    return sorted({t.context for t in _CATALOGUE})


def catalogue_size() -> int:
    return len(_CATALOGUE)
