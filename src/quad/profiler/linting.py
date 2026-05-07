"""HTP Linting Profiler — per-op cycle analysis for HTP backend optimization.

Based on SNPE "Linting Profile" documentation (80-63442-10 Rev AH, Apr 13 2026).

Linting mode provides cycle-based per-op profiling for ops running on the HTP backend.
Unlike other profiling levels that report microseconds, Linting reports raw cycle counts
because there is no reliable cycle→microseconds conversion for parallelized HTP execution.

Activation:
  CLI:  snpe-net-run --profiling_level=linting ...
  API:  Snpe_SNPEBuilder_SetProfilingLevel(..., SNPE_PROFILING_LEVEL_LINTING)

Metrics per HTP op:
  Wait           — foreground cycles: main thread execution since previous op
  Overlap        — background cycles: parallel ops executing during this op
  Overlap(wait)  — background cycles: parallel ops executing during this op's Wait period
  Resources      — hardware units used: HVX, HMX, DMA (any combination)

Optimization strategy:
  Bottleneck indicator: high cycle count + low Overlap (little parallel activity)
  Well-optimized:       high Overlap relative to total cycles (good parallelism)

Caveats:
  - Only available for HTP subnets; non-HTP silently falls back to Detailed
  - snpe-diagview --chrometrace produces one JSON per HTP subnet
  - Multi-subnet (e.g. 3 HTP + 2 non-HTP): expect 3 chrometrace files

Tool: snpe-diagview (same as other profiling levels)
Chrometrace: --chrometrace flag exports JSON for chrome://tracing visualization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from quad.profiler.levels import ProfilingLevel


# ══════════════════════════════════════════════════════════════════════════════
# Constants & Thresholds
# ══════════════════════════════════════════════════════════════════════════════

LINTING_PROFILING_LEVEL = ProfilingLevel.LINTING.value

# Bottleneck detection thresholds (from documentation examples)
# An op is a bottleneck candidate when its overlap ratio is below this value
BOTTLENECK_OVERLAP_THRESHOLD = 0.25  # < 25% overlap = good optimization target

# An op is significant when it contributes more than this fraction of total cycles
SIGNIFICANT_OP_CYCLE_FRACTION = 0.05  # > 5% of total = noteworthy

# Op substitution recommendations from documentation examples
OP_SUBSTITUTIONS: dict[str, dict[str, Any]] = {
    "Sub": {
        "replacement": "Conv2D (with designed weights)",
        "rationale": (
            "Sub op in showcase model 1 consumed ~50% of total graph cycles with only "
            "~21.5% overlap. Replacing with a convolution performing equivalent arithmetic "
            "reduced total from 4,327,266 to 1,374,349 cycles (~68% improvement)."
        ),
        "htp_cost": "very_high",
    },
    "RealDiv": {
        "replacement": "Mul (multiply by reciprocal)",
        "rationale": (
            "Div op consumed ~68% of total cycles (5,344,081 of 7,866,535) with ~10% overlap. "
            "Replacing with Mul reduced total to 2,741,387 cycles (~65% improvement). "
            "See SNPE best practices guidelines."
        ),
        "htp_cost": "very_high",
        "equivalent": "x / y  →  x * (1.0 / y)",
    },
    "PReLU": {
        "replacement": "ReLU",
        "rationale": (
            "PReLU ops had low overlap and appeared as background contributors for other "
            "bottleneck ops. Replacing PReLU with ReLU aligns with SNPE best practices. "
            "The showcase model 3 (PReLU) ran 2,789,467 cycles vs 1,374,349 for the "
            "ReLU equivalent."
        ),
        "htp_cost": "high",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════

class HTPResource(str, Enum):
    """Hardware resources used by an HTP op."""
    HVX = "HVX"   # Hexagon Vector eXtensions — SIMD/vector ops
    HMX = "HMX"   # Hexagon Matrix eXtensions — matrix multiply (conv)
    DMA = "DMA"   # Direct Memory Access — data movement


@dataclass
class LintingOpMetrics:
    """Per-op metrics from linting profiling output.

    Corresponds to a single op entry in snpe-diagview linting output:
      N: <op_name> (cycles) : <total_cycles> cycles : DSP
        Wait (Scheduler) time: <wait> cycles
        Overlap time: <overlap> cycles
          <background_op_1>
          ...
        Overlap (wait) time: <overlap_wait> cycles
          <background_op_1>
          ...
        Resources: <HVX, HMX, DMA>
    """
    index: int
    name: str
    total_cycles: int
    wait_cycles: int = 0
    overlap_cycles: int = 0
    overlap_wait_cycles: int = 0
    resources: list[HTPResource] = field(default_factory=list)
    overlap_contributors: list[str] = field(default_factory=list)
    overlap_wait_contributors: list[str] = field(default_factory=list)

    @property
    def overlap_ratio(self) -> float:
        """Fraction of total cycles that had parallel background activity.

        High ratio (near 1.0) = well-parallelized, hard to further optimize via scheduling.
        Low ratio (near 0.0) = bottleneck candidate — op runs mostly alone on main thread.
        """
        if self.total_cycles == 0:
            return 0.0
        return self.overlap_cycles / self.total_cycles

    @property
    def is_bottleneck_candidate(self) -> bool:
        """True when the op has low parallel utilization (optimization target).

        From documentation: ops with low Overlap relative to total cycles indicate
        'good potential for performance gain through optimization'.
        """
        return self.overlap_ratio < BOTTLENECK_OVERLAP_THRESHOLD and self.total_cycles > 0

    @property
    def uses_hvx(self) -> bool:
        return HTPResource.HVX in self.resources

    @property
    def uses_hmx(self) -> bool:
        return HTPResource.HMX in self.resources

    @property
    def uses_dma(self) -> bool:
        return HTPResource.DMA in self.resources


@dataclass
class LintingSubnetProfile:
    """Linting profiling results for a single HTP subnet.

    A network may have multiple subnets (e.g. 3 HTP + 2 CPU).
    Linting only applies to HTP subnets. Each subnet gets its own chrometrace.
    """
    subnet_index: int
    total_cycles: int
    ops: list[LintingOpMetrics] = field(default_factory=list)

    @property
    def bottleneck_ops(self) -> list[LintingOpMetrics]:
        """Ops that are bottleneck candidates (low overlap, non-zero cycles)."""
        return [
            op for op in self.ops
            if op.is_bottleneck_candidate and op.total_cycles > 0
        ]

    @property
    def top_ops_by_cycles(self) -> list[LintingOpMetrics]:
        """Ops sorted by total cycle count descending."""
        return sorted(self.ops, key=lambda op: op.total_cycles, reverse=True)

    def get_op_cycle_fraction(self, op: LintingOpMetrics) -> float:
        """Fraction of total subnet cycles consumed by this op."""
        if self.total_cycles == 0:
            return 0.0
        return op.total_cycles / self.total_cycles

    def significant_ops(
        self,
        threshold: float = SIGNIFICANT_OP_CYCLE_FRACTION,
    ) -> list[LintingOpMetrics]:
        """Ops consuming more than threshold fraction of total cycles."""
        return [
            op for op in self.ops
            if self.get_op_cycle_fraction(op) >= threshold
        ]


@dataclass
class LintingProfile:
    """Full linting profiling result — may span multiple HTP subnets."""
    subnets: list[LintingSubnetProfile] = field(default_factory=list)
    # Non-HTP subnets fall back to Detailed profiling (cycles not available)
    non_htp_subnet_count: int = 0

    @property
    def total_cycles(self) -> int:
        return sum(s.total_cycles for s in self.subnets)

    @property
    def all_bottleneck_ops(self) -> list[tuple[int, LintingOpMetrics]]:
        """(subnet_index, op) pairs for all bottleneck candidates across subnets."""
        result = []
        for subnet in self.subnets:
            for op in subnet.bottleneck_ops:
                result.append((subnet.subnet_index, op))
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Output Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_linting_output(text: str) -> LintingProfile:
    """Parse snpe-diagview linting profiling text output into structured data.

    Handles the standard snpe-diagview format:
      Per-Graph Execution Times:
      HTP Subnet N: <cycles> cycles

      Layer Times:
        0: <op_name> (cycles) : <N> cycles : DSP
          Wait (Scheduler) time: <N> cycles
          Overlap time: <N> cycles
            <contributor_op>
          Overlap (wait) time: <N> cycles
          Resources: HVX, HMX, DMA

    Args:
        text: Full text output from snpe-diagview for a linting run

    Returns:
        LintingProfile with parsed subnet and per-op metrics
    """
    import re

    subnets: list[LintingSubnetProfile] = []
    current_subnet: LintingSubnetProfile | None = None
    current_op: LintingOpMetrics | None = None
    current_section: str | None = None  # "overlap" | "overlap_wait"

    for line in text.splitlines():
        stripped = line.strip()

        # Subnet total: "HTP Subnet 0: 4327266 cycles"
        m = re.match(r"HTP Subnet\s+(\d+):\s+(\d+)\s+cycles", stripped)
        if m:
            if current_op and current_subnet is not None:
                current_subnet.ops.append(current_op)
                current_op = None
            if current_subnet is not None:
                subnets.append(current_subnet)
            current_subnet = LintingSubnetProfile(
                subnet_index=int(m.group(1)),
                total_cycles=int(m.group(2)),
            )
            current_section = None
            continue

        if current_subnet is None:
            continue

        # Op line: "  2: model_convStart_Conv2D:OpId_21 (cycles) : 147075 cycles : DSP"
        m = re.match(r"(\d+):\s+(.+?)\s+\(cycles\)\s*:\s*(\d+)\s+cycles\s*:", stripped)
        if m:
            if current_op is not None:
                current_subnet.ops.append(current_op)
            current_op = LintingOpMetrics(
                index=int(m.group(1)),
                name=m.group(2).strip(),
                total_cycles=int(m.group(3)),
            )
            current_section = None
            continue

        if current_op is None:
            continue

        # Wait line: "Wait (Scheduler) time: 629 cycles"
        m = re.match(r"Wait\s+\([^)]+\)\s+time:\s*(\d+)\s+cycles", stripped)
        if m:
            current_op.wait_cycles = int(m.group(1))
            current_section = None
            continue

        # Overlap (wait) line — check before plain Overlap
        m = re.match(r"Overlap\s+\(wait\)\s+time:\s*(\d+)\s+cycles", stripped)
        if m:
            current_op.overlap_wait_cycles = int(m.group(1))
            current_section = "overlap_wait"
            continue

        # Overlap line: "Overlap time: 85292 cycles"
        m = re.match(r"Overlap\s+time:\s*(\d+)\s+cycles", stripped)
        if m:
            current_op.overlap_cycles = int(m.group(1))
            current_section = "overlap"
            continue

        # Resources line: "Resources: HVX, HMX, DMA"
        m = re.match(r"Resources:\s*(.*)", stripped)
        if m:
            current_section = None
            res_str = m.group(1).strip()
            if res_str:
                for token in res_str.split(","):
                    token = token.strip().upper()
                    try:
                        current_op.resources.append(HTPResource(token))
                    except ValueError:
                        pass  # Unknown resource token — skip
            continue

        # Contributor op names (indented lines under Overlap / Overlap(wait))
        if current_section in ("overlap", "overlap_wait") and stripped:
            # Must not start with a known keyword
            if not re.match(r"(Wait|Overlap|Resources|Layer Times|Per-Graph|\d+:)", stripped):
                if current_section == "overlap":
                    current_op.overlap_contributors.append(stripped)
                else:
                    current_op.overlap_wait_contributors.append(stripped)

    # Flush last op and subnet
    if current_op is not None and current_subnet is not None:
        current_subnet.ops.append(current_op)
    if current_subnet is not None:
        subnets.append(current_subnet)

    return LintingProfile(subnets=subnets)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis & Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def analyze_bottlenecks(
    profile: LintingProfile,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Identify optimization bottlenecks from linting profile data.

    Returns a ranked list of bottleneck ops with optimization suggestions,
    ordered by potential impact (cycle contribution × bottleneck severity).

    Args:
        profile: Parsed LintingProfile
        top_n: Maximum number of bottlenecks to return

    Returns:
        List of dicts with keys: subnet, op_name, total_cycles, cycle_fraction,
        overlap_ratio, is_bottleneck, optimization_hint
    """
    results: list[dict[str, Any]] = []

    for subnet in profile.subnets:
        for op in subnet.ops:
            if op.total_cycles == 0:
                continue

            fraction = subnet.get_op_cycle_fraction(op)
            if fraction < SIGNIFICANT_OP_CYCLE_FRACTION:
                continue

            # Extract base op type from name (e.g. "model_sub_sub:OpId_57" → "Sub")
            op_type = _infer_op_type(op.name)
            substitution = OP_SUBSTITUTIONS.get(op_type)

            hint = None
            if op.is_bottleneck_candidate:
                if substitution:
                    hint = (
                        f"Replace {op_type} with {substitution['replacement']}. "
                        f"{substitution['rationale']}"
                    )
                else:
                    hint = (
                        f"Low parallelism ({op.overlap_ratio:.1%} overlap). "
                        "Consider fusing with adjacent ops or restructuring the graph."
                    )

            results.append({
                "subnet": subnet.subnet_index,
                "op_name": op.name,
                "op_type": op_type,
                "total_cycles": op.total_cycles,
                "cycle_fraction": fraction,
                "overlap_ratio": op.overlap_ratio,
                "is_bottleneck": op.is_bottleneck_candidate,
                "wait_cycles": op.wait_cycles,
                "resources": [r.value for r in op.resources],
                "optimization_hint": hint,
                "known_substitution": substitution,
            })

    # Sort by (is_bottleneck, cycle_fraction) descending
    results.sort(key=lambda x: (x["is_bottleneck"], x["cycle_fraction"]), reverse=True)
    return results[:top_n]


def _infer_op_type(op_name: str) -> str:
    """Infer the SNPE op type from an op name like 'model_sub_sub:OpId_57'.

    Applies heuristics matching the naming patterns in snpe-diagview output.
    """
    name_lower = op_name.lower()
    # Check for known bottleneck types first
    if "_sub_" in name_lower or name_lower.startswith("sub"):
        return "Sub"
    if "realdiv" in name_lower or "div" in name_lower:
        return "RealDiv"
    if "prelu" in name_lower:
        return "PReLU"
    if "relu" in name_lower:
        return "ReLU"
    if "conv" in name_lower:
        return "Conv2D"
    if "add" in name_lower:
        return "Add"
    if "mul" in name_lower:
        return "Mul"
    if "softmax" in name_lower:
        return "Softmax"
    if "pool" in name_lower:
        return "Pool"
    if "matmul" in name_lower or "gemm" in name_lower:
        return "MatMul"
    return "Unknown"


def format_linting_report(
    profile: LintingProfile,
    top_n: int = 5,
    include_overlap_contributors: bool = True,
) -> str:
    """Format a linting profile into a human-readable analysis report.

    Args:
        profile: Parsed LintingProfile
        top_n: Number of top ops to highlight per subnet
        include_overlap_contributors: Include background op contributor lists

    Returns:
        Multi-line report string
    """
    lines: list[str] = []
    lines.append("SNPE HTP Linting Profile Analysis")
    lines.append("=" * 50)
    lines.append(f"Total HTP subnets: {len(profile.subnets)}")
    if profile.non_htp_subnet_count:
        lines.append(
            f"Non-HTP subnets: {profile.non_htp_subnet_count} "
            "(fell back to Detailed profiling)"
        )
    lines.append(f"Total cycles (all subnets): {profile.total_cycles:,}")
    lines.append("")

    bottlenecks = analyze_bottlenecks(profile, top_n=top_n)

    for subnet in profile.subnets:
        lines.append(f"HTP Subnet {subnet.subnet_index}: {subnet.total_cycles:,} cycles")
        lines.append("-" * 40)

        top_ops = subnet.top_ops_by_cycles[:top_n]
        for op in top_ops:
            frac = subnet.get_op_cycle_fraction(op)
            marker = " *** BOTTLENECK" if op.is_bottleneck_candidate else ""
            lines.append(
                f"  [{op.index:2d}] {op.name}"
            )
            lines.append(
                f"       Cycles: {op.total_cycles:>10,}  ({frac:.1%} of subnet){marker}"
            )
            lines.append(
                f"       Wait: {op.wait_cycles:,}  "
                f"Overlap: {op.overlap_cycles:,} ({op.overlap_ratio:.1%})  "
                f"Overlap(wait): {op.overlap_wait_cycles:,}"
            )
            if op.resources:
                lines.append(f"       Resources: {', '.join(r.value for r in op.resources)}")
            if include_overlap_contributors and op.overlap_contributors:
                lines.append(
                    f"       Background ops: {', '.join(op.overlap_contributors[:3])}"
                    + (" ..." if len(op.overlap_contributors) > 3 else "")
                )
            lines.append("")

    if bottlenecks:
        lines.append("Optimization Recommendations")
        lines.append("=" * 50)
        for i, b in enumerate(bottlenecks, 1):
            if b["optimization_hint"]:
                lines.append(f"{i}. {b['op_name']} ({b['cycle_fraction']:.1%} of cycles)")
                lines.append(f"   Overlap: {b['overlap_ratio']:.1%}")
                lines.append(f"   Hint: {b['optimization_hint']}")
                lines.append("")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# snpe-net-run CLI Helper
# ══════════════════════════════════════════════════════════════════════════════

def build_linting_cli_args(
    container_path: str,
    input_list: str,
    output_dir: str = "linting_output",
    *,
    runtime: str = "dsp",
    perf_profile: str = "burst",
) -> list[str]:
    """Build snpe-net-run CLI arguments to run with linting profiling.

    Linting mode is activated via --profiling_level=linting.
    Only meaningful with --runtime dsp (or htp) since Linting is HTP-only.

    Args:
        container_path: Path to .dlc model file
        input_list: Path to input_list.txt file
        output_dir: Directory for profiling output files
        runtime: Inference runtime (should be "dsp" for HTP)
        perf_profile: Performance profile ("burst" recommended for profiling)

    Returns:
        List of CLI argument strings for snpe-net-run

    Example::
        args = build_linting_cli_args("model.dlc", "inputs.txt")
        # → ["snpe-net-run", "--container", "model.dlc", ...]
        subprocess.run(args)
    """
    args = [
        "snpe-net-run",
        "--container", container_path,
        "--input_list", input_list,
        "--output_dir", output_dir,
        "--runtime", runtime,
        "--perf_profile", perf_profile,
        "--profiling_level", LINTING_PROFILING_LEVEL,
    ]
    return args


def build_diagview_chrometrace_args(
    diaglog_path: str,
    output_prefix: str = "linting_trace",
) -> list[str]:
    """Build snpe-diagview CLI arguments to export linting chrometrace JSON files.

    For a network with N HTP subnets, snpe-diagview will generate N separate
    chrometrace files (one per HTP subnet).

    Args:
        diaglog_path: Path to SNPEDiag_*.bin file from snpe-net-run output
        output_prefix: Prefix for output chrometrace JSON files

    Returns:
        CLI argument list for snpe-diagview

    Example::
        args = build_diagview_chrometrace_args("linting_output/SNPEDiag_0.bin")
        # Open resulting JSON in chrome://tracing
    """
    return [
        "snpe-diagview",
        "--input_log", diaglog_path,
        "--chrometrace",
        "--output", output_prefix,
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

LINTING_PROFILE_NOTES: dict[str, Any] = {
    "description": (
        "HTP-only profiling mode. Reports per-op cycle counts instead of microseconds. "
        "No direct cycle→time conversion because of parallelized HTP execution. "
        "Use cycle counts for relative comparisons only."
    ),
    "activation": {
        "cli": "--profiling_level linting",
        "api": "Snpe_SNPEBuilder_SetProfilingLevel(builder, SNPE_PROFILING_LEVEL_LINTING)",
    },
    "metrics": {
        "Wait": "Foreground cycles on main thread since previous op (includes scheduling overhead)",
        "Overlap": "Background cycles: parallel ops running while this op is on main thread",
        "Overlap(wait)": "Background cycles occurring during this op's Wait period",
        "Resources": "Hardware units: HVX (vector), HMX (matrix/conv), DMA (memory)",
    },
    "bottleneck_detection": {
        "rule": "High cycle count + low Overlap ratio = bottleneck candidate",
        "threshold": f"Overlap < {BOTTLENECK_OVERLAP_THRESHOLD:.0%} of total cycles",
        "examples": {
            "sub_op": {
                "before": "4,327,266 total cycles; sub op: 2,165,162 (50%), overlap 21.5%",
                "after": "1,374,349 total cycles (−68%) via branch merge + conv substitution",
            },
            "div_op": {
                "before": "7,866,535 total cycles; div op: 5,344,081 (68%), overlap 10%",
                "after": "2,741,387 total cycles (−65%) via div→mul substitution",
            },
            "prelu_vs_relu": {
                "before": "2,789,467 cycles with PReLU",
                "after": "1,374,349 cycles with ReLU (same as Model 1 Optimized)",
            },
        },
    },
    "chrometrace": {
        "tool": "snpe-diagview --chrometrace",
        "viewer": "chrome://tracing",
        "output": "One JSON file per HTP subnet",
        "caveat": (
            "For multi-subnet models (HTP + non-HTP), only HTP subnets get chrometraces. "
            "3 HTP + 2 CPU subnets → 3 chrometrace files."
        ),
    },
    "caveats": [
        "Linting is HTP-only; non-HTP subnets silently fall back to Detailed profiling",
        "Metrics are averaged across all inference inputs",
        "Background ops waited on by main thread ops are NOT counted in Overlap",
        "Up to 10 contributor op names listed per Overlap entry",
    ],
    "op_substitutions": OP_SUBSTITUTIONS,
}
