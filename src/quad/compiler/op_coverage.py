"""ONNX op coverage map for QUAD.

Closes part of GAP_ANALYSIS T1.1: previously the compiler pipeline
returned ``b"QUAD_COMPILED_BINARY"`` as the binary content for every
target with no signal whether the model was actually compilable. This
module enumerates which ONNX ops have a known QUAD lowering for which
target, so the compiler can produce an honest coverage report.

The coverage table tracks ~140 ops across CPU / GPU / Adreno / NPU
(Hexagon HTP) backends. Entries come from the bundled SDK-tools
documentation (``src/quad/sdk_tools/`` — already integrated as
structured data from the SDK docs).

Usage:

    from quad.compiler.op_coverage import compute_coverage

    report = compute_coverage(ir_graph, target="hexagon_v75")
    print(f"{report.coverage_pct:.1f}% of ops supported")
    for unsupported in report.unsupported_ops:
        print(f"  ✗ {unsupported.op_type} ({unsupported.name})")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from quad.compiler.ir import IRGraph, IRNode


# ─── Op support matrix ─────────────────────────────────────────────────────


# Ops supported on the Hexagon HTP backend (NPU). Sourced from
# ``src/quad/sdk_tools/`` which already documents 130+ ops.
HTP_SUPPORTED_OPS: frozenset[str] = frozenset({
    "Abs", "Acos", "Add", "And", "ArgMax", "ArgMin", "Asin", "Atan",
    "AveragePool", "BatchNormalization", "Cast", "Ceil", "Clip",
    "Concat", "Constant", "ConstantOfShape", "Conv", "ConvTranspose",
    "Cos", "Cosh", "DepthToSpace", "DequantizeLinear", "Div", "Dropout",
    "Einsum", "Elu", "Equal", "Erf", "Exp", "Expand", "EyeLike",
    "Flatten", "Floor", "Gather", "GatherElements", "GatherND", "Gelu",
    "Gemm", "GlobalAveragePool", "GlobalMaxPool", "Greater",
    "GreaterOrEqual", "GridSample", "GroupNormalization", "GRU",
    "HardSigmoid", "HardSwish", "Identity", "If", "InstanceNormalization",
    "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual", "Log",
    "LogSoftmax", "Loop", "LpPool", "LRN", "LSTM", "MatMul", "Max",
    "MaxPool", "MaxRoiPool", "Mean", "Min", "Mod", "Mul", "Neg",
    "NonMaxSuppression", "NonZero", "Not", "OneHot", "Or", "Pad", "Pow",
    "PRelu", "QuantizeLinear", "Range", "Reciprocal", "ReduceL1",
    "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
    "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum",
    "ReduceSumSquare", "Relu", "Reshape", "Resize", "RMSNormalization",
    "RoiAlign", "Round", "Scan", "Scatter", "ScatterElements",
    "ScatterND", "Selu", "Shape", "Sigmoid", "Sign", "Sin", "Sinh",
    "Size", "Slice", "Softmax", "Softplus", "Softsign", "SpaceToDepth",
    "Split", "Sqrt", "Squeeze", "STFT", "Sub", "Sum", "Tan", "Tanh",
    "ThresholdedRelu", "Tile", "TopK", "Transpose", "Trilu", "Unsqueeze",
    "Upsample", "Where", "Xor",
})

# CPU runtime (qairt-net-run --use_cpu) supports nearly every ONNX op
# since it's a generic reference implementation. We approximate as
# "everything HTP supports plus a few extras the HTP doesn't".
CPU_SUPPORTED_OPS: frozenset[str] = HTP_SUPPORTED_OPS | frozenset({
    "Compress", "DynamicQuantizeLinear", "MatMulInteger", "QLinearConv",
    "QLinearMatMul", "ReverseSequence", "RoiPool", "SequenceAt",
    "SequenceConstruct", "SequenceEmpty", "SequenceErase",
    "SequenceInsert", "SequenceLength", "Shrink", "StringNormalizer",
    "TfIdfVectorizer", "Tokenizer", "Unique",
})

# GPU (Adreno via OpenCL/Vulkan) — slightly smaller than CPU but
# overlaps heavily with HTP. We use the HTP set as a conservative
# lower bound; in practice some ops fall back to CPU at runtime.
GPU_SUPPORTED_OPS: frozenset[str] = HTP_SUPPORTED_OPS | frozenset({
    "BitShift", "Compress", "DynamicQuantizeLinear", "GlobalLpPool",
    "MatMulInteger", "Multinomial", "QLinearConv", "QLinearMatMul",
    "RandomNormal", "RandomNormalLike", "RandomUniform",
    "RandomUniformLike", "Shrink",
})


def get_supported_ops(target: str) -> frozenset[str]:
    """Return the set of supported op_types for a given target.

    Args:
        target: 'hexagon_*' / 'qnpu_*' (NPU), 'adreno_*' (GPU), 'cpu_*' (CPU),
            or 'auto' (intersection of NPU + CPU — what the orchestrator
            can safely allocate).

    Unknown targets default to the HTP set.
    """
    t = target.lower()
    if "cpu" in t:
        return CPU_SUPPORTED_OPS
    if "gpu" in t or "adreno" in t:
        return GPU_SUPPORTED_OPS
    if t in ("auto", "all"):
        # Intersection — ops safe to allocate to any compute unit
        return HTP_SUPPORTED_OPS & GPU_SUPPORTED_OPS & CPU_SUPPORTED_OPS
    # Default — HTP / qnpu / hexagon
    return HTP_SUPPORTED_OPS


# ─── Coverage report ───────────────────────────────────────────────────────


@dataclass
class UnsupportedOp:
    """A node that has no known lowering on the target."""

    op_type: str
    name: str

    def to_dict(self) -> dict:
        return {"op_type": self.op_type, "name": self.name}


@dataclass
class CoverageReport:
    """Result of computing op coverage for an IR graph against a target."""

    target: str
    total_ops: int
    supported_ops: int
    unsupported_ops: list[UnsupportedOp] = field(default_factory=list)
    op_type_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def coverage_pct(self) -> float:
        if self.total_ops == 0:
            return 0.0
        return 100.0 * self.supported_ops / self.total_ops

    @property
    def is_fully_covered(self) -> bool:
        return self.total_ops > 0 and self.supported_ops == self.total_ops

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "total_ops": self.total_ops,
            "supported_ops": self.supported_ops,
            "coverage_pct": round(self.coverage_pct, 2),
            "is_fully_covered": self.is_fully_covered,
            "unsupported_ops": [u.to_dict() for u in self.unsupported_ops],
            "op_type_breakdown": self.op_type_breakdown,
            "fallback_recommendation": self._fallback_hint(),
        }

    def _fallback_hint(self) -> str:
        if self.is_fully_covered:
            return f"All ops supported on {self.target}; no CPU fallback needed."
        if not self.unsupported_ops:
            return ""
        unique = sorted({u.op_type for u in self.unsupported_ops})
        return (
            f"{len(self.unsupported_ops)} op(s) of types {unique[:5]}{'…' if len(unique) > 5 else ''} "
            f"will fall back to CPU. Consider replacing them with NPU-supported equivalents "
            f"or accept the latency penalty."
        )


def compute_coverage(graph: IRGraph, target: str = "hexagon_v75") -> CoverageReport:
    """Compute op coverage for a graph against a target backend.

    Args:
        graph: IRGraph to analyse
        target: Capability identifier (see ``compiler.capabilities``)
    """
    supported = get_supported_ops(target)
    total = len(graph.nodes)
    supported_count = 0
    unsupported: list[UnsupportedOp] = []
    breakdown: dict[str, int] = {}

    for node in graph.nodes:
        breakdown[node.op_type] = breakdown.get(node.op_type, 0) + 1
        if node.op_type in supported:
            supported_count += 1
        else:
            unsupported.append(UnsupportedOp(op_type=node.op_type, name=node.name))

    return CoverageReport(
        target=target,
        total_ops=total,
        supported_ops=supported_count,
        unsupported_ops=unsupported,
        op_type_breakdown=breakdown,
    )


def compute_coverage_for_targets(
    graph: IRGraph,
    targets: Iterable[str],
) -> dict[str, CoverageReport]:
    """Compute coverage across multiple targets."""
    return {t: compute_coverage(graph, target=t) for t in targets}
