"""Tests for ONNX op-coverage reporting (T1.1 partial)."""

from __future__ import annotations

import pytest

from quad.compiler.ir import IRGraph, IRNode, IRTensor
from quad.compiler.op_coverage import (
    CPU_SUPPORTED_OPS,
    GPU_SUPPORTED_OPS,
    HTP_SUPPORTED_OPS,
    CoverageReport,
    UnsupportedOp,
    compute_coverage,
    compute_coverage_for_targets,
    get_supported_ops,
)


def _make_graph(op_types: list[str], name: str = "test") -> IRGraph:
    """Build an IRGraph with one node per op_type."""
    g = IRGraph(name=name)
    g.inputs = [IRTensor(name="input", shape=[1, 3, 224, 224])]
    g.outputs = [IRTensor(name="output", shape=[1, 1000])]
    for i, op in enumerate(op_types):
        g.nodes.append(
            IRNode(
                name=f"node_{i}",
                op_type=op,
                inputs=["x"],
                outputs=[f"y{i}"],
            )
        )
    return g


# ─── Op support matrix ─────────────────────────────────────────────────────


class TestSupportedOps:
    def test_htp_includes_core_ops(self) -> None:
        for op in ("Conv", "Relu", "MatMul", "Add", "Reshape"):
            assert op in HTP_SUPPORTED_OPS, f"{op} should be HTP-supported"

    def test_cpu_is_superset_of_htp(self) -> None:
        assert HTP_SUPPORTED_OPS <= CPU_SUPPORTED_OPS

    def test_gpu_is_superset_of_htp(self) -> None:
        assert HTP_SUPPORTED_OPS <= GPU_SUPPORTED_OPS

    def test_get_supported_ops_for_cpu(self) -> None:
        s = get_supported_ops("cpu")
        assert s == CPU_SUPPORTED_OPS

    def test_get_supported_ops_for_npu_default(self) -> None:
        s = get_supported_ops("hexagon_v75")
        assert s == HTP_SUPPORTED_OPS

    def test_get_supported_ops_for_gpu(self) -> None:
        s = get_supported_ops("adreno_x1_85")
        assert s == GPU_SUPPORTED_OPS

    def test_auto_target_is_intersection(self) -> None:
        s = get_supported_ops("auto")
        # auto = intersection of NPU + GPU + CPU
        assert s <= HTP_SUPPORTED_OPS
        assert s <= GPU_SUPPORTED_OPS
        assert s <= CPU_SUPPORTED_OPS


# ─── compute_coverage ─────────────────────────────────────────────────────


class TestComputeCoverage:
    def test_fully_covered(self) -> None:
        g = _make_graph(["Conv", "Relu", "MatMul", "Add"])
        report = compute_coverage(g, target="hexagon_v75")
        assert report.total_ops == 4
        assert report.supported_ops == 4
        assert report.coverage_pct == 100.0
        assert report.is_fully_covered
        assert report.unsupported_ops == []

    def test_partially_covered(self) -> None:
        g = _make_graph(["Conv", "Relu", "CustomOpThatDoesntExist", "MatMul"])
        report = compute_coverage(g, target="hexagon_v75")
        assert report.total_ops == 4
        assert report.supported_ops == 3
        assert 70 < report.coverage_pct < 80
        assert not report.is_fully_covered
        assert len(report.unsupported_ops) == 1
        assert report.unsupported_ops[0].op_type == "CustomOpThatDoesntExist"

    def test_zero_ops(self) -> None:
        g = IRGraph(name="empty")
        report = compute_coverage(g)
        assert report.total_ops == 0
        assert report.coverage_pct == 0.0
        assert not report.is_fully_covered

    def test_op_type_breakdown(self) -> None:
        g = _make_graph(["Conv", "Conv", "Conv", "Relu", "Relu"])
        report = compute_coverage(g, target="hexagon_v75")
        assert report.op_type_breakdown == {"Conv": 3, "Relu": 2}

    def test_to_dict_structure(self) -> None:
        g = _make_graph(["Conv", "Relu"])
        report = compute_coverage(g, target="hexagon_v75")
        d = report.to_dict()
        assert d["target"] == "hexagon_v75"
        assert d["total_ops"] == 2
        assert d["coverage_pct"] == 100.0
        assert d["is_fully_covered"] is True
        assert "op_type_breakdown" in d
        assert "fallback_recommendation" in d
        assert "unsupported_ops" in d

    def test_fallback_recommendation_when_fully_covered(self) -> None:
        g = _make_graph(["Conv", "Relu"])
        report = compute_coverage(g, target="hexagon_v75")
        assert "All ops supported" in report._fallback_hint()

    def test_fallback_recommendation_lists_unsupported(self) -> None:
        g = _make_graph(["Conv", "MagicCustomOp"])
        report = compute_coverage(g, target="hexagon_v75")
        hint = report._fallback_hint()
        assert "MagicCustomOp" in hint
        assert "fall back to CPU" in hint


class TestComputeCoverageMultiTarget:
    def test_multiple_targets(self) -> None:
        g = _make_graph(["Conv", "Relu", "MatMul"])
        results = compute_coverage_for_targets(g, ["hexagon_v75", "cpu_oryon"])
        assert "hexagon_v75" in results
        assert "cpu_oryon" in results
        assert results["hexagon_v75"].total_ops == 3
        assert results["cpu_oryon"].total_ops == 3
