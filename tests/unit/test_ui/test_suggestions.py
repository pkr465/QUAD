"""Tests for the suggestions engine (Phase F)."""

from __future__ import annotations

import pytest

from quad.suggestions import (
    Suggestion,
    suggest_for_workflow,
    suggest_optimisations,
    suggest_power_mode,
    suggest_quantization,
    suggest_runtime,
)


class TestSuggestion:
    def test_to_dict_serialisable(self) -> None:
        s = Suggestion(
            title="x", rationale="y", severity="warning", confidence="high",
            command="z", category="quantization",
        )
        d = s.to_dict()
        assert d["title"] == "x"
        assert d["severity"] == "warning"

    def test_to_markdown_includes_icon(self) -> None:
        warn = Suggestion("x", "y", severity="warning")
        assert "⚠️" in warn.to_markdown()

        crit = Suggestion("x", "y", severity="critical")
        assert "🛑" in crit.to_markdown()

    def test_command_block_rendered(self) -> None:
        s = Suggestion("x", "y", command="quad detect")
        out = s.to_markdown()
        assert "```" in out
        assert "quad detect" in out


class TestSuggestQuantization:
    def test_large_fp32_recommends_int8(self) -> None:
        recs = suggest_quantization(model_size_mb=120, quantization="fp32")
        assert any("INT8" in r.title for r in recs)
        assert any(r.severity == "recommend" for r in recs)

    def test_small_fp32_suggests_int8_as_info(self) -> None:
        recs = suggest_quantization(model_size_mb=10, quantization="fp32")
        assert any("INT8" in r.title for r in recs)

    def test_already_int8_no_quantize_recommendation(self) -> None:
        recs = suggest_quantization(model_size_mb=50, quantization="int8")
        # Already INT8 — shouldn't suggest INT8 again
        assert not any("Quantize to INT8" in r.title for r in recs)

    def test_memory_budget_triggers_int4_suggestion(self) -> None:
        recs = suggest_quantization(
            model_size_mb=200, quantization="int8",
            target_memory_budget_mb=50,
        )
        assert any("INT4" in r.title for r in recs)


class TestSuggestRuntime:
    def test_high_coverage_picks_npu(self) -> None:
        recs = suggest_runtime(
            coverage_pct=98, npu_compatible_ops=98, total_ops=100,
            has_npu=True, has_gpu=True,
        )
        assert any("NPU" in r.title for r in recs)
        assert any("npu" in (r.command or "") for r in recs)

    def test_medium_coverage_recommends_auto(self) -> None:
        recs = suggest_runtime(
            coverage_pct=85, npu_compatible_ops=85, total_ops=100,
            has_npu=True, has_gpu=True,
        )
        assert any('"auto"' in (r.command or "") for r in recs)

    def test_low_coverage_recommends_gpu(self) -> None:
        recs = suggest_runtime(
            coverage_pct=60, npu_compatible_ops=60, total_ops=100,
            has_npu=True, has_gpu=True,
        )
        assert any("GPU" in r.title for r in recs)
        assert any('"gpu"' in (r.command or "") for r in recs)

    def test_no_npu_falls_back(self) -> None:
        recs = suggest_runtime(
            coverage_pct=100, npu_compatible_ops=100, total_ops=100,
            has_npu=False, has_gpu=True,
        )
        assert any("No NPU" in r.title for r in recs)


class TestSuggestPowerMode:
    def test_realtime_recommends_performance(self) -> None:
        recs = suggest_power_mode(use_case="camera")
        assert any("performance" in (r.command or "") for r in recs)

    def test_batch_recommends_efficiency(self) -> None:
        recs = suggest_power_mode(use_case="batch")
        assert any("efficiency" in (r.command or "") for r in recs)

    def test_interactive_default_balanced(self) -> None:
        recs = suggest_power_mode(use_case="interactive")
        assert any("balanced" in (r.command or "") for r in recs)

    def test_battery_warns_for_realtime(self) -> None:
        recs = suggest_power_mode(use_case="camera", on_battery=True)
        assert any("battery" in r.title.lower() or "battery" in r.rationale.lower() for r in recs)
        assert any(r.severity == "warning" for r in recs)


class TestSuggestOptimisations:
    def test_bottleneck_with_hint(self) -> None:
        bottlenecks = [
            {
                "op_type": "Sub",
                "name": "sub_op",
                "overlap_ratio": 0.21,
                "optimization_hint": "Replace with Conv on HTP v68+",
            }
        ]
        recs = suggest_optimisations(bottlenecks=bottlenecks)
        assert len(recs) == 1
        assert "sub_op" in recs[0].title
        assert recs[0].severity == "warning"

    def test_bottleneck_without_hint(self) -> None:
        bottlenecks = [
            {"op_type": "MyOp", "name": "node_3", "overlap_ratio": 0.15}
        ]
        recs = suggest_optimisations(bottlenecks=bottlenecks)
        assert any("MyOp" in r.title or "node_3" in r.title for r in recs)

    def test_no_bottlenecks_suggests_linting_when_detailed(self) -> None:
        recs = suggest_optimisations(bottlenecks=[], profiling_level="detailed")
        assert any("linting" in r.title.lower() for r in recs)

    def test_no_bottlenecks_in_linting_mode_no_recommendation(self) -> None:
        recs = suggest_optimisations(bottlenecks=[], profiling_level="linting")
        # When already linting and nothing found, no recs
        assert recs == []


class TestSuggestForWorkflow:
    def test_full_workflow_combines_categories(self) -> None:
        conversion = {"original_size_mb": 50, "quantization_applied": "fp32"}
        coverage = {
            "coverage_pct": 100, "supported_ops": 14, "total_ops": 14,
        }
        profile = {"linting_layers": [], "profiling_level": "linting"}

        recs = suggest_for_workflow(
            conversion=conversion,
            coverage=coverage,
            profile=profile,
            use_case="camera",
        )
        # Should have at least quantization + runtime + power
        categories = {r.category for r in recs}
        assert "quantization" in categories
        assert "runtime" in categories
        assert "power" in categories
