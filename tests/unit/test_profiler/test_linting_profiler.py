"""Tests for SNPE HTP Linting Profiler."""

from __future__ import annotations

import pytest

from quad.profiler.linting import (
    HTPResource,
    LintingOpMetrics,
    LintingProfile,
    LintingSubnetProfile,
    BOTTLENECK_OVERLAP_THRESHOLD,
    LINTING_PROFILING_LEVEL,
    LINTING_PROFILE_NOTES,
    OP_SUBSTITUTIONS,
    _infer_op_type,
    analyze_bottlenecks,
    build_diagview_chrometrace_args,
    build_linting_cli_args,
    format_linting_report,
    parse_linting_output,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — sample snpe-diagview output text
# ══════════════════════════════════════════════════════════════════════════════

SHOWCASE_MODEL_1_OUTPUT = """\
Per-Graph Execution Times:
---------------
HTP Subnet 0: 4327266 cycles

Layer Times:
---------------
  0: Input OpId_2 (cycles) : 0 cycles : DSP
    Wait (Scheduler) time: 0 cycles
    Overlap time: 0 cycles
    Overlap (wait) time: 0 cycles
    Resources:
  1: OpId_0 (cycles) : 8036 cycles : DSP
    Wait (Scheduler) time: 629 cycles
    Overlap time: 4770 cycles
    Overlap (wait) time: 565 cycles
    Resources:
  2: model_convStart_Conv2D:OpId_21 (cycles) : 147075 cycles : DSP
    Wait (Scheduler) time: 32 cycles
    Overlap time: 85292 cycles
      model_sub_sub:OpId_57
      Output OpId_3
    Overlap (wait) time: 32 cycles
      model_convStart_Conv2D:OpId_21
    Resources: HVX, HMX, DMA
  8: model_sub_sub:OpId_57 (cycles) : 2165162 cycles : DSP
    Wait (Scheduler) time: 0 cycles
    Overlap time: 465046 cycles
      model_sub_sub:OpId_57
      Output OpId_3
    Overlap (wait) time: 0 cycles
    Resources: HVX
  9: model_add_add:OpId_58 (cycles) : 525971 cycles : DSP
    Wait (Scheduler) time: 0 cycles
    Overlap time: 481468 cycles
    Overlap (wait) time: 0 cycles
    Resources: HVX
"""

SHOWCASE_MODEL_2_OUTPUT = """\
Per-Graph Execution Times:
---------------
HTP Subnet 0: 7866535 cycles

Layer Times:
---------------
  0: Input OpId_2 (cycles) : 0 cycles : DSP
    Wait (Scheduler) time: 0 cycles
    Overlap time: 0 cycles
    Overlap (wait) time: 0 cycles
    Resources:
  8: model_tf_op_layer_RealDiv_RealDiv:OpId_57 (cycles) : 5344081 cycles : DSP
    Wait (Scheduler) time: 0 cycles
    Overlap time: 528123 cycles
      model_tf_op_layer_RealDiv_RealDiv:OpId_57
      Output OpId_3
    Overlap (wait) time: 0 cycles
    Resources: HVX
"""

MULTI_SUBNET_OUTPUT = """\
Per-Graph Execution Times:
---------------
HTP Subnet 0: 1000000 cycles
HTP Subnet 1: 2000000 cycles

Layer Times:
---------------
  0: op_a (cycles) : 1000000 cycles : DSP
    Wait (Scheduler) time: 0 cycles
    Overlap time: 100000 cycles
    Overlap (wait) time: 0 cycles
    Resources: HVX
  0: op_b (cycles) : 2000000 cycles : DSP
    Wait (Scheduler) time: 0 cycles
    Overlap time: 200000 cycles
    Overlap (wait) time: 0 cycles
    Resources: HMX, DMA
"""


# ══════════════════════════════════════════════════════════════════════════════
# LintingOpMetrics
# ══════════════════════════════════════════════════════════════════════════════

class TestLintingOpMetrics:
    def test_overlap_ratio_normal(self) -> None:
        op = LintingOpMetrics(index=0, name="op", total_cycles=1000, overlap_cycles=200)
        assert op.overlap_ratio == pytest.approx(0.2)

    def test_overlap_ratio_zero_total(self) -> None:
        op = LintingOpMetrics(index=0, name="op", total_cycles=0)
        assert op.overlap_ratio == 0.0

    def test_is_bottleneck_low_overlap(self) -> None:
        op = LintingOpMetrics(index=0, name="sub_op", total_cycles=2000000, overlap_cycles=400000)
        # overlap_ratio = 0.20 < BOTTLENECK_OVERLAP_THRESHOLD (0.25)
        assert op.is_bottleneck_candidate is True

    def test_not_bottleneck_high_overlap(self) -> None:
        op = LintingOpMetrics(index=0, name="conv_op", total_cycles=500000, overlap_cycles=400000)
        # overlap_ratio = 0.80 > threshold
        assert op.is_bottleneck_candidate is False

    def test_not_bottleneck_zero_cycles(self) -> None:
        op = LintingOpMetrics(index=0, name="input", total_cycles=0)
        assert op.is_bottleneck_candidate is False

    def test_resource_flags(self) -> None:
        op = LintingOpMetrics(
            index=0, name="op", total_cycles=100,
            resources=[HTPResource.HVX, HTPResource.DMA],
        )
        assert op.uses_hvx is True
        assert op.uses_hmx is False
        assert op.uses_dma is True

    def test_sub_op_from_showcase1_is_bottleneck(self) -> None:
        """Reproduces showcase model 1: sub op at 2,165,162 cycles, overlap 465,046 (21.5%)."""
        op = LintingOpMetrics(
            index=8, name="model_sub_sub:OpId_57",
            total_cycles=2165162, overlap_cycles=465046,
            resources=[HTPResource.HVX],
        )
        assert op.overlap_ratio == pytest.approx(465046 / 2165162, rel=1e-4)
        assert op.is_bottleneck_candidate is True  # 21.5% < 25%

    def test_div_op_from_showcase2_is_bottleneck(self) -> None:
        """Reproduces showcase model 2: div op at 5,344,081 cycles, overlap 528,123 (~10%)."""
        op = LintingOpMetrics(
            index=8, name="model_tf_op_layer_RealDiv_RealDiv:OpId_57",
            total_cycles=5344081, overlap_cycles=528123,
            resources=[HTPResource.HVX],
        )
        assert op.overlap_ratio == pytest.approx(528123 / 5344081, rel=1e-4)
        assert op.is_bottleneck_candidate is True  # ~10% << 25%


# ══════════════════════════════════════════════════════════════════════════════
# LintingSubnetProfile
# ══════════════════════════════════════════════════════════════════════════════

class TestLintingSubnetProfile:
    def _make_subnet(self) -> LintingSubnetProfile:
        subnet = LintingSubnetProfile(subnet_index=0, total_cycles=4327266)
        subnet.ops = [
            LintingOpMetrics(index=0, name="Input", total_cycles=0),
            LintingOpMetrics(index=1, name="OpId_0", total_cycles=8036, overlap_cycles=4770),
            LintingOpMetrics(index=2, name="model_convStart_Conv2D:OpId_21",
                             total_cycles=147075, overlap_cycles=85292,
                             resources=[HTPResource.HVX, HTPResource.HMX, HTPResource.DMA]),
            LintingOpMetrics(index=8, name="model_sub_sub:OpId_57",
                             total_cycles=2165162, overlap_cycles=465046,
                             resources=[HTPResource.HVX]),
            LintingOpMetrics(index=9, name="model_add_add:OpId_58",
                             total_cycles=525971, overlap_cycles=481468,
                             resources=[HTPResource.HVX]),
        ]
        return subnet

    def test_bottleneck_ops_excludes_well_parallelized(self) -> None:
        subnet = self._make_subnet()
        bottlenecks = subnet.bottleneck_ops
        names = [op.name for op in bottlenecks]
        assert "model_sub_sub:OpId_57" in names
        # add op: 481468/525971 = 91.5% overlap → not a bottleneck
        assert "model_add_add:OpId_58" not in names

    def test_top_ops_sorted_descending(self) -> None:
        subnet = self._make_subnet()
        top = subnet.top_ops_by_cycles
        cycles = [op.total_cycles for op in top]
        assert cycles == sorted(cycles, reverse=True)

    def test_cycle_fraction_sub_op(self) -> None:
        subnet = self._make_subnet()
        sub_op = next(op for op in subnet.ops if "sub_sub" in op.name)
        frac = subnet.get_op_cycle_fraction(sub_op)
        # 2165162 / 4327266 ≈ 0.50
        assert frac == pytest.approx(2165162 / 4327266, rel=1e-3)

    def test_significant_ops_threshold(self) -> None:
        subnet = self._make_subnet()
        sig = subnet.significant_ops(threshold=0.05)
        # sub (~50%), add (~12%), conv (~3.4% — below threshold), OpId_0 (~0.2%)
        names = [op.name for op in sig]
        assert "model_sub_sub:OpId_57" in names
        assert "model_add_add:OpId_58" in names
        assert "Input" not in names  # zero cycles

    def test_cycle_fraction_zero_total(self) -> None:
        subnet = LintingSubnetProfile(subnet_index=0, total_cycles=0)
        op = LintingOpMetrics(index=0, name="op", total_cycles=100)
        assert subnet.get_op_cycle_fraction(op) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Parser
# ══════════════════════════════════════════════════════════════════════════════

class TestParseLintingOutput:
    def test_showcase_model_1_total_cycles(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        assert len(profile.subnets) == 1
        assert profile.subnets[0].total_cycles == 4327266

    def test_showcase_model_1_op_count(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        assert len(profile.subnets[0].ops) == 5

    def test_showcase_model_1_sub_op_parsed(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        sub_op = next(
            op for op in profile.subnets[0].ops
            if "sub_sub" in op.name
        )
        assert sub_op.total_cycles == 2165162
        assert sub_op.wait_cycles == 0
        assert sub_op.overlap_cycles == 465046
        assert sub_op.overlap_wait_cycles == 0
        assert HTPResource.HVX in sub_op.resources
        assert HTPResource.HMX not in sub_op.resources

    def test_showcase_model_1_conv_resources(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        conv_op = next(
            op for op in profile.subnets[0].ops
            if "convStart" in op.name
        )
        assert HTPResource.HVX in conv_op.resources
        assert HTPResource.HMX in conv_op.resources
        assert HTPResource.DMA in conv_op.resources

    def test_showcase_model_1_conv_overlap_contributors(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        conv_op = next(
            op for op in profile.subnets[0].ops
            if "convStart" in op.name
        )
        assert any("sub_sub" in c for c in conv_op.overlap_contributors)

    def test_showcase_model_2_div_op(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_2_OUTPUT)
        assert profile.subnets[0].total_cycles == 7866535
        div_op = next(
            op for op in profile.subnets[0].ops
            if "RealDiv" in op.name
        )
        assert div_op.total_cycles == 5344081
        assert div_op.overlap_cycles == 528123

    def test_empty_input(self) -> None:
        profile = parse_linting_output("")
        assert profile.subnets == []
        assert profile.total_cycles == 0

    def test_input_op_zero_cycles(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        input_op = profile.subnets[0].ops[0]
        assert input_op.name == "Input OpId_2"
        assert input_op.total_cycles == 0

    def test_total_cycles_sum(self) -> None:
        profile = parse_linting_output(MULTI_SUBNET_OUTPUT)
        assert profile.total_cycles == 3000000


# ══════════════════════════════════════════════════════════════════════════════
# Bottleneck Analysis
# ══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeBottlenecks:
    def test_sub_op_identified_as_bottleneck(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        results = analyze_bottlenecks(profile, top_n=10)
        bottleneck_names = [r["op_name"] for r in results if r["is_bottleneck"]]
        assert any("sub_sub" in name for name in bottleneck_names)

    def test_sub_op_has_substitution_hint(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        results = analyze_bottlenecks(profile, top_n=10)
        sub_result = next(
            (r for r in results if "sub_sub" in r["op_name"]), None
        )
        assert sub_result is not None
        assert sub_result["is_bottleneck"] is True
        assert sub_result["optimization_hint"] is not None
        assert "Conv" in sub_result["optimization_hint"] or "sub" in sub_result["optimization_hint"].lower()

    def test_div_op_has_mul_hint(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_2_OUTPUT)
        results = analyze_bottlenecks(profile, top_n=10)
        div_result = next(
            (r for r in results if "RealDiv" in r["op_name"]), None
        )
        assert div_result is not None
        assert div_result["is_bottleneck"] is True
        assert "mul" in (div_result["optimization_hint"] or "").lower() \
            or "Mul" in (div_result["known_substitution"] or {}).get("replacement", "")

    def test_well_parallelized_ops_not_bottleneck(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        results = analyze_bottlenecks(profile, top_n=10)
        # add op has 91.5% overlap — not a bottleneck
        add_result = next((r for r in results if "add_add" in r["op_name"]), None)
        if add_result:
            assert add_result["is_bottleneck"] is False

    def test_top_n_limit(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        results = analyze_bottlenecks(profile, top_n=2)
        assert len(results) <= 2

    def test_cycle_fraction_values(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        results = analyze_bottlenecks(profile, top_n=10)
        for r in results:
            assert 0.0 <= r["cycle_fraction"] <= 1.0
            assert 0.0 <= r["overlap_ratio"] <= 1.0

    def test_empty_profile(self) -> None:
        profile = LintingProfile()
        results = analyze_bottlenecks(profile)
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# Op Type Inference
# ══════════════════════════════════════════════════════════════════════════════

class TestInferOpType:
    def test_sub_op(self) -> None:
        assert _infer_op_type("model_sub_sub:OpId_57") == "Sub"

    def test_realdiv_op(self) -> None:
        assert _infer_op_type("model_tf_op_layer_RealDiv_RealDiv:OpId_57") == "RealDiv"

    def test_prelu_op(self) -> None:
        assert _infer_op_type("model_preluCombined1_add:OpId_37") == "PReLU"

    def test_relu_op(self) -> None:
        assert _infer_op_type("model_relu:OpId_10") == "ReLU"

    def test_conv_op(self) -> None:
        assert _infer_op_type("model_convStart_Conv2D:OpId_21") == "Conv2D"

    def test_add_op(self) -> None:
        assert _infer_op_type("model_add_add:OpId_58") == "Add"

    def test_mul_op(self) -> None:
        assert _infer_op_type("model_multiply_mul:OpId_57") == "Mul"

    def test_unknown_op(self) -> None:
        assert _infer_op_type("some_exotic_op:OpId_99") == "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# CLI Builders
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildLintingCliArgs:
    def test_profiling_level_present(self) -> None:
        args = build_linting_cli_args("model.dlc", "inputs.txt")
        assert "--profiling_level" in args
        idx = args.index("--profiling_level")
        assert args[idx + 1] == "linting"

    def test_container_arg(self) -> None:
        args = build_linting_cli_args("my_model.dlc", "inputs.txt")
        assert "--container" in args
        idx = args.index("--container")
        assert args[idx + 1] == "my_model.dlc"

    def test_input_list_arg(self) -> None:
        args = build_linting_cli_args("model.dlc", "my_inputs.txt")
        idx = args.index("--input_list")
        assert args[idx + 1] == "my_inputs.txt"

    def test_runtime_dsp(self) -> None:
        args = build_linting_cli_args("model.dlc", "inputs.txt", runtime="dsp")
        idx = args.index("--runtime")
        assert args[idx + 1] == "dsp"

    def test_custom_output_dir(self) -> None:
        args = build_linting_cli_args("model.dlc", "inputs.txt", output_dir="/tmp/lint_out")
        idx = args.index("--output_dir")
        assert args[idx + 1] == "/tmp/lint_out"

    def test_first_token_is_snpe_net_run(self) -> None:
        args = build_linting_cli_args("model.dlc", "inputs.txt")
        assert args[0] == "snpe-net-run"


class TestBuildDiagviewChrometraceArgs:
    def test_chrometrace_flag_present(self) -> None:
        args = build_diagview_chrometrace_args("SNPEDiag_0.bin")
        assert "--chrometrace" in args

    def test_input_log_arg(self) -> None:
        args = build_diagview_chrometrace_args("SNPEDiag_0.bin")
        idx = args.index("--input_log")
        assert args[idx + 1] == "SNPEDiag_0.bin"

    def test_first_token_is_snpe_diagview(self) -> None:
        args = build_diagview_chrometrace_args("diag.bin")
        assert args[0] == "snpe-diagview"


# ══════════════════════════════════════════════════════════════════════════════
# Format Report
# ══════════════════════════════════════════════════════════════════════════════

class TestFormatLintingReport:
    def test_report_contains_subnet_cycles(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        report = format_linting_report(profile)
        assert "4,327,266" in report

    def test_report_contains_bottleneck_marker(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        report = format_linting_report(profile)
        assert "BOTTLENECK" in report

    def test_report_contains_recommendations(self) -> None:
        profile = parse_linting_output(SHOWCASE_MODEL_1_OUTPUT)
        report = format_linting_report(profile)
        assert "Optimization Recommendations" in report or "Hint:" in report

    def test_empty_profile_report(self) -> None:
        report = format_linting_report(LintingProfile())
        assert "SNPE HTP Linting" in report
        assert "0" in report  # total subnets


# ══════════════════════════════════════════════════════════════════════════════
# Notes & Constants
# ══════════════════════════════════════════════════════════════════════════════

class TestLintingProfileNotes:
    def test_activation_keys(self) -> None:
        activation = LINTING_PROFILE_NOTES["activation"]
        assert "cli" in activation
        assert "linting" in activation["cli"]
        assert "api" in activation
        assert "SNPE_PROFILING_LEVEL_LINTING" in activation["api"]

    def test_metrics_keys(self) -> None:
        metrics = LINTING_PROFILE_NOTES["metrics"]
        for key in ("Wait", "Overlap", "Overlap(wait)", "Resources"):
            assert key in metrics

    def test_op_substitutions_has_div_and_sub(self) -> None:
        assert "Sub" in OP_SUBSTITUTIONS
        assert "RealDiv" in OP_SUBSTITUTIONS
        assert "PReLU" in OP_SUBSTITUTIONS

    def test_sub_substitution_replacement(self) -> None:
        assert "Conv" in OP_SUBSTITUTIONS["Sub"]["replacement"]

    def test_div_substitution_replacement(self) -> None:
        assert "Mul" in OP_SUBSTITUTIONS["RealDiv"]["replacement"] or \
               "mul" in OP_SUBSTITUTIONS["RealDiv"]["equivalent"]

    def test_chrometrace_notes(self) -> None:
        ct = LINTING_PROFILE_NOTES["chrometrace"]
        assert "snpe-diagview" in ct["tool"]
        assert "chrome" in ct["viewer"]

    def test_caveats_list(self) -> None:
        caveats = LINTING_PROFILE_NOTES["caveats"]
        assert isinstance(caveats, list)
        assert len(caveats) >= 3
        assert any("HTP" in c for c in caveats)

    def test_profiling_level_constant(self) -> None:
        assert LINTING_PROFILING_LEVEL == "linting"

    def test_bottleneck_threshold_range(self) -> None:
        assert 0.0 < BOTTLENECK_OVERLAP_THRESHOLD < 1.0

    def test_example_improvements_documented(self) -> None:
        """Validate the documented cycle improvement numbers from showcase models."""
        examples = LINTING_PROFILE_NOTES["bottleneck_detection"]["examples"]
        # sub op example: before 4327266, after 1374349
        sub_before = examples["sub_op"]["before"]
        assert "4,327,266" in sub_before or "4327266" in sub_before
        # div op example: before 7866535, after 2741387
        div_before = examples["div_op"]["before"]
        assert "7,866,535" in div_before or "7866535" in div_before
