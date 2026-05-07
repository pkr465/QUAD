"""Unit tests for quad.utils.layer_support."""

from __future__ import annotations

import pytest

from quad.utils.layer_support import (
    LAYER_SUPPORT_BY_NAME,
    LAYER_SUPPORT_TABLE,
    TOTAL_OPERATIONS,
    LayerSupportEntry,
    find_unsupported_in_model,
    get_runtime_coverage,
    get_supported_ops,
    get_unsupported_ops,
    is_supported,
    recommend_runtimes,
)


# ---------------------------------------------------------------------------
# Table integrity
# ---------------------------------------------------------------------------


class TestTableIntegrity:
    def test_total_operation_count(self):
        assert len(LAYER_SUPPORT_TABLE) == 134

    def test_total_operations_constant(self):
        assert TOTAL_OPERATIONS == 134

    def test_by_name_has_same_count(self):
        assert len(LAYER_SUPPORT_BY_NAME) == 134

    def test_all_entries_are_layer_support_entry(self):
        for entry in LAYER_SUPPORT_TABLE:
            assert isinstance(entry, LayerSupportEntry)

    def test_no_duplicate_operation_names(self):
        names = [e.operation for e in LAYER_SUPPORT_TABLE]
        assert len(names) == len(set(names))

    def test_by_name_keys_match_operation_field(self):
        for name, entry in LAYER_SUPPORT_BY_NAME.items():
            assert entry.operation == name


# ---------------------------------------------------------------------------
# Spot-checks: all-runtime support
# ---------------------------------------------------------------------------


class TestAllRuntimesSupported:
    """Operations that must be supported on every runtime."""

    @pytest.mark.parametrize(
        "op",
        [
            "Conv2d",
            "Relu",
            "Relu6",
            "Softmax",
            "Sigmoid",
            "MatMul",
            "FullyConnected",
            "Concat",
            "Reshape",
            "Batchnorm",
            "DepthWiseConv2d",
            "Transpose",
            "TransposeConv2d",
            "PoolAvg2d",
            "PoolMax2d",
            "Lstm",
        ],
    )
    def test_supported_on_all_runtimes(self, op: str):
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert is_supported(op, rt), f"{op} should be supported on {rt}"


# ---------------------------------------------------------------------------
# Spot-checks: no-runtime support
# ---------------------------------------------------------------------------


class TestNoRuntimeSupport:
    """Operations that are not supported on any runtime."""

    @pytest.mark.parametrize("op", ["ElementWiseFmod", "Gru"])
    def test_not_supported_on_any_runtime(self, op: str):
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert not is_supported(op, rt), f"{op} should NOT be supported on {rt}"

    @pytest.mark.parametrize("op", ["ElementWiseFmod", "Gru"])
    def test_supported_runtimes_is_empty(self, op: str):
        entry = LAYER_SUPPORT_BY_NAME[op]
        assert entry.supported_runtimes == []


# ---------------------------------------------------------------------------
# CPU runtime
# ---------------------------------------------------------------------------


class TestCpuRuntime:
    """CPU is the most permissive runtime."""

    def test_cpu_unsupported_ops(self):
        unsupported = get_unsupported_ops("cpu")
        # ElementWiseFmod, Gru, and Relu1 are not supported on CPU
        assert set(unsupported) == {"ElementWiseFmod", "Gru", "Relu1"}

    def test_cpu_supported_count(self):
        # 134 total - 3 not on CPU (ElementWiseFmod, Gru, Relu1) = 131
        assert len(get_supported_ops("cpu")) == 131

    def test_relu1_not_on_cpu(self):
        assert not is_supported("Relu1", "cpu")

    def test_elementwise_fmod_not_on_cpu(self):
        assert not is_supported("ElementWiseFmod", "cpu")

    def test_gru_not_on_cpu(self):
        assert not is_supported("Gru", "cpu")


# ---------------------------------------------------------------------------
# GPU runtime
# ---------------------------------------------------------------------------


class TestGpuRuntime:
    """GPU does not support a known set of operations."""

    @pytest.mark.parametrize(
        "op",
        [
            "Gelu",
            "GatherElements",
            "GatherNd",
            "Resize",
            "GridSample",
            "NonMaxSuppression",
            "NonZero",
            "MultiClassNms",
            "ScatterElements",
            "ScatterNd",
            "ElementWiseAsin",
            "ElementWiseAtan",
            "ExtractPatches",
            "GroupNorm",
            "Gru",
            "ElementWiseFmod",
        ],
    )
    def test_not_supported_on_gpu(self, op: str):
        assert not is_supported(op, "gpu"), f"{op} should NOT be supported on GPU"

    def test_relu1_is_supported_on_gpu(self):
        assert is_supported("Relu1", "gpu")

    def test_gpu_supported_count_less_than_cpu(self):
        assert len(get_supported_ops("gpu")) < len(get_supported_ops("cpu"))


# ---------------------------------------------------------------------------
# HTP runtime
# ---------------------------------------------------------------------------


class TestHtpRuntime:
    """HTP is a high-performance accelerator with broad but not complete support."""

    @pytest.mark.parametrize(
        "op",
        [
            "Gelu",
            "NonMaxSuppression",
            "CumulativeSum",
            "ExtractPatches",
            "GatherElements",
            "GatherNd",
            "GridSample",
            "MultiClassNms",
            "NonZero",
            "OneHot",
            "ScatterElements",
            "ScatterNd",
            "TransposeConv3d",
            "Resize",
        ],
    )
    def test_supported_on_htp(self, op: str):
        assert is_supported(op, "htp"), f"{op} should be supported on HTP"

    @pytest.mark.parametrize("op", ["ElementWiseFmod", "Gru"])
    def test_not_supported_on_htp(self, op: str):
        assert not is_supported(op, "htp")


# ---------------------------------------------------------------------------
# DSP runtime
# ---------------------------------------------------------------------------


class TestDspRuntime:
    """DSP (legacy) has a more limited set than HTP."""

    @pytest.mark.parametrize(
        "op",
        [
            "Relu1",
            "Gelu",
            "LogSoftmax",
            "LayerNorm",
            "HeatMapMaxKeyPoint",
            "ExpandDims",
            "ElementWiseCos",
            "ElementWiseSin",
            "ElementWiseAtan",
            "ElementWiseAnd",
            "ElementWiseNot",
            "ElementWiseSign",
            "GatherElements",
            "GatherNd",
            "GridSample",
            "MultiClassNms",
            "NonMaxSuppression",
            "NonZero",
            "OneHot",
            "ScatterElements",
            "ScatterNd",
            "Resize",
            "TopK",
            "TransposeConv3d",
            "ElementWiseFmod",
            "Gru",
        ],
    )
    def test_not_supported_on_dsp(self, op: str):
        assert not is_supported(op, "dsp"), f"{op} should NOT be supported on DSP"

    def test_dsp_supported_count_less_than_htp(self):
        assert len(get_supported_ops("dsp")) <= len(get_supported_ops("htp"))


# ---------------------------------------------------------------------------
# AIP runtime
# ---------------------------------------------------------------------------


class TestAipRuntime:
    """AIP supports a middle-ground set of operations."""

    @pytest.mark.parametrize(
        "op",
        [
            "Conv2d",
            "Relu",
            "Softmax",
            "FullyConnected",
            "Batchnorm",
            "BoxWithNmsLimit",
            "GenerateProposals",
            "RoiPooling",
            "Moments",
        ],
    )
    def test_supported_on_aip(self, op: str):
        assert is_supported(op, "aip"), f"{op} should be supported on AIP"

    @pytest.mark.parametrize(
        "op",
        [
            "Gelu",
            "GridSample",
            "NonMaxSuppression",
            "Relu1",
            "LayerNorm",
            "LogSoftmax",
            "ElementWiseAnd",
            "ElementWiseNot",
        ],
    )
    def test_not_supported_on_aip(self, op: str):
        assert not is_supported(op, "aip"), f"{op} should NOT be supported on AIP"


# ---------------------------------------------------------------------------
# LayerSupportEntry.supports() and .supported_runtimes
# ---------------------------------------------------------------------------


class TestLayerSupportEntryMethods:
    def test_supports_method_conv2d(self):
        entry = LAYER_SUPPORT_BY_NAME["Conv2d"]
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert entry.supports(rt)

    def test_supports_method_case_insensitive(self):
        entry = LAYER_SUPPORT_BY_NAME["Conv2d"]
        assert entry.supports("CPU")
        assert entry.supports("GPU")
        assert entry.supports("HTP")

    def test_supported_runtimes_conv2d(self):
        entry = LAYER_SUPPORT_BY_NAME["Conv2d"]
        assert entry.supported_runtimes == ["cpu", "gpu", "aip", "htp", "dsp"]

    def test_supported_runtimes_gru(self):
        entry = LAYER_SUPPORT_BY_NAME["Gru"]
        assert entry.supported_runtimes == []

    def test_supported_runtimes_relu1(self):
        # Relu1: cpu=N, gpu=Y, aip=N, htp=Y, dsp=N
        entry = LAYER_SUPPORT_BY_NAME["Relu1"]
        assert entry.supported_runtimes == ["gpu", "htp"]

    def test_supported_runtimes_resize(self):
        # Resize: cpu=Y, gpu=N, aip=N, htp=Y, dsp=N
        entry = LAYER_SUPPORT_BY_NAME["Resize"]
        assert entry.supported_runtimes == ["cpu", "htp"]

    def test_supports_invalid_runtime_raises(self):
        entry = LAYER_SUPPORT_BY_NAME["Conv2d"]
        with pytest.raises(ValueError, match="Unknown runtime"):
            entry.supports("tpu")

    def test_frozen_dataclass(self):
        """LayerSupportEntry is immutable."""
        entry = LAYER_SUPPORT_BY_NAME["Conv2d"]
        with pytest.raises((AttributeError, TypeError)):
            entry.cpu = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# is_supported edge cases
# ---------------------------------------------------------------------------


class TestIsSupported:
    def test_unknown_operation_returns_false(self):
        assert is_supported("NonExistentOp", "cpu") is False

    def test_invalid_runtime_raises(self):
        with pytest.raises(ValueError, match="Unknown runtime"):
            is_supported("Conv2d", "fpga")

    def test_case_sensitive_operation_name(self):
        assert is_supported("Conv2d", "cpu") is True
        assert is_supported("conv2d", "cpu") is False  # wrong case → unknown op

    def test_runtime_case_insensitive(self):
        assert is_supported("Conv2d", "CPU") is True
        assert is_supported("Conv2d", "Gpu") is True
        assert is_supported("Conv2d", "HTP") is True


# ---------------------------------------------------------------------------
# get_supported_ops / get_unsupported_ops
# ---------------------------------------------------------------------------


class TestGetSupportedUnsupportedOps:
    def test_invalid_runtime_raises_for_supported(self):
        with pytest.raises(ValueError):
            get_supported_ops("tpu")

    def test_invalid_runtime_raises_for_unsupported(self):
        with pytest.raises(ValueError):
            get_unsupported_ops("tpu")

    def test_supported_plus_unsupported_equals_total(self):
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            s = get_supported_ops(rt)
            u = get_unsupported_ops(rt)
            assert len(s) + len(u) == TOTAL_OPERATIONS

    def test_supported_and_unsupported_are_disjoint(self):
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            s = set(get_supported_ops(rt))
            u = set(get_unsupported_ops(rt))
            assert s.isdisjoint(u)

    def test_supported_ops_returns_list(self):
        result = get_supported_ops("cpu")
        assert isinstance(result, list)

    def test_unsupported_ops_returns_list(self):
        result = get_unsupported_ops("htp")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# find_unsupported_in_model
# ---------------------------------------------------------------------------


class TestFindUnsupportedInModel:
    def test_all_supported_model(self):
        # A model using only universally-supported ops
        model_ops = ["Conv2d", "Relu", "Softmax", "Reshape"]
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert find_unsupported_in_model(model_ops, rt) == []

    def test_finds_gelu_unsupported_on_gpu(self):
        model_ops = ["Conv2d", "Gelu", "Softmax"]
        result = find_unsupported_in_model(model_ops, "gpu")
        assert "Gelu" in result
        assert "Conv2d" not in result
        assert "Softmax" not in result

    def test_finds_gru_unsupported_everywhere(self):
        model_ops = ["Lstm", "Gru", "Relu"]
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            result = find_unsupported_in_model(model_ops, rt)
            assert "Gru" in result
            assert "Lstm" not in result

    def test_unknown_op_treated_as_unsupported(self):
        model_ops = ["Conv2d", "FancyNewOp"]
        result = find_unsupported_in_model(model_ops, "cpu")
        assert "FancyNewOp" in result
        assert "Conv2d" not in result

    def test_empty_model_ops(self):
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert find_unsupported_in_model([], rt) == []

    def test_deduplicates_repeated_ops(self):
        model_ops = ["Gelu", "Gelu", "Gelu"]
        result = find_unsupported_in_model(model_ops, "gpu")
        assert result == ["Gelu"]

    def test_preserves_order_of_first_occurrence(self):
        model_ops = ["Gelu", "Resize", "NonMaxSuppression"]
        result = find_unsupported_in_model(model_ops, "gpu")
        assert result.index("Gelu") < result.index("Resize")

    def test_invalid_runtime_raises(self):
        with pytest.raises(ValueError):
            find_unsupported_in_model(["Conv2d"], "tpu")

    def test_cpu_unsupported_for_transformer_model(self):
        # A model that includes a GRU (never supported) and ElementWiseFmod
        model_ops = ["MatMul", "LayerNorm", "Gelu", "Gru", "ElementWiseFmod"]
        result = find_unsupported_in_model(model_ops, "cpu")
        assert "Gru" in result
        assert "ElementWiseFmod" in result
        assert "MatMul" not in result
        assert "LayerNorm" not in result


# ---------------------------------------------------------------------------
# recommend_runtimes
# ---------------------------------------------------------------------------


class TestRecommendRuntimes:
    def test_all_runtimes_recommended_for_universal_model(self):
        model_ops = ["Conv2d", "Relu", "Softmax"]
        result = recommend_runtimes(model_ops)
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert result[rt] is True, f"{rt} should be recommended"

    def test_cpu_recommended_when_others_are_not(self):
        # Gelu is cpu+htp only; Relu1 is gpu+htp only → only htp covers both
        model_ops = ["Gelu", "Relu1"]
        result = recommend_runtimes(model_ops)
        assert result["htp"] is True
        assert result["cpu"] is False   # cpu doesn't support Relu1
        assert result["gpu"] is False   # gpu doesn't support Gelu
        assert result["dsp"] is False   # dsp supports neither
        assert result["aip"] is False   # aip supports neither

    def test_no_runtime_recommended_for_gru_model(self):
        model_ops = ["Conv2d", "Gru"]
        result = recommend_runtimes(model_ops)
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert result[rt] is False

    def test_returns_dict_with_all_five_runtimes(self):
        result = recommend_runtimes(["Conv2d"])
        assert set(result.keys()) == {"cpu", "gpu", "aip", "htp", "dsp"}

    def test_cpu_most_compatible_for_diverse_model(self):
        # A broad model that avoids the two completely unsupported ops
        all_cpu_ops = [
            e.operation
            for e in LAYER_SUPPORT_TABLE
            if e.cpu and e.operation not in ("ElementWiseFmod", "Gru")
        ]
        result = recommend_runtimes(all_cpu_ops)
        assert result["cpu"] is True

    def test_empty_model_all_runtimes_recommended(self):
        result = recommend_runtimes([])
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert result[rt] is True


# ---------------------------------------------------------------------------
# get_runtime_coverage
# ---------------------------------------------------------------------------


class TestGetRuntimeCoverage:
    def test_full_coverage_for_universal_ops(self):
        model_ops = ["Conv2d", "Relu", "Softmax"]
        result = get_runtime_coverage(model_ops)
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert result[rt] == pytest.approx(1.0)

    def test_zero_coverage_for_unsupported_ops(self):
        model_ops = ["Gru", "ElementWiseFmod"]
        result = get_runtime_coverage(model_ops)
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert result[rt] == pytest.approx(0.0)

    def test_cpu_coverage_is_highest(self):
        # Use a representative mix that stresses all runtimes
        model_ops = [
            "Conv2d", "Relu", "Gelu", "Relu1", "NonMaxSuppression",
            "LayerNorm", "LogSoftmax", "GridSample", "Resize",
        ]
        result = get_runtime_coverage(model_ops)
        # CPU has max coverage in this set (misses Relu1 only)
        assert result["cpu"] >= result["gpu"]
        assert result["cpu"] >= result["aip"]
        assert result["cpu"] >= result["dsp"]

    def test_htp_coverage_higher_than_dsp_for_modern_ops(self):
        # Ops that HTP supports but DSP does not
        model_ops = [
            "Gelu", "NonMaxSuppression", "GridSample", "GatherElements",
            "CumulativeSum", "ExtractPatches", "TransposeConv3d",
        ]
        result = get_runtime_coverage(model_ops)
        assert result["htp"] > result["dsp"]

    def test_partial_coverage(self):
        # Gelu: cpu=Y, gpu=N, aip=N, htp=Y, dsp=N
        # Conv2d: all Y
        model_ops = ["Conv2d", "Gelu"]
        result = get_runtime_coverage(model_ops)
        assert result["cpu"] == pytest.approx(1.0)
        assert result["htp"] == pytest.approx(1.0)
        assert result["gpu"] == pytest.approx(0.5)
        assert result["aip"] == pytest.approx(0.5)
        assert result["dsp"] == pytest.approx(0.5)

    def test_empty_model_returns_ones(self):
        result = get_runtime_coverage([])
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert result[rt] == pytest.approx(1.0)

    def test_returns_dict_with_all_five_runtimes(self):
        result = get_runtime_coverage(["Conv2d"])
        assert set(result.keys()) == {"cpu", "gpu", "aip", "htp", "dsp"}

    def test_deduplicates_model_ops_for_coverage(self):
        result_single = get_runtime_coverage(["Gelu"])
        result_repeated = get_runtime_coverage(["Gelu", "Gelu", "Gelu"])
        for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
            assert result_single[rt] == pytest.approx(result_repeated[rt])

    def test_coverage_values_are_between_zero_and_one(self):
        import random

        random.seed(42)
        sample = random.sample([e.operation for e in LAYER_SUPPORT_TABLE], 20)
        result = get_runtime_coverage(sample)
        for rt, cov in result.items():
            assert 0.0 <= cov <= 1.0, f"Coverage for {rt} out of range: {cov}"
