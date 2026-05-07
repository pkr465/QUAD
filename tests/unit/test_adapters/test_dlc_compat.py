"""Tests for DLC compatibility detection and migration guidance."""

from __future__ import annotations

import pytest

from quad.adapters.dlc_compat import (
    BATCH_DIM_CHANGE_VERSION,
    DLCCompatIssue,
    TensorDimChange,
    check_dlc_compatibility,
    get_tensor_dim_changes,
    is_post_116,
    parse_snpe_version,
    predict_converted_shape,
)


class TestVersionParsing:
    def test_parse_standard_version(self) -> None:
        assert parse_snpe_version("2.45.0") == (2, 45, 0)

    def test_parse_1_16(self) -> None:
        assert parse_snpe_version("1.16.0") == (1, 16, 0)

    def test_parse_with_build(self) -> None:
        # Only take major.minor.patch
        assert parse_snpe_version("2.28.0.250227") == (2, 28, 0)

    def test_batch_dim_change_version_constant(self) -> None:
        assert BATCH_DIM_CHANGE_VERSION == (1, 16, 0)


class TestPost116Detection:
    def test_2x_is_post_116(self) -> None:
        assert is_post_116("2.45.0") is True
        assert is_post_116("2.0.0") is True

    def test_1_16_is_post_116(self) -> None:
        assert is_post_116("1.16.0") is True

    def test_1_15_is_pre_116(self) -> None:
        assert is_post_116("1.15.0") is False

    def test_1_0_is_pre_116(self) -> None:
        assert is_post_116("1.0.0") is False


class TestPredictConvertedShape:
    """Verify shape prediction for pre/post 1.16 conversion."""

    # Post-1.16: batch dimension preserved, NCHW → NHWC
    def test_post_116_unity_batch(self) -> None:
        # {1,3,224,224} → {1,224,224,3}
        result = predict_converted_shape([1, 3, 224, 224], "2.0.0")
        assert result == (1, 224, 224, 3)

    def test_post_116_non_unity_batch(self) -> None:
        # {5,3,224,224} → {5,224,224,3}
        result = predict_converted_shape([5, 3, 224, 224], "2.0.0")
        assert result == (5, 224, 224, 3)
        # Size: 5 * 224 * 224 * 3 = 752,640
        assert 5 * 224 * 224 * 3 == 752_640

    # Pre-1.16: batch dimension dropped, 4D → 3D, NCHW → HWC
    def test_pre_116_unity_batch_drops_batch_dim(self) -> None:
        # {1,3,224,224} → {224,224,3}  (batch silently ignored)
        result = predict_converted_shape([1, 3, 224, 224], "1.15.0")
        assert result == (224, 224, 3)

    def test_pre_116_non_unity_batch_drops_batch_dim(self) -> None:
        # {5,3,224,224} → {224,224,3}  (batch silently ignored!)
        result = predict_converted_shape([5, 3, 224, 224], "1.15.0")
        assert result == (224, 224, 3)

    def test_non_4d_tensor_unaffected(self) -> None:
        """Non-4D tensors are not affected by the batch dim change."""
        shape_2d = [1, 1000]
        assert predict_converted_shape(shape_2d, "1.15.0") == (1, 1000)
        assert predict_converted_shape(shape_2d, "2.0.0") == (1, 1000)

    def test_3d_tensor_unaffected(self) -> None:
        shape_3d = [100, 100, 3]
        assert predict_converted_shape(shape_3d, "1.15.0") == (100, 100, 3)
        assert predict_converted_shape(shape_3d, "2.0.0") == (100, 100, 3)


class TestGetTensorDimChanges:
    def test_detects_batch_dim_change(self) -> None:
        changes = get_tensor_dim_changes({"input": [1, 3, 224, 224]})
        assert len(changes) == 1
        change = changes[0]
        assert change.input_name == "input"
        assert change.pre_116_shape == (224, 224, 3)
        assert change.post_116_shape == (1, 224, 224, 3)

    def test_element_counts(self) -> None:
        changes = get_tensor_dim_changes({"data": [5, 3, 224, 224]})
        assert len(changes) == 1
        c = changes[0]
        assert c.element_count_old == 224 * 224 * 3       # Pre-116: 150,528
        assert c.element_count_new == 5 * 224 * 224 * 3   # Post-116: 752,640

    def test_no_change_for_2d_inputs(self) -> None:
        changes = get_tensor_dim_changes({"output": [1, 1000]})
        assert changes == []

    def test_multiple_inputs(self) -> None:
        changes = get_tensor_dim_changes({
            "image": [1, 3, 224, 224],
            "scores": [1, 1000],    # Not affected
        })
        assert len(changes) == 1
        assert changes[0].input_name == "image"


class TestCheckDLCCompatibility:
    def test_old_sdk_warns_about_batch_dim(self) -> None:
        issues = check_dlc_compatibility(
            sdk_version="1.15.0",
            source_input_shapes={"input": [1, 3, 224, 224]},
        )
        codes = [i.code for i in issues]
        assert "BATCH_DIM_DROPPED" in codes

    def test_old_sdk_errors_on_non_unity_batch(self) -> None:
        issues = check_dlc_compatibility(
            sdk_version="1.15.0",
            source_input_shapes={"input": [5, 3, 224, 224]},
        )
        codes = [i.code for i in issues]
        assert "NON_UNITY_BATCH_DROPPED" in codes

    def test_old_sdk_without_shapes_only_info(self) -> None:
        issues = check_dlc_compatibility(sdk_version="1.15.0")
        codes = [i.code for i in issues]
        assert "OLD_SDK_VERSION" in codes
        assert "BATCH_DIM_DROPPED" not in codes

    def test_new_sdk_no_batch_issues(self) -> None:
        issues = check_dlc_compatibility(
            sdk_version="2.45.0",
            source_input_shapes={"input": [1, 3, 224, 224]},
        )
        codes = [i.code for i in issues]
        assert "BATCH_DIM_DROPPED" not in codes
        assert "NON_UNITY_BATCH_DROPPED" not in codes

    def test_issue_has_recommendation(self) -> None:
        issues = check_dlc_compatibility(
            sdk_version="1.15.0",
            source_input_shapes={"input": [1, 3, 224, 224]},
        )
        for issue in issues:
            assert len(issue.recommendation) > 0

    def test_severity_levels(self) -> None:
        issues = check_dlc_compatibility(
            sdk_version="1.15.0",
            source_input_shapes={"input": [5, 3, 224, 224]},
        )
        severities = {i.code: i.severity for i in issues}
        assert severities.get("NON_UNITY_BATCH_DROPPED") == "error"
