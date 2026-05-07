"""DLC compatibility utilities — version detection and migration guidance.

Documents the rules from SNPE Application Integration Tips:

Batch Dimension Change at Release 1.16.0:
  BEFORE 1.16: Batch dimension was silently dropped during conversion.
    {1,3,224,224} → {224,224,3}   (3D, batch ignored)
    {5,3,224,224} → {224,224,3}   (3D, batch ignored)

  FROM 1.16 onward: Batch dimension is preserved.
    {1,3,224,224} → {1,224,224,3}   (4D, batch preserved)
    {5,3,224,224} → {5,224,224,3}   (4D, batch preserved, size=5*224*224*3=752640)

  Impact: Application code assuming 3D input tensors must be updated to
          expect 4D tensors when loading DLCs converted with 1.16+.

General compatibility:
  - DLCs from any release after 1.0 are forward-compatible.
  - Users experiencing issues after upgrading should re-convert the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# SNPE version where batch dimension handling changed
BATCH_DIM_CHANGE_VERSION = (1, 16, 0)


@dataclass
class DLCCompatIssue:
    """A potential compatibility issue with a DLC file."""
    severity: str           # "error", "warning", "info"
    code: str               # Short machine-readable identifier
    message: str            # Human-readable explanation
    recommendation: str     # What the developer should do


@dataclass
class TensorDimChange:
    """Describes how a tensor shape changes between pre/post 1.16 conversion."""
    input_name: str
    original_shape: tuple[int, ...]     # Shape in source model
    pre_116_shape: tuple[int, ...]      # What old releases produced
    post_116_shape: tuple[int, ...]     # What 1.16+ produces
    element_count_old: int
    element_count_new: int


def parse_snpe_version(version_str: str) -> tuple[int, ...]:
    """Parse an SNPE version string into a comparable tuple.

    Args:
        version_str: e.g. "2.45.0", "1.16.0", "2.28.0.250227"

    Returns:
        Tuple of ints, e.g. (2, 45, 0)
    """
    parts = version_str.strip().split(".")
    result = []
    for part in parts[:3]:  # Only take major.minor.patch
        try:
            result.append(int(part))
        except ValueError:
            result.append(0)
    while len(result) < 3:
        result.append(0)
    return tuple(result)


def is_post_116(version_str: str) -> bool:
    """Return True if the SNPE version is 1.16.0 or later.

    All SNPE2.x versions (2.x.x) are post-1.16.
    """
    v = parse_snpe_version(version_str)
    return v >= BATCH_DIM_CHANGE_VERSION


def predict_converted_shape(
    source_shape: Sequence[int],
    converter_version: str,
) -> tuple[int, ...]:
    """Predict the DLC input tensor shape given the source model shape.

    Handles the pre/post 1.16 batch dimension behavior:
      - Pre-1.16: 4D NCHW input {N,C,H,W} → 3D HWC output {H,W,C}
      - Post-1.16: 4D NCHW input {N,C,H,W} → 4D NHWC output {N,H,W,C}

    This only applies to 4D tensors (images). Non-4D tensors are not affected.

    Args:
        source_shape: Input shape from the source model (e.g. [1, 3, 224, 224])
        converter_version: SNPE version used for conversion

    Returns:
        Predicted tensor shape in the resulting DLC.
    """
    shape = tuple(source_shape)

    if len(shape) != 4:
        # Only 4D tensors are affected by the batch dim change
        return shape

    n, c, h, w = shape  # Assuming NCHW order from source model

    if is_post_116(converter_version):
        # 1.16+: batch dimension preserved, layout converted NCHW → NHWC
        return (n, h, w, c)
    else:
        # Pre-1.16: batch dimension silently dropped, layout NCHW → HWC
        return (h, w, c)


def get_tensor_dim_changes(
    source_shapes: dict[str, Sequence[int]],
) -> list[TensorDimChange]:
    """Compute tensor dimension changes between pre/post 1.16 conversion.

    Useful for identifying which tensors need code updates when migrating
    from old DLCs (pre-1.16) to newly converted DLCs (post-1.16).

    Args:
        source_shapes: Dict of input_name → source model shape

    Returns:
        List of TensorDimChange for any shape that differs between versions.
    """
    changes = []
    for name, shape in source_shapes.items():
        pre = predict_converted_shape(shape, "1.15.0")
        post = predict_converted_shape(shape, "2.0.0")
        if pre != post:
            changes.append(TensorDimChange(
                input_name=name,
                original_shape=tuple(shape),
                pre_116_shape=pre,
                post_116_shape=post,
                element_count_old=_product(pre),
                element_count_new=_product(post),
            ))
    return changes


def check_dlc_compatibility(
    sdk_version: str,
    source_input_shapes: dict[str, Sequence[int]] | None = None,
) -> list[DLCCompatIssue]:
    """Check for known DLC compatibility issues given SDK version.

    Args:
        sdk_version: SNPE SDK version used to convert the DLC
        source_input_shapes: Optional dict of input_name → source model shape
            for detecting batch dimension migration issues.

    Returns:
        List of DLCCompatIssue (empty = no known issues)
    """
    issues: list[DLCCompatIssue] = []
    v = parse_snpe_version(sdk_version)

    # Batch dimension change warning
    if v < BATCH_DIM_CHANGE_VERSION and source_input_shapes:
        changes = get_tensor_dim_changes(source_input_shapes)
        for change in changes:
            n = change.post_116_shape[0]
            issues.append(DLCCompatIssue(
                severity="warning",
                code="BATCH_DIM_DROPPED",
                message=(
                    f"Input '{change.input_name}': DLC converted with SNPE {sdk_version} "
                    f"has shape {change.pre_116_shape} (batch dim dropped). "
                    f"Re-converting with SNPE 1.16+ will produce {change.post_116_shape}. "
                    f"Application code expecting 3D tensors will need updating."
                ),
                recommendation=(
                    f"Re-convert the model with the current SNPE release. "
                    f"Update application code to use 4D tensors: "
                    f"{change.post_116_shape} "
                    f"({change.element_count_new:,} elements, "
                    f"batch={change.post_116_shape[0]})."
                ),
            ))

    # Non-unity batch with old SDK
    if v < BATCH_DIM_CHANGE_VERSION and source_input_shapes:
        for name, shape in source_input_shapes.items():
            if len(shape) == 4 and shape[0] > 1:
                issues.append(DLCCompatIssue(
                    severity="error",
                    code="NON_UNITY_BATCH_DROPPED",
                    message=(
                        f"Input '{name}': Source model has non-unity batch={shape[0]}. "
                        f"SNPE {sdk_version} ignores the batch dimension, producing "
                        f"shape {predict_converted_shape(shape, sdk_version)} "
                        f"instead of the expected {predict_converted_shape(shape, '2.0.0')}. "
                        f"Runtime will receive {_product(predict_converted_shape(shape, sdk_version)):,} "
                        f"values, but {_product(predict_converted_shape(shape, '2.0.0')):,} are expected."
                    ),
                    recommendation=(
                        f"Re-convert with SNPE 1.16+ to get correct batch={shape[0]} support. "
                        f"Expected output shape after re-conversion: "
                        f"{predict_converted_shape(shape, '2.0.0')}."
                    ),
                ))

    # General upgrade recommendation
    if v < (2, 0, 0):
        issues.append(DLCCompatIssue(
            severity="info",
            code="OLD_SDK_VERSION",
            message=(
                f"DLC was converted with SNPE {sdk_version} (pre-SNPE2). "
                f"The DLC is still usable, but re-converting with the current "
                f"SNPE release may improve performance and compatibility."
            ),
            recommendation="Re-convert with the current SNPE/QAIRT release for best results.",
        ))

    return issues


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for d in shape:
        result *= d
    return result
