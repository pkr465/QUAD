"""SNPE supported network layers matrix.

This module encodes the complete SNPE layer support matrix across all five
runtimes (CPU, GPU, AIP, HTP, DSP) for 134 operations.  It is derived from
the official Qualcomm SNPE SDK documentation and is used by compatibility
checkers and runtime recommendation utilities inside QUAD.

Usage::

    from quad.utils.layer_support import (
        LAYER_SUPPORT_BY_NAME,
        LAYER_SUPPORT_TABLE,
        TOTAL_OPERATIONS,
        find_unsupported_in_model,
        get_runtime_coverage,
        get_supported_ops,
        get_unsupported_ops,
        is_supported,
        recommend_runtimes,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Valid runtime identifiers
# ---------------------------------------------------------------------------

VALID_RUNTIMES: frozenset[str] = frozenset({"cpu", "gpu", "aip", "htp", "dsp"})

# ---------------------------------------------------------------------------
# LayerSupportEntry dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerSupportEntry:
    """Support record for a single SNPE operation across all runtimes."""

    operation: str
    cpu: bool
    gpu: bool
    aip: bool
    htp: bool
    dsp: bool

    def supports(self, runtime: str) -> bool:
        """Return True if this operation is supported on *runtime*.

        Parameters
        ----------
        runtime:
            One of ``"cpu"``, ``"gpu"``, ``"aip"``, ``"htp"``, ``"dsp"``
            (case-insensitive).

        Raises
        ------
        ValueError
            If *runtime* is not a recognised SNPE runtime identifier.
        """
        rt = runtime.lower()
        if rt not in VALID_RUNTIMES:
            raise ValueError(
                f"Unknown runtime {runtime!r}. Valid values: {sorted(VALID_RUNTIMES)}"
            )
        return bool(getattr(self, rt))

    @property
    def supported_runtimes(self) -> list[str]:
        """Return the list of runtime names on which this operation runs."""
        return [rt for rt in ("cpu", "gpu", "aip", "htp", "dsp") if getattr(self, rt)]


# ---------------------------------------------------------------------------
# Full 134-entry support table
# ---------------------------------------------------------------------------
# Columns: operation, cpu, gpu, aip, htp, dsp
# Y = True, N = False

LAYER_SUPPORT_TABLE: list[LayerSupportEntry] = [
    LayerSupportEntry("ArgbToRgb",                    True,  True,  True,  True,  True),
    LayerSupportEntry("Argmax",                       True,  True,  True,  True,  True),
    LayerSupportEntry("Argmin",                       True,  True,  True,  True,  True),
    LayerSupportEntry("AxisAlignedBboxTransform",     True,  False, False, False, False),
    LayerSupportEntry("Batchnorm",                    True,  True,  True,  True,  True),
    LayerSupportEntry("BatchPermutation",             True,  False, False, False, False),
    LayerSupportEntry("BatchToSpace",                 True,  True,  True,  True,  True),
    LayerSupportEntry("BboxTransform",                True,  False, False, False, False),
    LayerSupportEntry("BoxWithNmsLimit",              True,  False, True,  False, True),
    LayerSupportEntry("Cast",                         True,  True,  True,  True,  True),
    LayerSupportEntry("ChannelShuffle",               True,  True,  True,  True,  True),
    LayerSupportEntry("CollectRpnProposals",          True,  False, False, False, False),
    LayerSupportEntry("Concat",                       True,  True,  True,  True,  True),
    LayerSupportEntry("ConstantOfShape",              True,  False, False, False, False),
    LayerSupportEntry("Conv2d",                       True,  True,  True,  True,  True),
    LayerSupportEntry("Conv3d",                       True,  False, True,  True,  True),
    LayerSupportEntry("Convert",                      True,  False, True,  True,  True),
    LayerSupportEntry("Correlation1D",                True,  False, True,  False, True),
    LayerSupportEntry("CropAndResize",                True,  False, False, False, False),
    LayerSupportEntry("CumulativeSum",                True,  False, False, True,  False),
    LayerSupportEntry("DepthToSpace",                 True,  True,  True,  True,  True),
    LayerSupportEntry("DepthWiseConv2d",              True,  True,  True,  True,  True),
    LayerSupportEntry("Dequantize",                   True,  True,  True,  True,  True),
    LayerSupportEntry("DetectionOutput",              True,  True,  True,  True,  True),
    LayerSupportEntry("DistributeFpnProposals",       True,  False, False, False, False),
    LayerSupportEntry("ElementWiseAbs",               True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseAdd",               True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseAnd",               True,  True,  False, True,  False),
    LayerSupportEntry("ElementWiseAsin",              True,  False, False, False, False),
    LayerSupportEntry("ElementWiseAtan",              True,  False, False, True,  False),
    LayerSupportEntry("ElementWiseCeil",              True,  False, True,  True,  True),
    LayerSupportEntry("ElementWiseCos",               True,  True,  False, True,  False),
    LayerSupportEntry("ElementWiseDivide",            True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseEqual",             True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseExp",               True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseFloor",             True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseFloorDiv",          True,  False, True,  True,  True),
    LayerSupportEntry("ElementWiseFmod",              False, False, False, False, False),
    LayerSupportEntry("ElementWiseGreater",           True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseGreaterEqual",      True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseLess",              True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseLessEqual",         True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseLog",               True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseMaximum",           True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseMinimum",           True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseMod",               True,  False, False, False, False),
    LayerSupportEntry("ElementWiseMultiply",          True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseNeg",               True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseNot",               True,  True,  False, True,  False),
    LayerSupportEntry("ElementWiseNotEqual",          True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseOr",                True,  True,  False, False, False),
    LayerSupportEntry("ElementWisePower",             True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseRound",             True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseRsqrt",             True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseSelect",            True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseSign",              True,  False, False, True,  False),
    LayerSupportEntry("ElementWiseSin",               True,  True,  False, True,  False),
    LayerSupportEntry("ElementWiseSoftplus",          True,  False, False, False, False),
    LayerSupportEntry("ElementWiseSquaredDifference", True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseSquareRoot",        True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseSubtract",          True,  True,  True,  True,  True),
    LayerSupportEntry("ElementWiseUnary",             True,  False, False, False, False),
    LayerSupportEntry("ElementWiseXor",               True,  True,  False, False, False),
    LayerSupportEntry("Elu",                          True,  True,  True,  True,  True),
    LayerSupportEntry("ExpandDims",                   True,  True,  False, True,  False),
    LayerSupportEntry("ExtractGlimpse",               True,  False, True,  True,  True),
    LayerSupportEntry("ExtractPatches",               True,  False, False, True,  False),
    LayerSupportEntry("FullyConnected",               True,  True,  True,  True,  True),
    LayerSupportEntry("Gather",                       True,  True,  True,  True,  True),
    LayerSupportEntry("GatherElements",               True,  False, False, True,  False),
    LayerSupportEntry("GatherNd",                     True,  False, False, True,  False),
    LayerSupportEntry("Gelu",                         True,  False, False, True,  False),
    LayerSupportEntry("GenerateProposals",            True,  False, True,  False, True),
    LayerSupportEntry("GridSample",                   True,  False, False, True,  False),
    LayerSupportEntry("GroupNorm",                    True,  False, False, False, False),
    LayerSupportEntry("Gru",                          False, False, False, False, False),
    LayerSupportEntry("HardSwish",                    True,  True,  True,  True,  True),
    LayerSupportEntry("HeatMapMaxKeyPoint",           True,  True,  False, True,  False),
    LayerSupportEntry("ImageProjectionTransform",     True,  False, True,  True,  True),
    LayerSupportEntry("InstanceNorm",                 True,  True,  True,  True,  True),
    LayerSupportEntry("L2Norm",                       True,  True,  True,  True,  True),
    LayerSupportEntry("L2Pool2d",                     True,  True,  False, False, False),
    LayerSupportEntry("LayerNorm",                    True,  True,  False, True,  False),
    LayerSupportEntry("LogSoftmax",                   True,  True,  False, True,  False),
    LayerSupportEntry("Lrn",                          True,  True,  True,  True,  True),
    LayerSupportEntry("Lstm",                         True,  True,  True,  True,  True),
    LayerSupportEntry("MatMul",                       True,  True,  True,  True,  True),
    LayerSupportEntry("Moments",                      True,  False, True,  False, True),
    LayerSupportEntry("MultiClassNms",                True,  False, False, True,  False),
    LayerSupportEntry("NonMaxSuppression",            True,  False, False, True,  False),
    LayerSupportEntry("NonZero",                      True,  False, False, True,  False),
    LayerSupportEntry("Nv12ToRgb",                    True,  True,  True,  True,  True),
    LayerSupportEntry("Nv21ToRgb",                    True,  True,  True,  True,  True),
    LayerSupportEntry("OneHot",                       True,  False, False, True,  False),
    LayerSupportEntry("Pack",                         True,  True,  True,  True,  True),
    LayerSupportEntry("Pad",                          True,  True,  True,  True,  True),
    LayerSupportEntry("PoolAvg2d",                    True,  True,  True,  True,  True),
    LayerSupportEntry("PoolAvg3d",                    True,  False, True,  True,  True),
    LayerSupportEntry("PoolMax2d",                    True,  True,  True,  True,  True),
    LayerSupportEntry("PoolMax3d",                    True,  False, True,  False, True),
    LayerSupportEntry("Prelu",                        True,  True,  True,  True,  True),
    LayerSupportEntry("Quantize",                     True,  False, True,  True,  True),
    LayerSupportEntry("ReduceMax",                    True,  True,  True,  True,  True),
    LayerSupportEntry("ReduceMean",                   True,  True,  True,  True,  True),
    LayerSupportEntry("ReduceMin",                    True,  True,  True,  True,  True),
    LayerSupportEntry("ReduceProd",                   True,  True,  False, False, False),
    LayerSupportEntry("ReduceSum",                    True,  True,  True,  True,  True),
    LayerSupportEntry("Relu",                         True,  True,  True,  True,  True),
    LayerSupportEntry("Relu1",                        False, True,  False, True,  False),
    LayerSupportEntry("Relu6",                        True,  True,  True,  True,  True),
    LayerSupportEntry("ReluMinMax",                   True,  True,  True,  True,  True),
    LayerSupportEntry("Reshape",                      True,  True,  True,  True,  True),
    LayerSupportEntry("Resize",                       True,  False, False, True,  False),
    LayerSupportEntry("ResizeBilinear",               True,  True,  True,  True,  True),
    LayerSupportEntry("ResizeNearestNeighbor",        True,  True,  True,  True,  True),
    LayerSupportEntry("RoiAlign",                     True,  True,  True,  True,  True),
    LayerSupportEntry("RoiPooling",                   True,  False, True,  False, True),
    LayerSupportEntry("ScatterElements",              True,  False, False, True,  False),
    LayerSupportEntry("ScatterNd",                    True,  False, False, True,  False),
    LayerSupportEntry("Shape",                        True,  False, False, False, False),
    LayerSupportEntry("Sigmoid",                      True,  True,  True,  True,  True),
    LayerSupportEntry("Softmax",                      True,  True,  True,  True,  True),
    LayerSupportEntry("SpaceToBatch",                 True,  True,  True,  True,  True),
    LayerSupportEntry("SpaceToDepth",                 True,  True,  True,  True,  True),
    LayerSupportEntry("Split",                        True,  True,  True,  True,  True),
    LayerSupportEntry("Squeeze",                      True,  True,  True,  True,  True),
    LayerSupportEntry("StridedSlice",                 True,  True,  True,  True,  True),
    LayerSupportEntry("Tanh",                         True,  True,  True,  True,  True),
    LayerSupportEntry("Tile",                         True,  True,  True,  True,  True),
    LayerSupportEntry("TopK",                         True,  True,  False, True,  False),
    LayerSupportEntry("Transpose",                    True,  True,  True,  True,  True),
    LayerSupportEntry("TransposeConv2d",              True,  True,  True,  True,  True),
    LayerSupportEntry("TransposeConv3d",              True,  False, False, True,  False),
    LayerSupportEntry("UnPack",                       True,  True,  True,  True,  True),
]

# ---------------------------------------------------------------------------
# Derived lookup structures
# ---------------------------------------------------------------------------

TOTAL_OPERATIONS: int = 134

assert len(LAYER_SUPPORT_TABLE) == TOTAL_OPERATIONS, (
    f"Expected {TOTAL_OPERATIONS} entries, found {len(LAYER_SUPPORT_TABLE)}"
)

LAYER_SUPPORT_BY_NAME: dict[str, LayerSupportEntry] = {
    entry.operation: entry for entry in LAYER_SUPPORT_TABLE
}

# ---------------------------------------------------------------------------
# Public utility functions
# ---------------------------------------------------------------------------


def _validate_runtime(runtime: str) -> str:
    """Normalise and validate a runtime string."""
    rt = runtime.lower()
    if rt not in VALID_RUNTIMES:
        raise ValueError(
            f"Unknown runtime {runtime!r}. Valid values: {sorted(VALID_RUNTIMES)}"
        )
    return rt


def is_supported(operation: str, runtime: str) -> bool:
    """Check if *operation* is supported on *runtime*.

    Parameters
    ----------
    operation:
        SNPE internal operation name, e.g. ``"Conv2d"``.  The lookup is
        case-sensitive and must match the names used in
        :data:`LAYER_SUPPORT_TABLE`.
    runtime:
        One of ``"cpu"``, ``"gpu"``, ``"aip"``, ``"htp"``, ``"dsp"``
        (case-insensitive).

    Returns
    -------
    bool
        ``True`` if the operation is supported on the runtime, ``False``
        otherwise.  Returns ``False`` for unknown operation names rather than
        raising so callers can use this in a simple boolean guard.
    """
    rt = _validate_runtime(runtime)
    entry = LAYER_SUPPORT_BY_NAME.get(operation)
    if entry is None:
        return False
    return bool(getattr(entry, rt))


def get_supported_ops(runtime: str) -> list[str]:
    """Return all operation names that are supported on *runtime*.

    Parameters
    ----------
    runtime:
        One of ``"cpu"``, ``"gpu"``, ``"aip"``, ``"htp"``, ``"dsp"``
        (case-insensitive).
    """
    rt = _validate_runtime(runtime)
    return [entry.operation for entry in LAYER_SUPPORT_TABLE if getattr(entry, rt)]


def get_unsupported_ops(runtime: str) -> list[str]:
    """Return all operation names that are NOT supported on *runtime*.

    Parameters
    ----------
    runtime:
        One of ``"cpu"``, ``"gpu"``, ``"aip"``, ``"htp"``, ``"dsp"``
        (case-insensitive).
    """
    rt = _validate_runtime(runtime)
    return [
        entry.operation for entry in LAYER_SUPPORT_TABLE if not getattr(entry, rt)
    ]


def find_unsupported_in_model(
    model_ops: Sequence[str],
    runtime: str,
) -> list[str]:
    """Find which operations from *model_ops* are NOT supported on *runtime*.

    Unknown operation names (i.e. not present in :data:`LAYER_SUPPORT_TABLE`)
    are treated as unsupported and included in the returned list.

    Parameters
    ----------
    model_ops:
        Sequence of SNPE operation names used by the model.
    runtime:
        Target runtime (``"cpu"``, ``"gpu"``, ``"aip"``, ``"htp"``,
        ``"dsp"``; case-insensitive).

    Returns
    -------
    list[str]
        Operations that cannot run on *runtime*, preserving the input order
        and de-duplicating by first occurrence.
    """
    rt = _validate_runtime(runtime)
    seen: set[str] = set()
    unsupported: list[str] = []
    for op in model_ops:
        if op in seen:
            continue
        seen.add(op)
        entry = LAYER_SUPPORT_BY_NAME.get(op)
        if entry is None or not getattr(entry, rt):
            unsupported.append(op)
    return unsupported


def recommend_runtimes(model_ops: Sequence[str]) -> dict[str, bool]:
    """For each runtime return ``True`` if ALL *model_ops* are supported.

    Parameters
    ----------
    model_ops:
        Sequence of SNPE operation names used by the model.

    Returns
    -------
    dict[str, bool]
        Keys are the five runtime names; value is ``True`` only when every
        operation in *model_ops* is available on that runtime.
    """
    result: dict[str, bool] = {}
    for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
        result[rt] = len(find_unsupported_in_model(model_ops, rt)) == 0
    return result


def get_runtime_coverage(model_ops: Sequence[str]) -> dict[str, float]:
    """For each runtime return the fraction of *model_ops* that are supported.

    Parameters
    ----------
    model_ops:
        Sequence of SNPE operation names used by the model.

    Returns
    -------
    dict[str, float]
        Keys are the five runtime names; value is a float in ``[0.0, 1.0]``
        representing the proportion of *model_ops* that run on that runtime.
        Returns ``1.0`` for every runtime when *model_ops* is empty.
    """
    unique_ops = list(dict.fromkeys(model_ops))  # deduplicate, preserve order
    total = len(unique_ops)
    if total == 0:
        return {rt: 1.0 for rt in ("cpu", "gpu", "aip", "htp", "dsp")}

    result: dict[str, float] = {}
    for rt in ("cpu", "gpu", "aip", "htp", "dsp"):
        supported_count = sum(
            1
            for op in unique_ops
            if is_supported(op, rt)
        )
        result[rt] = supported_count / total
    return result
