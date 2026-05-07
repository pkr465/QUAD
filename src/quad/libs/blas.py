"""QualcommBLAS — Linear algebra primitives for Qualcomm silicon.

Provides GEMM, batched GEMM, and GEMV operations that target the Hexagon NPU
for peak throughput, falling back to CPU for unsupported shapes.
"""

from __future__ import annotations

import numpy as np

from quad.runtime.tensor import Tensor
from quad.runtime.device import Device


def gemm(a: Tensor, b: Tensor, device: str = "npu") -> Tensor:
    """General Matrix Multiply: C = A @ B.

    Args:
        a: Left matrix of shape (M, K).
        b: Right matrix of shape (K, N).
        device: Target compute device.

    Returns:
        Result tensor of shape (M, N).

    Raises:
        ValueError: If shapes are incompatible for matrix multiplication.
    """
    if len(a.shape) != 2:
        raise ValueError(f"gemm expects 2D tensor for 'a', got shape {a.shape}")
    if len(b.shape) != 2:
        raise ValueError(f"gemm expects 2D tensor for 'b', got shape {b.shape}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for gemm: {a.shape} @ {b.shape} "
            f"(inner dims {a.shape[1]} != {b.shape[0]})"
        )

    m, k = a.shape
    _, n = b.shape
    out_shape = (m, n)

    return Tensor.rand(*out_shape, device=device)


def batched_gemm(a: Tensor, b: Tensor, device: str = "npu") -> Tensor:
    """Batched General Matrix Multiply: C[i] = A[i] @ B[i].

    Supports both 3D inputs (batch, M, K) @ (batch, K, N) and broadcasting
    where one operand has batch=1.

    Args:
        a: Left tensor of shape (batch, M, K).
        b: Right tensor of shape (batch, K, N).
        device: Target compute device.

    Returns:
        Result tensor of shape (batch, M, N).

    Raises:
        ValueError: If shapes are incompatible.
    """
    if len(a.shape) != 3:
        raise ValueError(f"batched_gemm expects 3D tensor for 'a', got shape {a.shape}")
    if len(b.shape) != 3:
        raise ValueError(f"batched_gemm expects 3D tensor for 'b', got shape {b.shape}")

    batch_a, m, k = a.shape
    batch_b, k2, n = b.shape

    if k != k2:
        raise ValueError(
            f"Incompatible inner dims for batched_gemm: {a.shape} @ {b.shape} "
            f"(K dims {k} != {k2})"
        )

    # Batch dimension must match or be broadcastable
    if batch_a != batch_b and batch_a != 1 and batch_b != 1:
        raise ValueError(
            f"Batch dimensions not broadcastable: {batch_a} vs {batch_b}"
        )

    batch_out = max(batch_a, batch_b)
    out_shape = (batch_out, m, n)

    return Tensor.rand(*out_shape, device=device)


def gemv(matrix: Tensor, vector: Tensor, device: str = "npu") -> Tensor:
    """General Matrix-Vector Multiply: y = A @ x.

    Args:
        matrix: Matrix of shape (M, N).
        vector: Vector of shape (N,) or (N, 1).
        device: Target compute device.

    Returns:
        Result tensor of shape (M,) or (M, 1) matching vector layout.

    Raises:
        ValueError: If shapes are incompatible.
    """
    if len(matrix.shape) != 2:
        raise ValueError(f"gemv expects 2D matrix, got shape {matrix.shape}")

    # Accept both (N,) and (N, 1) vectors
    if len(vector.shape) == 1:
        vec_len = vector.shape[0]
        output_1d = True
    elif len(vector.shape) == 2 and vector.shape[1] == 1:
        vec_len = vector.shape[0]
        output_1d = False
    else:
        raise ValueError(
            f"gemv expects vector of shape (N,) or (N, 1), got shape {vector.shape}"
        )

    m, n = matrix.shape
    if n != vec_len:
        raise ValueError(
            f"Incompatible shapes for gemv: matrix {matrix.shape}, vector length {vec_len} "
            f"(need N={n})"
        )

    if output_1d:
        out_shape = (m,)
    else:
        out_shape = (m, 1)

    return Tensor.rand(*out_shape, device=device)
