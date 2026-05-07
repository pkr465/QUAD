"""QUAD Hardware Primitives — Hexagon HVX intrinsics (mock implementations).

Provides Python-callable equivalents of Hexagon HVX hardware instructions.
In mock mode, these operate on numpy arrays. In real mode, they map to
actual HVX vector instructions via the compiler backend.

These primitives are analogous to CUDA intrinsics (__syncthreads, __shfl, etc.)
but for Qualcomm's Hexagon Vector eXtensions (HVX).

Usage:
    from quad.kernels.primitives import hvx_vload, hvx_vadd, barrier

    vec_a = hvx_vload(tensor, offset=0)
    vec_b = hvx_vload(tensor, offset=128)
    result = hvx_vadd(vec_a, vec_b)
    barrier()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# HVX vector width: 128 bytes (1024 bits)
HVX_VECTOR_BYTES = 128
HVX_VECTOR_ELEMENTS_F32 = HVX_VECTOR_BYTES // 4  # 32 float32 elements


@dataclass
class VTCMBuffer:
    """Tightly Coupled Memory (VTCM) buffer for scratchpad allocation.

    VTCM is on-chip SRAM with single-cycle access latency.
    Used for high-bandwidth data staging between DMA and HVX.
    """

    size_bytes: int
    data: np.ndarray = field(init=False)
    allocated: bool = field(default=True, init=False)

    def __post_init__(self):
        # Allocate as byte array
        self.data = np.zeros(self.size_bytes, dtype=np.uint8)

    def read(self, offset: int, nbytes: int) -> np.ndarray:
        """Read bytes from VTCM buffer."""
        return self.data[offset : offset + nbytes].copy()

    def write(self, offset: int, data: np.ndarray) -> None:
        """Write bytes to VTCM buffer."""
        flat = data.view(np.uint8).flatten()
        self.data[offset : offset + len(flat)] = flat

    def free(self) -> None:
        """Release the VTCM allocation."""
        self.allocated = False
        self.data = np.zeros(0, dtype=np.uint8)

    def __repr__(self) -> str:
        return f"VTCMBuffer(size={self.size_bytes}, allocated={self.allocated})"


@dataclass
class DMATransfer:
    """Asynchronous DMA transfer descriptor.

    Models an in-flight DMA operation between system memory and VTCM
    or between different memory regions.
    """

    src: Any
    dst: Any
    size_bytes: int
    completed: bool = field(default=False, init=False)

    def wait(self) -> None:
        """Block until DMA transfer completes."""
        # In mock mode, DMA is always instant
        self.completed = True

    @property
    def is_done(self) -> bool:
        return self.completed

    def __repr__(self) -> str:
        return f"DMATransfer(size={self.size_bytes}, completed={self.completed})"


# --- Vector Load/Store ---


def hvx_vload(tensor: np.ndarray, offset: int = 0) -> np.ndarray:
    """Vector load — read 128 bytes (one HVX vector) from memory.

    Loads a contiguous vector of HVX_VECTOR_BYTES bytes starting at
    the given byte offset in the flattened tensor.

    Args:
        tensor: Source numpy array.
        offset: Byte offset to start reading from.

    Returns:
        numpy array containing one HVX vector width of data.
    """
    flat = tensor.flatten().view(np.uint8)
    end = min(offset + HVX_VECTOR_BYTES, len(flat))
    vec = np.zeros(HVX_VECTOR_BYTES, dtype=np.uint8)
    vec[: end - offset] = flat[offset:end]
    # Return as float32 vector (32 elements)
    return vec.view(np.float32)


def hvx_vstore(tensor: np.ndarray, offset: int, vec: np.ndarray) -> None:
    """Vector store — write 128 bytes (one HVX vector) to memory.

    Args:
        tensor: Destination numpy array (modified in-place).
        offset: Byte offset to start writing at.
        vec: HVX vector data to store.
    """
    flat = tensor.flatten().view(np.uint8)
    vec_bytes = vec.view(np.uint8)
    end = min(offset + HVX_VECTOR_BYTES, len(flat))
    n = end - offset
    flat[offset:end] = vec_bytes[:n]


# --- Vector Arithmetic ---


def hvx_vadd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector addition — element-wise add of two HVX vectors.

    Args:
        a: First vector (float32 array).
        b: Second vector (float32 array).

    Returns:
        Element-wise sum as float32 array.
    """
    return (a.astype(np.float32) + b.astype(np.float32)).astype(np.float32)


def hvx_vsub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector subtraction — element-wise subtract.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Element-wise difference.
    """
    return (a.astype(np.float32) - b.astype(np.float32)).astype(np.float32)


def hvx_vmpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector multiply — element-wise multiplication of two HVX vectors.

    Args:
        a: First vector (float32 array).
        b: Second vector (float32 array).

    Returns:
        Element-wise product as float32 array.
    """
    return (a.astype(np.float32) * b.astype(np.float32)).astype(np.float32)


def hvx_vmax(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector max — element-wise maximum.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Element-wise maximum.
    """
    return np.maximum(a.astype(np.float32), b.astype(np.float32)).astype(np.float32)


def hvx_vmin(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vector min — element-wise minimum.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Element-wise minimum.
    """
    return np.minimum(a.astype(np.float32), b.astype(np.float32)).astype(np.float32)


# --- Shuffle/Permute ---


def hvx_shuffle(tensor: np.ndarray, pattern: np.ndarray | list[int]) -> np.ndarray:
    """Cross-lane shuffle — permute elements within an HVX vector.

    Analogous to CUDA's __shfl_sync. Rearranges elements according to
    the index pattern provided.

    Args:
        tensor: Input vector to shuffle.
        pattern: Array of indices specifying the permutation.
                 pattern[i] = j means output[i] = input[j].

    Returns:
        Shuffled vector.
    """
    tensor_f32 = tensor.flatten().astype(np.float32)
    pattern_arr = np.asarray(pattern, dtype=np.int32)
    # Clamp indices to valid range
    valid_indices = np.clip(pattern_arr, 0, len(tensor_f32) - 1)
    return tensor_f32[valid_indices]


# --- Memory Management ---


def vtcm_alloc(size_bytes: int) -> VTCMBuffer:
    """Allocate a scratchpad buffer in VTCM (on-chip SRAM).

    VTCM provides single-cycle latency and very high bandwidth.
    Typically used as a staging area for DMA transfers.

    Args:
        size_bytes: Size in bytes to allocate.

    Returns:
        VTCMBuffer representing the allocation.
    """
    return VTCMBuffer(size_bytes=size_bytes)


def dma_async(dst: Any, src: Any) -> DMATransfer:
    """Initiate an asynchronous DMA transfer.

    Models hardware DMA engines that move data between system memory
    and on-chip VTCM without CPU involvement.

    Args:
        dst: Destination buffer (VTCMBuffer or numpy array).
        src: Source data (numpy array or VTCMBuffer).

    Returns:
        DMATransfer object that can be waited on.
    """
    # In mock mode, perform the copy immediately
    if isinstance(dst, VTCMBuffer) and isinstance(src, np.ndarray):
        flat = src.view(np.uint8).flatten()
        size = min(len(flat), dst.size_bytes)
        dst.data[:size] = flat[:size]
        transfer = DMATransfer(src=src, dst=dst, size_bytes=size)
    elif isinstance(dst, np.ndarray) and isinstance(src, VTCMBuffer):
        flat_dst = dst.view(np.uint8).flatten()
        size = min(len(flat_dst), src.size_bytes)
        flat_dst[:size] = src.data[:size]
        transfer = DMATransfer(src=src, dst=dst, size_bytes=size)
    else:
        # Generic case
        transfer = DMATransfer(src=src, dst=dst, size_bytes=0)

    transfer.completed = True
    return transfer


# --- Synchronization ---


def barrier() -> None:
    """Synchronize all HVX hardware threads.

    Analogous to CUDA's __syncthreads(). Ensures all threads in the
    workgroup have reached this point before proceeding.

    In mock mode, this is a no-op since execution is sequential.
    """
    pass


# --- Math Primitives ---


def tanh(x: np.ndarray | float) -> np.ndarray | float:
    """Hyperbolic tangent — hardware-accelerated in real mode.

    Args:
        x: Input value(s).

    Returns:
        tanh(x), element-wise.
    """
    return np.tanh(x)


def exp(x: np.ndarray | float) -> np.ndarray | float:
    """Exponential function — hardware-accelerated in real mode.

    Args:
        x: Input value(s).

    Returns:
        exp(x), element-wise.
    """
    return np.exp(x)


def sqrt(x: np.ndarray | float) -> np.ndarray | float:
    """Square root — hardware-accelerated in real mode.

    Args:
        x: Input value(s).

    Returns:
        sqrt(x), element-wise.
    """
    return np.sqrt(x)


def rsqrt(x: np.ndarray | float) -> np.ndarray | float:
    """Reciprocal square root — 1/sqrt(x).

    Args:
        x: Input value(s).

    Returns:
        1/sqrt(x), element-wise.
    """
    return 1.0 / np.sqrt(x)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid activation — 1/(1+exp(-x)).

    Args:
        x: Input value(s).

    Returns:
        sigmoid(x), element-wise.
    """
    return 1.0 / (1.0 + np.exp(-x))
