"""QUAD Memory — memory pools and device memory management."""

from __future__ import annotations

from typing import Sequence

from quad.runtime.device import Device
from quad.runtime.tensor import Tensor


class MemoryPool:
    """Pre-allocated memory pool for inference serving.

    Eliminates allocation overhead by reusing pre-allocated buffers.
    Critical for high-throughput serving (like CUDA memory pools).

    Usage:
        pool = MemoryPool(device=Device("npu"), size_mb=64)
        tensor = pool.allocate([1, 3, 224, 224])
        # ... use tensor ...
        pool.release(tensor)
        pool.destroy()
    """

    def __init__(self, device: Device | str = "npu", size_mb: int = 64):
        if isinstance(device, str):
            device = Device(device)
        self._device = device
        self._size_mb = size_mb
        self._allocated_mb: float = 0.0
        self._num_allocations: int = 0
        self._active_tensors: list[Tensor] = []

    @property
    def device(self) -> Device:
        return self._device

    @property
    def size_mb(self) -> int:
        return self._size_mb

    @property
    def used_mb(self) -> float:
        return self._allocated_mb

    @property
    def free_mb(self) -> float:
        return self._size_mb - self._allocated_mb

    @property
    def num_active(self) -> int:
        return len(self._active_tensors)

    def allocate(self, shape: Sequence[int], dtype: str = "float32") -> Tensor:
        """Allocate a tensor from the pool.

        Raises:
            MemoryError: If pool is exhausted.
        """
        tensor = Tensor(shape, device=self._device, dtype=dtype)
        size_mb = tensor.nbytes / (1024 * 1024)

        if self._allocated_mb + size_mb > self._size_mb:
            raise MemoryError(
                f"Pool exhausted: need {size_mb:.1f}MB, "
                f"free {self.free_mb:.1f}MB of {self._size_mb}MB"
            )

        self._allocated_mb += size_mb
        self._num_allocations += 1
        self._active_tensors.append(tensor)
        return tensor

    def release(self, tensor: Tensor) -> None:
        """Return a tensor to the pool."""
        if tensor in self._active_tensors:
            self._active_tensors.remove(tensor)
            self._allocated_mb -= tensor.nbytes / (1024 * 1024)

    def destroy(self) -> None:
        """Destroy the pool and free all memory."""
        self._active_tensors.clear()
        self._allocated_mb = 0.0

    def __repr__(self) -> str:
        return (
            f"MemoryPool(device='{self._device.type}', "
            f"used={self._allocated_mb:.1f}/{self._size_mb}MB, "
            f"active={self.num_active})"
        )
