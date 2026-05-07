"""QUAD Tensor — device-aware tensor with unified memory abstraction."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from quad.runtime.device import Device


class Tensor:
    """Device-aware tensor for QUAD runtime.

    Wraps numpy arrays with device placement information.
    In real mode, manages DMA transfers between CPU/GPU/NPU memory.

    Usage:
        t = Tensor([1, 3, 224, 224], device=Device("npu"))
        t = Tensor.from_numpy(np_array, device=Device("npu"))
        np_data = t.to_numpy()
    """

    def __init__(
        self,
        shape: Sequence[int] | np.ndarray,
        device: Device | str = "cpu",
        dtype: str = "float32",
    ):
        if isinstance(device, str):
            device = Device(device)
        self._device = device

        if isinstance(shape, np.ndarray):
            self._data = shape
            self._shape = tuple(shape.shape)
            self._dtype = str(shape.dtype)
        else:
            self._shape = tuple(shape)
            self._dtype = dtype
            self._data = np.zeros(self._shape, dtype=dtype)

    @classmethod
    def from_numpy(cls, array: np.ndarray, device: Device | str = "cpu") -> Tensor:
        """Create tensor from numpy array."""
        t = cls(array, device=device)
        return t

    @classmethod
    def rand(cls, *shape: int, device: Device | str = "cpu", dtype: str = "float32") -> Tensor:
        """Create tensor with random values."""
        data = np.random.randn(*shape).astype(dtype)
        return cls(data, device=device)

    @classmethod
    def zeros(cls, *shape: int, device: Device | str = "cpu", dtype: str = "float32") -> Tensor:
        """Create tensor filled with zeros."""
        data = np.zeros(shape, dtype=dtype)
        return cls(data, device=device)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    @property
    def size(self) -> int:
        """Total number of elements."""
        result = 1
        for s in self._shape:
            result *= s
        return result

    def to(self, device: Device | str) -> Tensor:
        """Move tensor to a different device."""
        if isinstance(device, str):
            device = Device(device)
        return Tensor(self._data.copy(), device=device)

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (copies data to CPU if on device)."""
        return self._data.copy()

    def copy_from(self, data: np.ndarray) -> None:
        """Copy data into this tensor (in-place update)."""
        self._data = data.astype(self._dtype)
        self._shape = tuple(data.shape)

    @classmethod
    def from_batch(
        cls,
        tensors: list["Tensor"],
        device: Device | str = "cpu",
    ) -> "Tensor":
        """Stack a list of tensors into a single batched tensor.

        Equivalent to concatenating inputs before passing to SNPE.
        The batch dimension is prepended (or the first dim is expanded).

        Args:
            tensors: List of tensors with identical shapes
            device: Target device for the batched tensor

        Returns:
            Batched tensor with shape (len(tensors), *per_item_shape)

        Example:
            # Batch 4 image tensors [3, 224, 224] → [4, 3, 224, 224]
            batch = Tensor.from_batch([t1, t2, t3, t4], device="npu")
        """
        if not tensors:
            raise ValueError("Cannot create batch from empty list")
        arrays = [t.to_numpy() for t in tensors]
        batched = np.stack(arrays, axis=0)
        return cls(batched, device=device)

    def split_batch(self, batch_size: int) -> list["Tensor"]:
        """Split a batched tensor back into individual tensors.

        Inverse of from_batch() — used to separate batched inference output
        into per-item result tensors.

        Args:
            batch_size: Number of items in the batch

        Returns:
            List of batch_size tensors, each with shape[1:]

        Example:
            # Output [4, 1000] → [Tensor([1000]), Tensor([1000]), ...]
            results = output.split_batch(4)
        """
        data = self._data
        if data.ndim == 1 or data.shape[0] != batch_size:
            # Handle flat concatenation (no explicit batch dim)
            chunk = data.size // batch_size
            splits = [
                Tensor(np.array(data.flat[i * chunk:(i + 1) * chunk]), device=self._device)
                for i in range(batch_size)
            ]
        else:
            splits = [
                Tensor(data[i], device=self._device)
                for i in range(batch_size)
            ]
        return splits

    def __repr__(self) -> str:
        return f"Tensor(shape={self._shape}, dtype='{self._dtype}', device='{self._device.type}')"

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0
