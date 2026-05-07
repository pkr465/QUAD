"""QUAD Kernel DSL — Python decorator-based kernel programming for Hexagon NPU.

Provides a Python DSL for writing custom kernels that compile to Hexagon HVX
instructions. In mock mode, kernels execute as numpy operations.

Usage:
    @quad.kernels.kernel
    def fused_gelu(x, output):
        for i in grid(x.shape):
            val = x[i]
            output[i] = 0.5 * val * (1 + tanh(0.7978845 * (val + 0.044715 * val**3)))

    # Execute in mock mode
    fused_gelu(input_tensor, output_tensor)

    # Compile for target hardware
    compiled = fused_gelu.compile(target="hexagon_v73")
    compiled(input_tensor, output_tensor)
"""

from __future__ import annotations

import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Global registry for custom ops
_OP_REGISTRY: dict[str, CompiledKernel] = {}


class Grid:
    """Iteration space helper for kernel loops.

    Represents the N-dimensional iteration domain for a kernel.
    In mock mode, provides iteration over all indices in the shape.

    Usage:
        for i in Grid((batch, height, width)):
            ...  # iterates over all (b, h, w) indices
    """

    def __init__(self, shape: tuple[int, ...] | list[int]):
        if isinstance(shape, (list, tuple)):
            self._shape = tuple(shape)
        else:
            self._shape = (shape,)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def total_elements(self) -> int:
        """Total number of grid points."""
        result = 1
        for s in self._shape:
            result *= s
        return result

    def __iter__(self):
        """Iterate over all indices in the grid."""
        if len(self._shape) == 1:
            for i in range(self._shape[0]):
                yield (i,)
        else:
            yield from self._nd_iter(self._shape)

    def _nd_iter(self, shape: tuple[int, ...]):
        """Generate all N-dimensional indices."""
        if len(shape) == 1:
            for i in range(shape[0]):
                yield (i,)
        else:
            for i in range(shape[0]):
                for rest in self._nd_iter(shape[1:]):
                    yield (i,) + rest

    def __repr__(self) -> str:
        return f"Grid(shape={self._shape})"


def grid(shape) -> Grid:
    """Create a Grid iteration space (convenience function for use in kernels)."""
    return Grid(shape)


class KernelFunc:
    """Wrapper for a Python function marked as a QUAD kernel.

    Stores the function source, validates the signature, and provides
    mock execution via numpy and compilation to target hardware.

    Attributes:
        name: Kernel function name.
        source: Original Python source code of the kernel body.
        compiled: Whether this kernel has been compiled.
        target: Compilation target (None if not compiled).
    """

    def __init__(self, func: Callable):
        self._func = func
        self.name: str = func.__name__
        self.source: str = textwrap.dedent(inspect.getsource(func))
        self.compiled: bool = False
        self.target: str | None = None

        # Validate signature — must accept at least one parameter (tensor args)
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            raise ValueError(
                f"Kernel '{self.name}' must accept at least one tensor argument"
            )
        self._params = list(sig.parameters.keys())

    @property
    def num_params(self) -> int:
        """Number of parameters the kernel accepts."""
        return len(self._params)

    def __call__(self, *tensors) -> Any:
        """Execute the kernel in mock mode (numpy fallback).

        Args:
            *tensors: Input/output tensors. Can be Tensor objects or numpy arrays.

        Returns:
            Result of the kernel function execution.
        """
        # Convert Tensor objects to numpy for mock execution
        numpy_args = []
        for t in tensors:
            if hasattr(t, "to_numpy"):
                numpy_args.append(t._data)
            elif isinstance(t, np.ndarray):
                numpy_args.append(t)
            else:
                numpy_args.append(t)

        # Execute the original function with numpy arrays
        return self._func(*numpy_args)

    def compile(self, target: str = "hexagon_v73") -> CompiledKernel:
        """Compile this kernel for the specified target hardware.

        Args:
            target: Target architecture (e.g., "hexagon_v73", "hexagon_v75").

        Returns:
            A CompiledKernel ready for execution on the target.
        """
        compiled = CompiledKernel(
            name=self.name,
            source=self.source,
            target=target,
            func=self._func,
        )
        self.compiled = True
        self.target = target
        return compiled


@dataclass
class CompiledKernel:
    """A kernel compiled for a specific hardware target.

    In mock mode, still executes via numpy. In real mode, this would
    contain the compiled Hexagon binary blob.

    Attributes:
        name: Kernel name.
        source: Original source code.
        target: Compilation target architecture.
        compiled: Always True for CompiledKernel.
    """

    name: str
    source: str
    target: str
    func: Callable
    compiled: bool = field(default=True, init=False)
    _registered_op: str | None = field(default=None, init=False)

    def __call__(self, *tensors) -> Any:
        """Execute the compiled kernel.

        In mock mode, falls back to numpy execution.
        In real mode, dispatches to compiled Hexagon binary.
        """
        numpy_args = []
        for t in tensors:
            if hasattr(t, "to_numpy"):
                numpy_args.append(t._data)
            elif isinstance(t, np.ndarray):
                numpy_args.append(t)
            else:
                numpy_args.append(t)

        return self.func(*numpy_args)

    def register_as_op(self, name: str) -> None:
        """Register this compiled kernel as a custom ONNX operator.

        Args:
            name: Operator name for the ONNX registry (e.g., "com.quad.fused_gelu").
        """
        self._registered_op = name
        _OP_REGISTRY[name] = self


def kernel(func: Callable) -> KernelFunc:
    """Decorator that marks a Python function as a QUAD kernel.

    The decorated function can be called directly (mock execution via numpy)
    or compiled for a target hardware architecture.

    Usage:
        @kernel
        def my_kernel(x, output):
            for i in grid(x.shape):
                output[i] = x[i] * 2.0

        # Mock execution
        my_kernel(input_tensor, output_tensor)

        # Compile and run
        compiled = my_kernel.compile(target="hexagon_v73")
        compiled(input_tensor, output_tensor)
    """
    return KernelFunc(func)


def compile_kernel(func: Callable | KernelFunc, target: str = "hexagon_v73") -> CompiledKernel:
    """Compile a kernel function for the specified target.

    Args:
        func: A function decorated with @kernel, or a raw function.
        target: Target architecture (default: "hexagon_v73").

    Returns:
        CompiledKernel ready for execution.
    """
    if isinstance(func, KernelFunc):
        return func.compile(target=target)
    # If passed a raw function, wrap it first
    kernel_func = KernelFunc(func)
    return kernel_func.compile(target=target)


def register_op(name: str, kernel: CompiledKernel) -> None:
    """Register a compiled kernel as a custom operator.

    Args:
        name: Operator name (e.g., "com.quad.fused_gelu").
        kernel: The compiled kernel to register.
    """
    kernel.register_as_op(name)


def get_registered_ops() -> dict[str, CompiledKernel]:
    """Return the global registry of custom ops."""
    return dict(_OP_REGISTRY)
