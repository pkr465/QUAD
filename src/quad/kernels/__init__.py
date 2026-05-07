"""QUAD Kernels & Streams — custom NPU programming via Python DSL and captured execution."""

from quad.kernels.dsl import (
    Grid,
    KernelFunc,
    compile_kernel,
    kernel,
    register_op,
)
from quad.kernels.graph import Graph

__all__ = [
    "kernel",
    "compile_kernel",
    "register_op",
    "Grid",
    "KernelFunc",
    "Graph",
]
