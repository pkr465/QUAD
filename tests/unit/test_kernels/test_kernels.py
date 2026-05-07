"""Tests for QUAD Kernels & Streams (Phase E).

Covers:
- Kernel decorator creates KernelFunc
- KernelFunc can be called with tensors
- compile_kernel produces CompiledKernel
- register_op stores kernel
- Grid iteration
- Primitives (hvx_vload, hvx_vadd, etc.)
- Graph capture and replay
- Graph node count matches operations
"""

from __future__ import annotations

import numpy as np
import pytest

from quad.kernels import Graph, Grid, KernelFunc, compile_kernel, kernel, register_op
from quad.kernels.dsl import CompiledKernel, get_registered_ops, grid
from quad.kernels.graph import GraphNode, record_op
from quad.kernels.primitives import (
    DMATransfer,
    VTCMBuffer,
    barrier,
    dma_async,
    exp,
    hvx_shuffle,
    hvx_vadd,
    hvx_vload,
    hvx_vmpy,
    sqrt,
    tanh,
    vtcm_alloc,
)
from quad.runtime.device import Device
from quad.runtime.tensor import Tensor


# ---------------------------------------------------------------------------
# Kernel Decorator Tests
# ---------------------------------------------------------------------------


class TestKernelDecorator:
    """Test that the @kernel decorator creates KernelFunc correctly."""

    def test_kernel_creates_kernel_func(self):
        @kernel
        def my_add(x, y, output):
            for i in grid(x.shape):
                output[i] = x[i] + y[i]

        assert isinstance(my_add, KernelFunc)

    def test_kernel_preserves_name(self):
        @kernel
        def fused_relu(x, output):
            pass

        assert fused_relu.name == "fused_relu"

    def test_kernel_preserves_source(self):
        @kernel
        def simple_kernel(x):
            pass

        assert "simple_kernel" in simple_kernel.source

    def test_kernel_not_compiled_initially(self):
        @kernel
        def some_kernel(x):
            pass

        assert some_kernel.compiled is False
        assert some_kernel.target is None

    def test_kernel_requires_at_least_one_param(self):
        with pytest.raises(ValueError, match="must accept at least one"):

            @kernel
            def bad_kernel():
                pass

    def test_kernel_num_params(self):
        @kernel
        def binary_kernel(a, b):
            pass

        assert binary_kernel.num_params == 2


# ---------------------------------------------------------------------------
# KernelFunc Execution Tests
# ---------------------------------------------------------------------------


class TestKernelExecution:
    """Test that KernelFunc can be called with tensors."""

    def test_call_with_numpy(self):
        @kernel
        def double(x, output):
            output[:] = x * 2.0

        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = np.zeros(3, dtype=np.float32)
        double(x, output)
        np.testing.assert_array_almost_equal(output, [2.0, 4.0, 6.0])

    def test_call_with_tensor(self):
        @kernel
        def negate(x, output):
            output[:] = -x

        t_in = Tensor.from_numpy(np.array([1.0, -2.0, 3.0], dtype=np.float32))
        t_out = Tensor.zeros(3)
        negate(t_in, t_out)
        # The kernel operates on the underlying _data arrays
        np.testing.assert_array_almost_equal(t_out._data, [-1.0, 2.0, -3.0])

    def test_gelu_kernel(self):
        @kernel
        def fused_gelu(x, output):
            val = x
            output[:] = 0.5 * val * (1 + np.tanh(0.7978845 * (val + 0.044715 * val**3)))

        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        output = np.zeros(3, dtype=np.float32)
        fused_gelu(x, output)

        # GELU(0) = 0, GELU(1) ~ 0.8413, GELU(-1) ~ -0.1587
        assert abs(output[0]) < 1e-5
        assert 0.8 < output[1] < 0.9
        assert -0.2 < output[2] < -0.1


# ---------------------------------------------------------------------------
# compile_kernel Tests
# ---------------------------------------------------------------------------


class TestCompileKernel:
    """Test compile_kernel produces CompiledKernel."""

    def test_compile_from_kernel_func(self):
        @kernel
        def add_one(x, output):
            output[:] = x + 1

        compiled = compile_kernel(add_one)
        assert isinstance(compiled, CompiledKernel)
        assert compiled.compiled is True
        assert compiled.target == "hexagon_v73"

    def test_compile_with_custom_target(self):
        @kernel
        def scale(x, output):
            output[:] = x * 0.5

        compiled = compile_kernel(scale, target="hexagon_v75")
        assert compiled.target == "hexagon_v75"

    def test_compile_from_raw_function(self):
        def raw_func(x, output):
            output[:] = x + 100

        compiled = compile_kernel(raw_func, target="hexagon_v73")
        assert isinstance(compiled, CompiledKernel)
        assert compiled.name == "raw_func"

    def test_compiled_kernel_callable(self):
        @kernel
        def triple(x, output):
            output[:] = x * 3

        compiled = triple.compile()
        x = np.array([1.0, 2.0], dtype=np.float32)
        output = np.zeros(2, dtype=np.float32)
        compiled(x, output)
        np.testing.assert_array_almost_equal(output, [3.0, 6.0])


# ---------------------------------------------------------------------------
# register_op Tests
# ---------------------------------------------------------------------------


class TestRegisterOp:
    """Test register_op stores kernel in the registry."""

    def test_register_op(self):
        @kernel
        def custom_op(x, output):
            output[:] = x ** 2

        compiled = custom_op.compile()
        register_op("com.quad.square", compiled)

        registry = get_registered_ops()
        assert "com.quad.square" in registry
        assert registry["com.quad.square"] is compiled

    def test_register_as_op_method(self):
        @kernel
        def another_op(x, output):
            output[:] = x + 42

        compiled = another_op.compile()
        compiled.register_as_op("com.quad.add42")

        registry = get_registered_ops()
        assert "com.quad.add42" in registry


# ---------------------------------------------------------------------------
# Grid Tests
# ---------------------------------------------------------------------------


class TestGrid:
    """Test Grid iteration space helper."""

    def test_grid_1d(self):
        g = Grid((4,))
        indices = list(g)
        assert indices == [(0,), (1,), (2,), (3,)]

    def test_grid_2d(self):
        g = Grid((2, 3))
        indices = list(g)
        assert len(indices) == 6
        assert (0, 0) in indices
        assert (1, 2) in indices

    def test_grid_3d(self):
        g = Grid((2, 2, 2))
        indices = list(g)
        assert len(indices) == 8

    def test_grid_shape_property(self):
        g = Grid((3, 4, 5))
        assert g.shape == (3, 4, 5)

    def test_grid_ndim(self):
        g = Grid((3, 4))
        assert g.ndim == 2

    def test_grid_total_elements(self):
        g = Grid((2, 3, 4))
        assert g.total_elements == 24

    def test_grid_function(self):
        g = grid((5,))
        assert isinstance(g, Grid)
        assert g.total_elements == 5


# ---------------------------------------------------------------------------
# Primitives Tests
# ---------------------------------------------------------------------------


class TestPrimitives:
    """Test Hexagon HVX primitive mock implementations."""

    def test_hvx_vload(self):
        data = np.arange(64, dtype=np.float32)  # 64 * 4 = 256 bytes
        vec = hvx_vload(data, offset=0)
        # Should load 128 bytes = 32 float32 values
        assert vec.dtype == np.float32
        assert len(vec) == 32

    def test_hvx_vload_offset(self):
        data = np.arange(64, dtype=np.float32)
        vec = hvx_vload(data, offset=128)  # Skip first 32 floats
        assert vec.dtype == np.float32
        assert len(vec) == 32

    def test_hvx_vadd(self):
        a = np.ones(32, dtype=np.float32) * 3.0
        b = np.ones(32, dtype=np.float32) * 7.0
        result = hvx_vadd(a, b)
        np.testing.assert_array_almost_equal(result, np.ones(32) * 10.0)

    def test_hvx_vmpy(self):
        a = np.ones(32, dtype=np.float32) * 3.0
        b = np.ones(32, dtype=np.float32) * 4.0
        result = hvx_vmpy(a, b)
        np.testing.assert_array_almost_equal(result, np.ones(32) * 12.0)

    def test_hvx_shuffle(self):
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        pattern = [3, 2, 1, 0]  # Reverse
        result = hvx_shuffle(data, pattern)
        np.testing.assert_array_almost_equal(result, [40.0, 30.0, 20.0, 10.0])

    def test_vtcm_alloc(self):
        buf = vtcm_alloc(1024)
        assert isinstance(buf, VTCMBuffer)
        assert buf.size_bytes == 1024
        assert buf.allocated is True

    def test_vtcm_read_write(self):
        buf = vtcm_alloc(256)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf.write(0, data)
        readback = buf.read(0, 16)  # 4 floats * 4 bytes
        np.testing.assert_array_equal(readback, data.view(np.uint8))

    def test_dma_async(self):
        buf = vtcm_alloc(128)
        src = np.ones(32, dtype=np.float32)
        transfer = dma_async(buf, src)
        assert isinstance(transfer, DMATransfer)
        assert transfer.is_done is True

    def test_barrier_is_noop(self):
        # Should not raise
        barrier()

    def test_tanh_scalar(self):
        assert abs(tanh(0.0)) < 1e-7

    def test_tanh_array(self):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = tanh(x)
        expected = np.tanh(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_exp_array(self):
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = exp(x)
        np.testing.assert_array_almost_equal(result, np.exp(x))

    def test_sqrt_array(self):
        x = np.array([1.0, 4.0, 9.0], dtype=np.float32)
        result = sqrt(x)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Graph Tests
# ---------------------------------------------------------------------------


class TestGraph:
    """Test Graph capture and replay mechanism."""

    def test_graph_capture_basic(self):
        results = []

        def op_a():
            results.append("a")
            return 1

        def op_b():
            results.append("b")
            return 2

        with Graph.capture() as g:
            g.add_node("op_a", op_a)
            g.add_node("op_b", op_b)

        assert g.is_captured is True
        assert results == ["a", "b"]

    def test_graph_node_count(self):
        def noop():
            return None

        with Graph.capture() as g:
            g.add_node("step1", noop)
            g.add_node("step2", noop)
            g.add_node("step3", noop)

        assert g.num_nodes == 3

    def test_graph_replay(self):
        counter = {"value": 0}

        def increment():
            counter["value"] += 1
            return counter["value"]

        with Graph.capture() as g:
            g.add_node("inc1", increment)
            g.add_node("inc2", increment)

        # After capture, counter is at 2
        assert counter["value"] == 2

        # Replay should increment twice more
        g.replay()
        assert counter["value"] == 4

        # Replay again
        g.replay()
        assert counter["value"] == 6

    def test_graph_replay_before_capture_raises(self):
        g = Graph()
        with pytest.raises(RuntimeError, match="has not been captured"):
            g.replay()

    def test_graph_with_arguments(self):
        results = []

        def add(a, b):
            result = a + b
            results.append(result)
            return result

        with Graph.capture() as g:
            g.add_node("add_1_2", add, args=(1, 2))
            g.add_node("add_3_4", add, args=(3, 4))

        assert results == [3, 7]

        # Replay
        g.replay()
        assert results == [3, 7, 3, 7]

    def test_graph_repr(self):
        with Graph.capture() as g:
            pass

        assert "captured" in repr(g)
        assert "nodes=0" in repr(g)

    def test_graph_reset(self):
        def noop():
            pass

        with Graph.capture() as g:
            g.add_node("op", noop)

        assert g.num_nodes == 1
        g.reset()
        assert g.num_nodes == 0
        assert g.is_captured is False

    def test_record_op_without_capture(self):
        """record_op executes directly when no capture is active."""
        result = record_op("direct_call", lambda: 42)
        assert result == 42

    def test_record_op_during_capture(self):
        """record_op records into active graph during capture."""
        with Graph.capture() as g:
            result = record_op("test_op", lambda: 99)

        assert result == 99
        assert g.num_nodes == 1
        assert g.nodes[0].name == "test_op"

    def test_graph_with_tensor_operations(self):
        """Integration test: graph captures tensor operations."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        output = np.zeros(3, dtype=np.float32)

        def scale_op():
            output[:] = data * 2.0
            return output

        with Graph.capture() as g:
            g.add_node("scale", scale_op)

        np.testing.assert_array_almost_equal(output, [2.0, 4.0, 6.0])

        # Modify input data and replay
        data[:] = [10.0, 20.0, 30.0]
        g.replay()
        np.testing.assert_array_almost_equal(output, [20.0, 40.0, 60.0])

    def test_is_capturing_context(self):
        """Test that is_capturing returns correct state."""
        assert Graph.is_capturing() is False

        with Graph.capture() as g:
            assert Graph.is_capturing() is True

        assert Graph.is_capturing() is False
