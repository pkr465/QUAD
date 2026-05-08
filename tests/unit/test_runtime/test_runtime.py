"""Tests for QUAD Runtime API."""

from __future__ import annotations

import numpy as np
import pytest

from quad.runtime import (
    Device,
    Event,
    MemoryPool,
    Model,
    PowerMode,
    PowerMonitor,
    Stream,
    Tensor,
    estimate_battery_life,
    is_available,
    list_devices,
    load,
)


class TestDevice:
    def test_create_npu(self) -> None:
        d = Device("npu")
        assert d.type == "npu"
        assert d.is_npu
        # tops may vary if the host probe found real hardware; fallback is 45.0
        assert d.tops > 0
        # Name is "Hexagon NPU" in fallback mode, or the real probed
        # name (e.g. "Snapdragon(R) X Elite ... Hexagon(TM) NPU")
        assert "Hexagon" in d.name or "NPU" in d.name

    def test_create_gpu(self) -> None:
        d = Device("gpu")
        assert d.is_gpu
        # tflops is 4.6 in fallback; 0.0 if real probe found GPU but
        # couldn't read tflops (Win32 probe doesn't expose it)
        assert d.tflops >= 0

    def test_create_cpu(self) -> None:
        d = Device("cpu")
        assert d.is_cpu
        # cores is 12 in fallback (Oryon); real probe returns the
        # actual core count from /proc/cpuinfo or Win32_Processor
        assert d.cores >= 1

    def test_auto_selects_npu(self) -> None:
        d = Device("auto")
        assert d.type == "npu"  # NPU has highest priority

    def test_indexed_device(self) -> None:
        d = Device("npu:0")
        assert d.type == "npu"
        assert d.index == 0

    def test_list_devices(self) -> None:
        devices = list_devices()
        assert len(devices) == 3
        types = [d.type for d in devices]
        assert "npu" in types
        assert "gpu" in types
        assert "cpu" in types

    def test_is_available(self) -> None:
        assert is_available("npu")
        assert is_available("gpu")
        assert is_available("cpu")
        assert not is_available("tpu")

    def test_device_equality(self) -> None:
        d1 = Device("npu")
        d2 = Device("npu")
        assert d1 == d2

    def test_device_repr(self) -> None:
        d = Device("npu")
        # When running on real Snapdragon hardware the device name reflects
        # the probed value (e.g. "Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Hexagon(TM) NPU");
        # in fallback mode (CI Linux) it stays "Hexagon NPU". Either way
        # the repr should mention "Hexagon" and the tops value.
        assert "Hexagon" in repr(d) or "NPU" in repr(d)
        assert "45" in repr(d)


class TestTensor:
    def test_create_from_shape(self) -> None:
        t = Tensor([1, 3, 224, 224], device="npu")
        assert t.shape == (1, 3, 224, 224)
        assert t.device.type == "npu"
        assert t.dtype == "float32"

    def test_create_from_numpy(self) -> None:
        arr = np.ones((2, 3), dtype="float32")
        t = Tensor.from_numpy(arr, device="gpu")
        assert t.shape == (2, 3)
        assert t.device.type == "gpu"

    def test_rand(self) -> None:
        t = Tensor.rand(1, 3, 224, 224, device="npu")
        assert t.shape == (1, 3, 224, 224)
        assert t.size == 1 * 3 * 224 * 224

    def test_zeros(self) -> None:
        t = Tensor.zeros(4, 4, device="cpu")
        np.testing.assert_array_equal(t.to_numpy(), np.zeros((4, 4)))

    def test_to_device(self) -> None:
        t = Tensor.rand(1, 10, device="cpu")
        t_npu = t.to("npu")
        assert t_npu.device.type == "npu"
        assert t.device.type == "cpu"  # Original unchanged

    def test_to_numpy(self) -> None:
        arr = np.array([[1, 2], [3, 4]], dtype="float32")
        t = Tensor.from_numpy(arr)
        result = t.to_numpy()
        np.testing.assert_array_equal(result, arr)

    def test_nbytes(self) -> None:
        t = Tensor([1, 3, 224, 224], dtype="float32")
        assert t.nbytes == 1 * 3 * 224 * 224 * 4

    def test_repr(self) -> None:
        t = Tensor([1, 3, 224, 224], device="npu")
        assert "npu" in repr(t)
        assert "(1, 3, 224, 224)" in repr(t)


class TestModel:
    def test_load_onnx(self) -> None:
        model = load("resnet50.onnx", device="npu")
        assert model.is_loaded
        assert model.device.type == "npu"
        assert model.format == "onnx"

    def test_load_qbin(self) -> None:
        model = load("model.qbin", device="auto")
        assert model.format == "qbin"

    def test_inference(self) -> None:
        model = load("model.onnx", device="npu")
        input_t = Tensor.rand(1, 3, 224, 224, device="npu")
        output = model(input_t)
        assert isinstance(output, Tensor)
        assert output.shape == (1, 1000)
        assert output.device.type == "npu"

    def test_inference_with_numpy(self) -> None:
        model = load("model.onnx", device="npu")
        input_np = np.random.randn(1, 3, 224, 224).astype("float32")
        output = model(input_np)
        assert isinstance(output, Tensor)

    def test_power_budget(self) -> None:
        model = load("model.onnx", device="npu", power_budget_mw=3000)
        assert model.power_budget_mw == 3000

    def test_set_power_mode(self) -> None:
        model = load("model.onnx", device="npu")
        model.set_power_mode("efficiency")
        assert model.power_budget_mw == 3000.0

    def test_infer_async(self) -> None:
        model = load("model.onnx", device="npu")
        input_t = Tensor.rand(1, 3, 224, 224, device="npu")
        future = model.infer_async(input_t)
        assert future.done
        result = future.result()
        assert isinstance(result, Tensor)

    def test_unload(self) -> None:
        model = load("model.onnx", device="npu")
        model.unload()
        assert not model.is_loaded

    def test_repr(self) -> None:
        model = load("resnet50.onnx", device="npu")
        assert "resnet50.onnx" in repr(model)
        assert "npu" in repr(model)


class TestStream:
    def test_create_stream(self) -> None:
        s = Stream()
        assert s.id > 0

    def test_synchronize(self) -> None:
        s = Stream()
        s.synchronize()  # Should not raise

    def test_context_manager(self) -> None:
        with Stream() as s:
            assert s.id > 0
        # No error = synchronized on exit

    def test_record_event(self) -> None:
        s = Stream()
        e = Event()
        s.record(e)
        assert e.is_recorded

    def test_wait_event(self) -> None:
        s1 = Stream()
        s2 = Stream()
        e = Event()
        s1.record(e)
        s2.wait(e)  # Should not block in mock mode


class TestEvent:
    def test_create_unrecorded(self) -> None:
        e = Event()
        assert not e.is_recorded

    def test_record(self) -> None:
        e = Event()
        e.record()
        assert e.is_recorded
        assert e.elapsed_ms >= 0


class TestMemoryPool:
    def test_create_pool(self) -> None:
        pool = MemoryPool(device="npu", size_mb=64)
        assert pool.size_mb == 64
        assert pool.free_mb == 64
        assert pool.num_active == 0

    def test_allocate(self) -> None:
        pool = MemoryPool(device="npu", size_mb=64)
        t = pool.allocate([1, 3, 224, 224])
        assert t.device.type == "npu"
        assert pool.num_active == 1
        assert pool.used_mb > 0

    def test_release(self) -> None:
        pool = MemoryPool(device="npu", size_mb=64)
        t = pool.allocate([1, 3, 224, 224])
        pool.release(t)
        assert pool.num_active == 0

    def test_pool_exhaustion(self) -> None:
        pool = MemoryPool(device="npu", size_mb=1)  # 1MB pool
        with pytest.raises(MemoryError):
            pool.allocate([256, 256, 256])  # Way too large

    def test_destroy(self) -> None:
        pool = MemoryPool(device="npu", size_mb=64)
        pool.allocate([1, 3, 224, 224])
        pool.destroy()
        assert pool.num_active == 0
        assert pool.used_mb == 0


class TestPowerMonitor:
    def test_context_manager(self) -> None:
        with PowerMonitor(device_type="npu") as pm:
            pass  # Simulate work
        report = pm.report
        assert report.avg_power_mw > 0
        assert report.energy_mj >= 0
        assert "npu" in report.breakdown

    def test_breakdown_has_all_units(self) -> None:
        with PowerMonitor(device_type="npu") as pm:
            pass
        breakdown = pm.report.breakdown
        assert "npu" in breakdown
        assert "gpu" in breakdown
        assert "cpu" in breakdown


class TestBatteryEstimate:
    def test_basic_estimate(self) -> None:
        est = estimate_battery_life(
            power_mw=2500,
            duty_cycle=0.3,
            battery_mah=5000,
            voltage=3.7,
        )
        assert est.hours > 0
        assert est.inference_count > 0
        assert est.energy_per_inference_mj > 0
        assert est.duty_cycle == 0.3

    def test_higher_power_shorter_life(self) -> None:
        low = estimate_battery_life(power_mw=1000, duty_cycle=0.5)
        high = estimate_battery_life(power_mw=5000, duty_cycle=0.5)
        assert low.hours > high.hours

    def test_higher_duty_cycle_shorter_life(self) -> None:
        low_duty = estimate_battery_life(power_mw=2500, duty_cycle=0.1)
        high_duty = estimate_battery_life(power_mw=2500, duty_cycle=0.9)
        assert low_duty.hours > high_duty.hours
