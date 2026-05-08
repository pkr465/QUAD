"""Tests for the AIMET adapter (T1.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quad.adapters.aimet_adapter import (
    AIMETAdapter,
    AIMETUnavailableError,
    QuantizationConfig,
    QuantizationResult,
    _iterate_calibration,
    aimet_onnx_available,
    aimet_torch_available,
    select_backend,
)
from quad.exceptions import QuantizationError


# ─── QuantizationConfig validation ───────────────────────────────────────────


class TestQuantizationConfig:
    def test_default_int8_per_channel(self) -> None:
        cfg = QuantizationConfig()
        assert cfg.bitwidth == 8
        assert cfg.scheme == "symmetric_per_channel"
        assert cfg.activation_bitwidth == 8

    def test_explicit_int4(self) -> None:
        cfg = QuantizationConfig(bitwidth=4)
        assert cfg.bitwidth == 4
        assert cfg.activation_bitwidth == 4

    def test_int4_per_tensor_rejected(self) -> None:
        # INT4 per-tensor doesn't work in practice — should raise
        with pytest.raises(ValueError, match="per-channel"):
            QuantizationConfig(bitwidth=4, scheme="symmetric_per_tensor")

    def test_invalid_bitwidth(self) -> None:
        with pytest.raises(ValueError, match="bitwidth"):
            QuantizationConfig(bitwidth=12)  # type: ignore[arg-type]

    def test_to_dict(self) -> None:
        cfg = QuantizationConfig(bitwidth=8, calibration_samples=200)
        d = cfg.to_dict()
        assert d["bitwidth"] == 8
        assert d["calibration_samples"] == 200


# ─── Backend detection ───────────────────────────────────────────────────────


class TestBackendDetection:
    def test_mock_always_available(self) -> None:
        assert select_backend("mock") == "mock"

    def test_aimet_torch_unavailable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the package detection to return False
        import quad.adapters.aimet_adapter as mod
        monkeypatch.setattr(mod, "aimet_torch_available", lambda: False)
        with pytest.raises(AIMETUnavailableError):
            select_backend("aimet_torch")

    def test_auto_falls_back_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import quad.adapters.aimet_adapter as mod
        monkeypatch.setattr(mod, "aimet_torch_available", lambda: False)
        monkeypatch.setattr(mod, "aimet_onnx_available", lambda: False)
        # Clear env var that would force a backend
        monkeypatch.delenv("QUAD_AIMET_BACKEND", raising=False)
        assert select_backend("auto") == "mock"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUAD_AIMET_BACKEND", "mock")
        # Even if real backends would be available, env override picks mock
        assert select_backend("auto") == "mock"


# ─── Calibration iteration ──────────────────────────────────────────────────


class TestCalibrationIteration:
    def test_list_of_arrays(self) -> None:
        cal = [np.zeros((1, 3, 4, 4)) for _ in range(5)]
        batches = list(_iterate_calibration(cal, num_samples=3))
        assert len(batches) == 3
        assert all("input" in b for b in batches)

    def test_directory_of_npy_files(self, tmp_path: Path) -> None:
        for i in range(3):
            np.save(tmp_path / f"sample_{i}.npy", np.ones((1, 3, 4, 4)))
        batches = list(_iterate_calibration(tmp_path, num_samples=10))
        assert len(batches) == 3

    def test_iterable_of_dicts(self) -> None:
        cal = [{"a": np.ones(3), "b": np.zeros(3)} for _ in range(4)]
        batches = list(_iterate_calibration(iter(cal), num_samples=2))
        assert len(batches) == 2
        assert "a" in batches[0]
        assert "b" in batches[0]

    def test_callable_source(self) -> None:
        def gen():
            for i in range(5):
                yield np.full((1, 3, 4, 4), float(i))

        batches = list(_iterate_calibration(gen, num_samples=3))
        assert len(batches) == 3
        # Inputs should be sequenced 0, 1, 2
        assert batches[0]["input"][0, 0, 0, 0] == 0.0
        assert batches[2]["input"][0, 0, 0, 0] == 2.0

    def test_invalid_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(QuantizationError):
            list(_iterate_calibration(tmp_path / "nonexistent"))


# ─── AIMETAdapter (mock backend) ─────────────────────────────────────────────


class TestAIMETAdapterMock:
    def test_init_falls_back_to_mock_when_aimet_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quad.adapters.aimet_adapter as mod
        monkeypatch.setattr(mod, "aimet_torch_available", lambda: False)
        monkeypatch.setattr(mod, "aimet_onnx_available", lambda: False)
        monkeypatch.delenv("QUAD_AIMET_BACKEND", raising=False)
        a = AIMETAdapter()
        assert a.backend == "mock"

    def test_strict_raises_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quad.adapters.aimet_adapter as mod
        monkeypatch.setattr(mod, "aimet_torch_available", lambda: False)
        with pytest.raises(AIMETUnavailableError):
            AIMETAdapter(backend="aimet_torch", strict=True)

    def test_quantize_int8(self, tmp_path: Path) -> None:
        model = tmp_path / "m.dlc"
        model.write_bytes(b"original")
        a = AIMETAdapter(backend="mock")
        result = a.quantize(model)
        assert isinstance(result, QuantizationResult)
        assert result.backend == "mock"
        assert result.bitwidth == 8
        assert result.weight_size_compression == 4.0  # int8 vs float32
        assert Path(result.output_path).exists()

    def test_quantize_int4_per_channel(self, tmp_path: Path) -> None:
        model = tmp_path / "m.dlc"
        model.write_bytes(b"x")
        a = AIMETAdapter(backend="mock")
        cfg = QuantizationConfig(bitwidth=4, scheme="symmetric_per_channel")
        result = a.quantize(model, config=cfg)
        assert result.bitwidth == 4
        assert result.weight_size_compression == 8.0  # int4 vs float32
        # INT4 has higher accuracy drop estimate
        assert result.accuracy_drop_estimate_pct > 1.0

    def test_quantize_with_calibration(self, tmp_path: Path) -> None:
        model = tmp_path / "m.dlc"
        model.write_bytes(b"x")
        cal = [np.random.randn(1, 3, 4, 4).astype(np.float32) for _ in range(20)]
        a = AIMETAdapter(backend="mock")
        cfg = QuantizationConfig(calibration_samples=10)
        result = a.quantize(model, config=cfg, calibration=cal)
        assert result.calibration_samples_used == 10  # capped at config

    def test_quantize_without_existing_source(self, tmp_path: Path) -> None:
        # Mock backend handles a non-existent source gracefully
        a = AIMETAdapter(backend="mock")
        result = a.quantize(tmp_path / "missing.dlc", output_path=tmp_path / "out.dlc")
        assert Path(result.output_path).exists()

    def test_doctor_reports_mock(self) -> None:
        a = AIMETAdapter(backend="mock")
        d = a.doctor()
        assert d["backend"] == "mock"
        assert "aimet_torch_installed" in d
        assert "supported_bitwidths" in d
        assert 4 in d["supported_bitwidths"]
        assert 8 in d["supported_bitwidths"]


class TestAIMETAdapterRealBackends:
    """Real backends raise NotImplementedError for now — verify the error path."""

    def test_aimet_torch_real_not_yet_implemented(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import quad.adapters.aimet_adapter as mod
        monkeypatch.setattr(mod, "aimet_torch_available", lambda: True)
        a = AIMETAdapter(backend="aimet_torch")
        assert a.backend == "aimet_torch"
        model = tmp_path / "m.dlc"
        model.write_bytes(b"x")
        with pytest.raises(NotImplementedError):
            a.quantize(model)

    def test_aimet_onnx_real_not_yet_implemented(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import quad.adapters.aimet_adapter as mod
        monkeypatch.setattr(mod, "aimet_torch_available", lambda: False)
        monkeypatch.setattr(mod, "aimet_onnx_available", lambda: True)
        a = AIMETAdapter(backend="aimet_onnx")
        assert a.backend == "aimet_onnx"
        model = tmp_path / "m.dlc"
        model.write_bytes(b"x")
        with pytest.raises(NotImplementedError):
            a.quantize(model)
