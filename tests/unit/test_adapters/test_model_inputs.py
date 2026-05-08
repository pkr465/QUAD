"""Tests for model input introspection and dummy-input generation (T2.8)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from quad.adapters.model_inputs import (
    DEFAULT_FALLBACK_INPUT,
    ModelIO,
    TensorSpec,
    _parse_dlc_info_output,
    create_input_list_for_model,
    generate_random_input,
    introspect_model,
    write_input_list,
)


# ─── TensorSpec / ModelIO basics ─────────────────────────────────────────────


class TestTensorSpec:
    def test_numel(self) -> None:
        s = TensorSpec(name="x", shape=(1, 3, 224, 224), dtype="float32")
        assert s.numel == 1 * 3 * 224 * 224

    def test_num_bytes_float32(self) -> None:
        s = TensorSpec(name="x", shape=(1, 3, 4, 4), dtype="float32")
        assert s.num_bytes == 48 * 4

    def test_num_bytes_int8(self) -> None:
        s = TensorSpec(name="x", shape=(1, 3, 4, 4), dtype="int8")
        assert s.num_bytes == 48

    def test_to_dict_serialisable(self) -> None:
        s = TensorSpec(name="x", shape=(1, 3), dtype="float16")
        d = s.to_dict()
        assert d == {"name": "x", "shape": [1, 3], "dtype": "float16"}

    def test_zero_dim_handled(self) -> None:
        # Symbolic dims sometimes come through as 0; numel should still be ≥1
        s = TensorSpec(name="x", shape=(0, 3, 224, 224), dtype="float32")
        assert s.numel == 1 * 3 * 224 * 224


class TestModelIO:
    def test_is_empty_when_no_inputs(self) -> None:
        assert ModelIO().is_empty is True
        assert ModelIO(inputs=[DEFAULT_FALLBACK_INPUT]).is_empty is False


# ─── Random input generation ────────────────────────────────────────────────


class TestRandomInput:
    def test_float32_shape_and_dtype(self) -> None:
        s = TensorSpec(name="x", shape=(1, 3, 4, 4), dtype="float32")
        arr = generate_random_input(s, seed=42)
        assert arr.shape == (1, 3, 4, 4)
        assert arr.dtype == np.float32

    def test_int8_shape_and_dtype(self) -> None:
        s = TensorSpec(name="x", shape=(2, 8), dtype="int8")
        arr = generate_random_input(s, seed=42)
        assert arr.shape == (2, 8)
        assert arr.dtype == np.int8

    def test_seed_reproducible(self) -> None:
        s = TensorSpec(name="x", shape=(2, 2), dtype="float32")
        a = generate_random_input(s, seed=99)
        b = generate_random_input(s, seed=99)
        assert np.array_equal(a, b)

    def test_uint8_in_range(self) -> None:
        s = TensorSpec(name="x", shape=(1000,), dtype="uint8")
        arr = generate_random_input(s, seed=1)
        assert arr.min() >= 0
        assert arr.max() <= 255


# ─── ONNX introspection ──────────────────────────────────────────────────────


class TestONNXIntrospection:
    """Uses the bundled mobilenetv2-12.onnx if present (downloaded in
    Phase A example app); otherwise skips."""

    @pytest.fixture
    def mobilenet_path(self) -> Path | None:
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / "examples" / "models" / "mobilenetv2-12.onnx"
        return candidate if candidate.exists() else None

    def test_real_onnx_introspection(self, mobilenet_path: Path | None) -> None:
        if mobilenet_path is None:
            pytest.skip("examples/models/mobilenetv2-12.onnx not present")
        try:
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("onnx package not installed")

        result = introspect_model(mobilenet_path)
        assert result.source == "onnx-py"
        assert len(result.inputs) == 1
        assert result.inputs[0].name == "input"
        # MobileNetV2 input: [batch, 3, 224, 224]
        assert result.inputs[0].shape[1:] == (3, 224, 224)
        assert result.inputs[0].dtype == "float32"


# ─── DLC info parser ─────────────────────────────────────────────────────────


class TestDLCInfoParser:
    def test_parses_typical_output(self) -> None:
        sample = """\
DLC Info:
  Version: 2.45.0

Inputs:
  Input Name        | Dimensions    | Type
  ----------------- | ------------- | -----
  input             | 1,3,224,224   | float32

Outputs:
  Output Name       | Dimensions    | Type
  ----------------- | ------------- | -----
  output            | 1,1000        | float32

Total parameters: 4,253,864
"""
        result = _parse_dlc_info_output(sample)
        assert result is not None
        assert len(result.inputs) == 1
        assert result.inputs[0].name == "input"
        assert result.inputs[0].shape == (1, 3, 224, 224)
        assert result.inputs[0].dtype == "float32"
        assert len(result.outputs) == 1
        assert result.outputs[0].shape == (1, 1000)

    def test_returns_none_when_no_inputs_section(self) -> None:
        result = _parse_dlc_info_output("just some text\nno inputs here\n")
        assert result is None

    def test_parses_multiple_inputs(self) -> None:
        sample = """\
Inputs:
  Input Name | Dimensions | Type
  ---------- | ---------- | ----
  image      | 1,3,224,224 | float32
  prompt     | 1,512      | int32
"""
        result = _parse_dlc_info_output(sample)
        assert result is not None
        assert len(result.inputs) == 2
        assert result.inputs[0].name == "image"
        assert result.inputs[1].name == "prompt"
        assert result.inputs[1].dtype == "int32"


# ─── Fallback path ───────────────────────────────────────────────────────────


class TestIntrospectFallback:
    def test_unknown_extension_returns_fallback(self, tmp_path: Path) -> None:
        # A .qnn binary or anything else we don't know how to introspect
        unknown = tmp_path / "model.qnn"
        unknown.write_bytes(b"\x00" * 100)
        result = introspect_model(unknown)
        assert result.source == "fallback"
        assert result.inputs == [DEFAULT_FALLBACK_INPUT]


# ─── write_input_list / create_input_list_for_model ──────────────────────────


class TestWriteInputList:
    def test_single_input_writes_files(self, tmp_path: Path) -> None:
        model_io = ModelIO(
            inputs=[TensorSpec(name="input", shape=(1, 3, 4, 4), dtype="float32")],
        )
        list_path = write_input_list(model_io, output_dir=tmp_path)
        assert Path(list_path).exists()
        assert (tmp_path / "input.raw").exists()
        # Raw file size should match the spec (1*3*4*4 floats × 4 bytes)
        assert (tmp_path / "input.raw").stat().st_size == 1 * 3 * 4 * 4 * 4
        # input_list.txt should reference the raw file
        content = (tmp_path / "input_list.txt").read_text()
        assert "input.raw" in content

    def test_multiple_samples(self, tmp_path: Path) -> None:
        model_io = ModelIO(
            inputs=[TensorSpec(name="x", shape=(1, 3, 8, 8), dtype="float32")],
        )
        write_input_list(model_io, output_dir=tmp_path, num_samples=4)
        # Should have 4 input lines
        content = (tmp_path / "input_list.txt").read_text().strip().splitlines()
        assert len(content) == 4
        # And 4 .raw files (input0_x.raw … input3_x.raw)
        raws = sorted(tmp_path.glob("input*_x.raw"))
        assert len(raws) == 4

    def test_multi_input_uses_named_format(self, tmp_path: Path) -> None:
        model_io = ModelIO(
            inputs=[
                TensorSpec(name="image", shape=(1, 3, 4, 4), dtype="float32"),
                TensorSpec(name="prompt", shape=(1, 4), dtype="int32"),
            ],
        )
        list_path = write_input_list(model_io, output_dir=tmp_path)
        content = Path(list_path).read_text()
        # Multi-input lines use the "name:=path" syntax
        assert "image:=" in content
        assert "prompt:=" in content

    def test_calibration_data_used_instead_of_random(self, tmp_path: Path) -> None:
        model_io = ModelIO(
            inputs=[TensorSpec(name="x", shape=(1, 3, 4, 4), dtype="float32")],
        )
        cal = np.ones((1, 3, 4, 4), dtype=np.float32) * 7.5
        write_input_list(
            model_io,
            output_dir=tmp_path,
            calibration_data={"x": cal},
        )
        # Read back the .raw and verify it contains all 7.5 (not random)
        raw_bytes = (tmp_path / "input.raw").read_bytes()
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(1, 3, 4, 4)
        assert np.allclose(arr, 7.5)

    def test_empty_model_io_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no inputs"):
            write_input_list(ModelIO(), output_dir=tmp_path)


class TestCreateInputListForModel:
    def test_unknown_format_uses_fallback(self, tmp_path: Path) -> None:
        unknown = tmp_path / "model.qnn"
        unknown.write_bytes(b"")
        list_path, model_io = create_input_list_for_model(
            unknown, output_dir=tmp_path / "out"
        )
        assert Path(list_path).exists()
        assert model_io.source == "fallback"
        assert (tmp_path / "out" / "input.raw").exists()

    def test_seed_determinism(self, tmp_path: Path) -> None:
        unknown = tmp_path / "m.qnn"
        unknown.write_bytes(b"")
        d1 = tmp_path / "out1"
        d2 = tmp_path / "out2"
        create_input_list_for_model(unknown, output_dir=d1, seed=99)
        create_input_list_for_model(unknown, output_dir=d2, seed=99)
        assert (d1 / "input.raw").read_bytes() == (d2 / "input.raw").read_bytes()
