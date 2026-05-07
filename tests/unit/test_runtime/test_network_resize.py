"""Tests for network resizing (dynamic input dimensions)."""

from __future__ import annotations

import pytest

from quad.runtime.model import Model, load, InputShapeMap
from quad.runtime.device import Device


class TestModelInputDimensions:
    def test_load_with_input_dimensions(self) -> None:
        """Model can be loaded with custom input shapes."""
        model = load(
            "model.dlc",
            device="npu",
            input_dimensions={"data": (3, 1080, 1440, 3)},
        )
        assert model.input_dimensions == {"data": (3, 1080, 1440, 3)}
        assert (3, 1080, 1440, 3) in model.input_shapes

    def test_load_without_input_dimensions_uses_defaults(self) -> None:
        model = load("model.dlc", device="npu")
        assert model.input_dimensions == {}
        assert model.input_shapes == [(1, 3, 224, 224)]

    def test_resize_input_updates_shapes(self) -> None:
        model = load("model.dlc", device="npu")
        assert model.input_shapes == [(1, 3, 224, 224)]

        model.resize_input({"input": (1, 3, 1080, 1920)})

        assert model.input_dimensions == {"input": (1, 3, 1080, 1920)}
        assert (1, 3, 1080, 1920) in model.input_shapes

    def test_resize_batch_size(self) -> None:
        """Common use case: change batch size from 1 to 4."""
        model = load("model.dlc", device="npu")
        model.resize_input({"input": (4, 3, 224, 224)})
        assert model.input_dimensions["input"][0] == 4

    def test_multiple_inputs_resize(self) -> None:
        """Multi-input model with different resized shapes."""
        model = load("model.dlc", device="npu", input_dimensions={
            "image": (1, 3, 640, 640),
            "meta": (1, 4),
        })
        assert len(model.input_dimensions) == 2
        assert model.input_dimensions["image"] == (1, 3, 640, 640)
        assert model.input_dimensions["meta"] == (1, 4)

    def test_resize_does_not_affect_other_properties(self) -> None:
        model = load("model.dlc", device="npu")
        original_device = model.device.type
        model.resize_input({"data": (2, 3, 299, 299)})
        assert model.device.type == original_device
        assert model.is_loaded

    def test_inference_works_after_resize(self) -> None:
        """Model is still callable after resize."""
        import numpy as np
        model = load("model.dlc", device="npu")
        model.resize_input({"input": (1, 3, 299, 299)})
        input_data = np.random.randn(1, 3, 299, 299).astype("float32")
        output = model(input_data)
        assert output is not None


class TestResizeInTemplates:
    """Verify template rendering with input_dimensions variable."""

    def test_cpp_template_with_resize(self) -> None:
        """Verify TensorShapeMap appears when input_dimensions is set."""
        from pathlib import Path
        from jinja2 import Environment, FileSystemLoader

        # Render the snpe/cpp template directly via Jinja2
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        template = env.get_template("inference.cpp.j2")
        rendered = template.render(
            model_path="model.dlc",
            sdk="snpe",
            runtime="dsp",
            input_dimensions={"data": [3, 1080, 1440, 3]},
        )
        assert "TensorShapeMap" in rendered
        assert "setInputDimensions" in rendered
        assert "1080" in rendered

    def test_cpp_template_without_resize(self) -> None:
        """Without input_dimensions, no TensorShapeMap code is generated."""
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        template = env.get_template("inference.cpp.j2")
        rendered = template.render(
            model_path="model.dlc",
            sdk="snpe",
            runtime="cpu",
        )
        assert "builder.setInputDimensions(inputShapeMap)" not in rendered
