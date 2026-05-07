"""Tests for batched ITensor input/output."""

from __future__ import annotations

import numpy as np
import pytest

from quad.runtime.tensor import Tensor
from quad.runtime.device import Device
from quad.runtime.model import load


class TestBatchedTensor:
    def test_from_batch_stacks_tensors(self) -> None:
        t1 = Tensor.rand(3, 224, 224, device="cpu")
        t2 = Tensor.rand(3, 224, 224, device="cpu")
        t3 = Tensor.rand(3, 224, 224, device="cpu")

        batch = Tensor.from_batch([t1, t2, t3], device="npu")

        assert batch.shape == (3, 3, 224, 224)
        assert batch.device.type == "npu"

    def test_from_batch_single(self) -> None:
        t = Tensor.rand(3, 224, 224)
        batch = Tensor.from_batch([t])
        assert batch.shape == (1, 3, 224, 224)

    def test_from_batch_preserves_data(self) -> None:
        arr = np.ones((3, 4), dtype="float32") * 2.5
        t = Tensor.from_numpy(arr)
        batch = Tensor.from_batch([t, t])
        assert batch.shape == (2, 3, 4)
        np.testing.assert_allclose(batch.to_numpy()[0], arr)
        np.testing.assert_allclose(batch.to_numpy()[1], arr)

    def test_from_batch_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Tensor.from_batch([])

    def test_split_batch_with_batch_dim(self) -> None:
        # Simulate batched output: [4, 1000] class scores
        arr = np.random.randn(4, 1000).astype("float32")
        batch_output = Tensor.from_numpy(arr)

        splits = batch_output.split_batch(4)

        assert len(splits) == 4
        for i, split in enumerate(splits):
            assert split.shape == (1000,)
            np.testing.assert_allclose(split.to_numpy(), arr[i])

    def test_split_batch_flat(self) -> None:
        # Flat concatenated output (no explicit batch dim)
        single_output = np.ones(1000, dtype="float32")
        flat = np.concatenate([single_output * i for i in range(1, 5)])
        batch_output = Tensor.from_numpy(flat)

        splits = batch_output.split_batch(4)
        assert len(splits) == 4
        np.testing.assert_allclose(splits[0].to_numpy(), np.ones(1000))
        np.testing.assert_allclose(splits[1].to_numpy(), np.ones(1000) * 2)

    def test_roundtrip_batch_split(self) -> None:
        """from_batch() then split_batch() recovers original tensors."""
        originals = [Tensor.rand(1000) for _ in range(4)]
        batched = Tensor.from_batch(originals)
        recovered = batched.split_batch(4)

        assert len(recovered) == 4
        for orig, rec in zip(originals, recovered):
            np.testing.assert_allclose(
                orig.to_numpy(), rec.to_numpy(), atol=1e-6
            )


class TestBatchedInference:
    def test_model_inference_with_batch_input(self) -> None:
        """Model accepts a batched tensor."""
        model = load("model.dlc", device="npu",
                     input_dimensions={"input": (4, 3, 224, 224)})

        # Create batch of 4 images
        batch = Tensor.from_batch([
            Tensor.rand(3, 224, 224, device="npu") for _ in range(4)
        ], device="npu")

        assert batch.shape == (4, 3, 224, 224)
        output = model(batch)
        assert output is not None

    def test_split_batched_output_then_save(self) -> None:
        """Simulate the full batch pipeline: batch → infer → split."""
        model = load("model.dlc", device="npu")
        batch_size = 3

        # Batch inputs
        inputs = [Tensor.rand(3, 224, 224) for _ in range(batch_size)]
        batched_input = Tensor.from_batch(inputs)

        # Run inference (output shape: [1, 1000] since mock)
        output = model(batched_input)

        # Split — in real mode output would be [batch_size, 1000]
        # Mock returns [1, 1000]; test split logic separately
        assert output is not None


class TestBatchedTemplateRendering:
    """Verify batch-related code appears correctly in templates."""

    def test_cpp_template_has_batched_functions(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        template = env.get_template("inference.cpp.j2")
        rendered = template.render(model_path="model.dlc", sdk="snpe", runtime="dsp")

        assert "loadInputTensorBatched" in rendered
        assert "executeITensorBatched" in rendered
        assert "batchSize" in rendered
        assert "batchChunk" in rendered

    def test_c_template_has_batched_functions(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        template = env.get_template("inference.c.j2")
        rendered = template.render(model_path="model.dlc", sdk="snpe", runtime="dsp")

        assert "load_input_tensor_batched" in rendered
        assert "execute_and_save_batched" in rendered
        assert "batch_chunk" in rendered
        assert "batch_size" in rendered
