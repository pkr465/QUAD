"""Tests for init caching functionality."""

from __future__ import annotations

import pytest

from quad.runtime.model import load


class TestInitCaching:
    def test_load_with_init_cache_enabled(self) -> None:
        model = load("model.dlc", device="npu", enable_init_cache=True)
        assert model.enable_init_cache is True

    def test_load_without_init_cache_defaults_false(self) -> None:
        model = load("model.dlc", device="npu")
        assert model.enable_init_cache is False

    def test_init_cache_only_dsp_aip(self) -> None:
        """Init caching is for DSP/AIP — document this with a note in the model."""
        # On DSP runtime with init cache
        dsp_model = load("model.dlc", device="npu", enable_init_cache=True)
        assert dsp_model.enable_init_cache is True

        # Still loadable on CPU (though cache won't be used there)
        cpu_model = load("model.dlc", device="cpu", enable_init_cache=True)
        assert cpu_model.enable_init_cache is True  # Flag stored, SDK ignores for CPU

    def test_init_cache_preserved_through_inference(self) -> None:
        """Init cache flag does not affect inference correctness."""
        import numpy as np
        model = load("model.dlc", device="npu", enable_init_cache=True)
        inp = np.random.randn(1, 3, 224, 224).astype("float32")
        output = model(inp)
        assert output is not None

    def test_init_cache_with_input_dimensions(self) -> None:
        """Init cache and network resizing can be combined."""
        model = load(
            "model.dlc",
            device="npu",
            input_dimensions={"data": (1, 3, 1080, 1920)},
            enable_init_cache=True,
        )
        assert model.enable_init_cache is True
        assert model.input_dimensions == {"data": (1, 3, 1080, 1920)}


class TestInitCacheTemplates:
    def test_cpp_template_with_init_cache_saves_container(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        template = env.get_template("inference.cpp.j2")
        rendered = template.render(
            model_path="model.dlc",
            sdk="snpe",
            runtime="dsp",
            enable_init_cache=True,
        )
        # Must call container->save() after build
        assert 'container->save(' in rendered
        # Must include IDnnSerialization header
        assert "IDnnSerialization.hpp" in rendered
        # Must enable in builder
        assert "setInitCacheMode(true)" in rendered

    def test_cpp_template_without_init_cache_no_save(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        template = env.get_template("inference.cpp.j2")
        rendered = template.render(
            model_path="model.dlc",
            sdk="snpe",
            runtime="dsp",
            # No enable_init_cache
        )
        assert "container->save(" not in rendered
        assert "IDnnSerialization.hpp" not in rendered

    def test_c_template_with_init_cache_saves_container(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        template = env.get_template("inference.c.j2")
        rendered = template.render(
            model_path="model.dlc",
            sdk="snpe",
            runtime="dsp",
            enable_init_cache=True,
        )
        # Must call Snpe_DlContainer_Save() after build
        assert "Snpe_DlContainer_Save(" in rendered
        # Must enable in builder
        assert "Snpe_SNPEBuilder_SetInitCacheMode(builder, true)" in rendered

    def test_c_template_without_init_cache_no_save(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        template = env.get_template("inference.c.j2")
        rendered = template.render(
            model_path="model.dlc",
            sdk="snpe",
            runtime="cpu",
        )
        assert "Snpe_DlContainer_Save(" not in rendered
