"""Tests for CPU Fixed Point Mode."""

from __future__ import annotations

import pytest

from quad.runtime.model import load


class TestCPUFixedPointMode:
    def test_default_is_disabled(self) -> None:
        """Default: CPU dequantizes to float32 for backward compatibility."""
        model = load("model.dlc", device="cpu")
        assert model.enable_cpu_fxp is False

    def test_load_with_cpu_fxp_enabled(self) -> None:
        model = load("model.dlc", device="cpu", enable_cpu_fxp=True)
        assert model.enable_cpu_fxp is True

    def test_fxp_preserved_through_inference(self) -> None:
        import numpy as np
        model = load("model.dlc", device="cpu", enable_cpu_fxp=True)
        out = model(np.random.randn(1, 3, 224, 224).astype("float32"))
        assert model.enable_cpu_fxp is True
        assert out is not None

    def test_fxp_combined_with_other_options(self) -> None:
        """CPU FXP + init cache + network resize all work together."""
        model = load(
            "model.dlc",
            device="cpu",
            enable_cpu_fxp=True,
            enable_init_cache=True,
            input_dimensions={"input": (1, 3, 224, 224)},
        )
        assert model.enable_cpu_fxp is True
        assert model.enable_init_cache is True


class TestCPUFXPTemplates:
    def test_cpp_template_with_cpu_fxp(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("inference.cpp.j2")
        rendered = t.render(model_path="model.dlc", runtime="cpu", enable_cpu_fxp=True)
        assert "setCpuFixedPointMode(true)" in rendered

    def test_cpp_template_without_cpu_fxp(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("inference.cpp.j2")
        rendered = t.render(model_path="model.dlc", runtime="cpu")
        assert "setCpuFixedPointMode" not in rendered

    def test_c_template_with_cpu_fxp(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        t = env.get_template("inference.c.j2")
        rendered = t.render(model_path="model.dlc", runtime="cpu", enable_cpu_fxp=True)
        assert "Snpe_SNPEBuilder_SetCpuFixedPointMode(builder, true)" in rendered

    def test_c_template_without_cpu_fxp(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        t = env.get_template("inference.c.j2")
        rendered = t.render(model_path="model.dlc", runtime="cpu")
        assert "SetCpuFixedPointMode" not in rendered

    def test_android_template_with_cpu_fxp(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/android"))
        t = env.get_template("InferenceEngine.kt.j2")
        rendered = t.render(model_path="model.dlc", enable_cpu_fxp=True)
        assert "setCpuFixedPointMode(true)" in rendered

    def test_android_template_without_cpu_fxp(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/android"))
        t = env.get_template("InferenceEngine.kt.j2")
        rendered = t.render(model_path="model.dlc")
        assert "setCpuFixedPointMode" not in rendered
