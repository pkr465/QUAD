"""Tests for Protection Domain (PD) type — Signed vs Unsigned."""

from __future__ import annotations

import pytest

from quad.adapters.dsp_env import (
    PDType,
    RuntimeCheckOption,
    check_runtime_available,
    get_platform_option,
)
from quad.runtime.model import load


class TestPDTypeConstants:
    def test_unsigned_constant(self) -> None:
        assert PDType.UNSIGNED == "unsigned"

    def test_signed_constant(self) -> None:
        assert PDType.SIGNED == "signed"

    def test_runtime_check_options(self) -> None:
        assert RuntimeCheckOption.UNSIGNEDPD_CHECK == "UNSIGNEDPD_CHECK"
        assert RuntimeCheckOption.NORMAL_CHECK == "NORMAL_CHECK"
        assert RuntimeCheckOption.BASIC_CHECK == "BASIC_CHECK"


class TestPlatformOptions:
    def test_unsigned_pd_returns_on(self) -> None:
        assert get_platform_option(PDType.UNSIGNED) == "unsignedPD:ON"

    def test_signed_pd_returns_off(self) -> None:
        assert get_platform_option(PDType.SIGNED) == "unsignedPD:OFF"


class TestRuntimeCheckMatrix:
    """Verify the isRuntimeAvailable() truth table from the docs."""

    # Unsigned PD
    def test_unsigned_passes_unsignedpd_check(self) -> None:
        assert check_runtime_available(PDType.UNSIGNED, RuntimeCheckOption.UNSIGNEDPD_CHECK) is True

    def test_unsigned_fails_normal_check(self) -> None:
        assert check_runtime_available(PDType.UNSIGNED, RuntimeCheckOption.NORMAL_CHECK) is False

    def test_unsigned_fails_basic_check(self) -> None:
        assert check_runtime_available(PDType.UNSIGNED, RuntimeCheckOption.BASIC_CHECK) is False

    # Signed PD
    def test_signed_fails_unsignedpd_check(self) -> None:
        assert check_runtime_available(PDType.SIGNED, RuntimeCheckOption.UNSIGNEDPD_CHECK) is False

    def test_signed_passes_normal_check(self) -> None:
        assert check_runtime_available(PDType.SIGNED, RuntimeCheckOption.NORMAL_CHECK) is True

    def test_signed_passes_basic_check(self) -> None:
        assert check_runtime_available(PDType.SIGNED, RuntimeCheckOption.BASIC_CHECK) is True

    def test_default_check_is_unsignedpd(self) -> None:
        """Default check option should be UNSIGNEDPD_CHECK (unsigned is default)."""
        assert check_runtime_available(PDType.UNSIGNED) is True
        assert check_runtime_available(PDType.SIGNED) is False


class TestModelPDType:
    def test_default_pd_type_is_unsigned(self) -> None:
        model = load("model.dlc", device="npu")
        assert model.pd_type == "unsigned"

    def test_load_with_signed_pd(self) -> None:
        model = load("model.dlc", device="npu", pd_type="signed")
        assert model.pd_type == "signed"

    def test_platform_options_unsigned(self) -> None:
        model = load("model.dlc", device="npu", pd_type="unsigned")
        assert model.platform_options == "unsignedPD:ON"

    def test_platform_options_signed(self) -> None:
        model = load("model.dlc", device="npu", pd_type="signed")
        assert model.platform_options == "unsignedPD:OFF"

    def test_pd_type_preserved_through_inference(self) -> None:
        import numpy as np
        model = load("model.dlc", device="npu", pd_type="signed")
        output = model(np.random.randn(1, 3, 224, 224).astype("float32"))
        assert model.pd_type == "signed"  # unchanged


class TestPDTypeTemplates:
    def test_cpp_signed_pd_sets_platform_config(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("inference.cpp.j2")
        rendered = t.render(model_path="model.dlc", runtime="dsp", pd_type="signed")
        assert "PlatformConfig" in rendered
        assert "unsignedPD:OFF" in rendered
        assert "setPlatformConfig" in rendered

    def test_cpp_unsigned_pd_no_platform_config(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/cpp"))
        t = env.get_template("inference.cpp.j2")
        rendered = t.render(model_path="model.dlc", runtime="dsp")
        # No explicit platform config block for default unsigned PD
        assert "setPlatformConfig(platformConfig)" not in rendered

    def test_c_signed_pd_sets_platform_config(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        t = env.get_template("inference.c.j2")
        rendered = t.render(model_path="model.dlc", runtime="dsp", pd_type="signed")
        assert "Snpe_PlatformConfig_Create()" in rendered
        assert "unsignedPD:OFF" in rendered
        assert "Snpe_SNPEBuilder_SetPlatformConfig" in rendered
        assert "Snpe_PlatformConfig_Delete(platformConfig)" in rendered  # Must delete

    def test_c_unsigned_pd_no_platform_config(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/c"))
        t = env.get_template("inference.c.j2")
        rendered = t.render(model_path="model.dlc", runtime="dsp")
        assert "setPlatformConfig(platformConfig)" not in rendered

    def test_psnpe_config_has_unsigned_pd_on(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader("templates/snpe/psnpe"))
        t = env.get_template("model_configs.json.j2")
        rendered = t.render(model_name="test", model_file_path="model.dlc")
        assert "unsignedPD:ON" in rendered
