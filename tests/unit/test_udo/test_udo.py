"""Tests for QUAD UDO (User-Defined Operations)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quad.udo import UDOManager, UDOPackage, UDORuntime


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _force_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure QAIRT_SDK_ROOT is unset so all tests run in mock mode."""
    monkeypatch.delenv("QAIRT_SDK_ROOT", raising=False)


@pytest.fixture()
def mgr() -> UDOManager:
    return UDOManager()


@pytest.fixture()
def softmax_config(tmp_path: Path) -> str:
    """Write a minimal UDO JSON config for a Softmax_Htp package."""
    cfg = tmp_path / "Softmax_Htp.json"
    cfg.write_text(
        '{"UdoPackage": {"PackageName": "SoftmaxUdoPackage", '
        '"Operators": [{"type": "Softmax", "runtimes": '
        '[{"runtime": "cpu"}, {"runtime": "htp"}]}]}}'
    )
    return str(cfg)


@pytest.fixture()
def generated_pkg(mgr: UDOManager, softmax_config: str, tmp_path: Path) -> UDOPackage:
    return mgr.generate_package(
        config_json=softmax_config,
        output_dir=str(tmp_path / "pkgs"),
    )


# ---------------------------------------------------------------------------
# UDORuntime enum
# ---------------------------------------------------------------------------

class TestUDORuntime:
    def test_enum_values(self) -> None:
        assert UDORuntime.CPU.value == "cpu"
        assert UDORuntime.GPU.value == "gpu"
        assert UDORuntime.DSP_V65.value == "dsp_v65"
        assert UDORuntime.DSP_V66.value == "dsp_v66"
        assert UDORuntime.DSP_V68.value == "dsp_v68"
        assert UDORuntime.HTP.value == "htp"
        assert UDORuntime.AIP.value == "aip"

    def test_all_runtimes_present(self) -> None:
        values = {r.value for r in UDORuntime}
        assert {"cpu", "gpu", "dsp_v65", "dsp_v66", "dsp_v68", "htp", "aip"} == values


# ---------------------------------------------------------------------------
# UDOManager — mock mode detection
# ---------------------------------------------------------------------------

class TestUDOManagerMockMode:
    def test_mock_when_env_unset(self) -> None:
        assert UDOManager().is_mock is True

    def test_real_when_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt")
        mgr = UDOManager()
        assert mgr.is_mock is False

    def test_sdk_root_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QAIRT_SDK_ROOT", raising=False)
        mgr = UDOManager(sdk_root="/opt/qairt")
        assert mgr.is_mock is False

    def test_explicit_none_still_mock(self) -> None:
        mgr = UDOManager(sdk_root=None)
        assert mgr.is_mock is True


# ---------------------------------------------------------------------------
# generate_package
# ---------------------------------------------------------------------------

class TestGeneratePackage:
    def test_returns_udo_package(self, mgr: UDOManager, softmax_config: str, tmp_path: Path) -> None:
        pkg = mgr.generate_package(
            config_json=softmax_config,
            output_dir=str(tmp_path / "pkgs"),
        )
        assert isinstance(pkg, UDOPackage)

    def test_package_name_from_config(self, mgr: UDOManager, softmax_config: str, tmp_path: Path) -> None:
        pkg = mgr.generate_package(
            config_json=softmax_config,
            output_dir=str(tmp_path / "pkgs"),
        )
        assert pkg.name == "SoftmaxUdoPackage"

    def test_package_dir_contains_package_name(
        self, mgr: UDOManager, softmax_config: str, tmp_path: Path
    ) -> None:
        pkg = mgr.generate_package(
            config_json=softmax_config,
            output_dir=str(tmp_path / "pkgs"),
        )
        assert "SoftmaxUdoPackage" in pkg.package_dir

    def test_package_dir_is_under_output_dir(
        self, mgr: UDOManager, softmax_config: str, tmp_path: Path
    ) -> None:
        out_dir = str(tmp_path / "pkgs")
        pkg = mgr.generate_package(config_json=softmax_config, output_dir=out_dir)
        assert pkg.package_dir.startswith(out_dir)

    def test_config_json_stored(self, mgr: UDOManager, softmax_config: str, tmp_path: Path) -> None:
        pkg = mgr.generate_package(
            config_json=softmax_config,
            output_dir=str(tmp_path / "pkgs"),
        )
        assert pkg.config_json == softmax_config

    def test_supported_runtimes_parsed(
        self, mgr: UDOManager, softmax_config: str, tmp_path: Path
    ) -> None:
        pkg = mgr.generate_package(
            config_json=softmax_config,
            output_dir=str(tmp_path / "pkgs"),
        )
        assert "cpu" in pkg.supported_runtimes
        assert "htp" in pkg.supported_runtimes

    def test_not_compiled_initially(
        self, mgr: UDOManager, softmax_config: str, tmp_path: Path
    ) -> None:
        pkg = mgr.generate_package(
            config_json=softmax_config,
            output_dir=str(tmp_path / "pkgs"),
        )
        assert pkg.is_compiled is False

    def test_nonexistent_config_uses_filename_heuristic(
        self, mgr: UDOManager, tmp_path: Path
    ) -> None:
        """A missing config file falls back to deriving names from the filename."""
        fake_config = str(tmp_path / "MyOp_Htp.json")
        pkg = mgr.generate_package(
            config_json=fake_config,
            output_dir=str(tmp_path / "pkgs"),
        )
        assert "MyOp" in pkg.name or "MyOpUdoPackage" in pkg.name
        assert "htp" in pkg.supported_runtimes


# ---------------------------------------------------------------------------
# convert_model
# ---------------------------------------------------------------------------

class TestConvertModel:
    def test_returns_dlc_path(self, mgr: UDOManager, tmp_path: Path) -> None:
        output = mgr.convert_model(
            model_path="model.onnx",
            output_dlc=str(tmp_path / "model.dlc"),
            udo_config="Softmax_Htp.json",
        )
        assert output.endswith(".dlc")

    def test_output_path_contains_expected_stem(self, mgr: UDOManager, tmp_path: Path) -> None:
        output = mgr.convert_model(
            model_path="resnet50.onnx",
            output_dlc=str(tmp_path / "resnet50.dlc"),
            udo_config="config.json",
        )
        assert "resnet50" in output

    def test_default_format_onnx(self, mgr: UDOManager, tmp_path: Path) -> None:
        # No source_format arg — should not raise
        out = mgr.convert_model(
            model_path="model.onnx",
            output_dlc=str(tmp_path / "model.dlc"),
            udo_config="config.json",
        )
        assert out

    def test_tensorflow_format_accepted(self, mgr: UDOManager, tmp_path: Path) -> None:
        out = mgr.convert_model(
            model_path="model.pb",
            output_dlc=str(tmp_path / "model.dlc"),
            udo_config="config.json",
            source_format="tensorflow",
        )
        assert out.endswith(".dlc")

    def test_unsupported_format_raises(self, mgr: UDOManager, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported source_format"):
            mgr.convert_model(
                model_path="model.tflite",
                output_dlc=str(tmp_path / "model.dlc"),
                udo_config="config.json",
                source_format="tflite_unknown",
            )


# ---------------------------------------------------------------------------
# compile_package
# ---------------------------------------------------------------------------

class TestCompilePackage:
    def test_returns_dict(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="cpu_android")
        assert isinstance(result, dict)

    def test_contains_reg_lib(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="cpu_android")
        reg_keys = [k for k in result if "Reg" in k]
        assert reg_keys, f"No Reg lib found in {list(result)}"

    def test_contains_impl_lib(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="cpu_android")
        impl_keys = [k for k in result if "Impl" in k]
        assert impl_keys, f"No Impl lib found in {list(result)}"

    def test_lib_paths_are_strings(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="cpu_x86")
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_lib_paths_end_with_so(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="cpu_android")
        for path in result.values():
            assert path.endswith(".so"), f"Expected .so, got {path}"

    def test_cpu_x86_runtime(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="cpu_x86")
        assert len(result) >= 2

    def test_dsp_v68_runtime(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="dsp_v68")
        assert isinstance(result, dict)

    def test_gpu_android_runtime(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="gpu_android")
        assert len(result) >= 2

    def test_package_dir_in_lib_paths(self, mgr: UDOManager, generated_pkg: UDOPackage) -> None:
        result = mgr.compile_package(generated_pkg.package_dir, runtime="cpu_android")
        for path in result.values():
            assert generated_pkg.package_dir in path


# ---------------------------------------------------------------------------
# quantize_with_udo
# ---------------------------------------------------------------------------

class TestQuantizeWithUDO:
    def test_returns_string(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.quantize_with_udo(
            input_dlc=str(tmp_path / "model.dlc"),
            output_dlc=str(tmp_path / "model_q.dlc"),
            input_list="inputs.txt",
            reg_lib_path=str(tmp_path / "libReg.so"),
        )
        assert isinstance(result, str)

    def test_returns_dlc_path(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.quantize_with_udo(
            input_dlc=str(tmp_path / "model.dlc"),
            output_dlc=str(tmp_path / "model_q.dlc"),
            input_list="inputs.txt",
            reg_lib_path=str(tmp_path / "libReg.so"),
        )
        assert result.endswith(".dlc")

    def test_output_marked_quantized(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.quantize_with_udo(
            input_dlc=str(tmp_path / "model.dlc"),
            output_dlc=str(tmp_path / "model_q.dlc"),
            input_list="inputs.txt",
            reg_lib_path=str(tmp_path / "libReg.so"),
        )
        assert "quantized" in Path(result).name

    def test_htp_flag_accepted(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.quantize_with_udo(
            input_dlc=str(tmp_path / "model.dlc"),
            output_dlc=str(tmp_path / "model_q.dlc"),
            input_list="inputs.txt",
            reg_lib_path=str(tmp_path / "libReg.so"),
            enable_htp=True,
            htp_socs="sm8750",
        )
        assert result.endswith(".dlc")


# ---------------------------------------------------------------------------
# UDOPackage — get_reg_lib / get_impl_lib
# ---------------------------------------------------------------------------

class TestUDOPackageLibPaths:
    @pytest.fixture()
    def pkg(self, tmp_path: Path) -> UDOPackage:
        return UDOPackage(
            name="SoftmaxUdoPackage",
            package_dir=str(tmp_path / "SoftmaxUdoPackage"),
            config_json="Softmax_Htp.json",
            supported_runtimes=["cpu", "htp"],
        )

    def test_get_reg_lib_ends_with_so(self, pkg: UDOPackage) -> None:
        path = pkg.get_reg_lib("cpu")
        assert path.endswith(".so")

    def test_get_reg_lib_contains_package_name(self, pkg: UDOPackage) -> None:
        path = pkg.get_reg_lib("cpu")
        assert "SoftmaxUdoPackage" in path

    def test_get_reg_lib_contains_reg(self, pkg: UDOPackage) -> None:
        path = pkg.get_reg_lib("cpu")
        assert "Reg" in Path(path).name

    def test_get_reg_lib_default_arch(self, pkg: UDOPackage) -> None:
        path = pkg.get_reg_lib("cpu")
        assert "arm64-v8a" in path

    def test_get_reg_lib_custom_arch(self, pkg: UDOPackage) -> None:
        path = pkg.get_reg_lib("cpu", arch="x86-64_linux_clang")
        assert "x86-64_linux_clang" in path

    def test_get_impl_lib_ends_with_so(self, pkg: UDOPackage) -> None:
        path = pkg.get_impl_lib("cpu")
        assert path.endswith(".so")

    def test_get_impl_lib_contains_impl(self, pkg: UDOPackage) -> None:
        path = pkg.get_impl_lib("cpu")
        assert "Impl" in Path(path).name

    def test_get_impl_lib_contains_runtime_tag(self, pkg: UDOPackage) -> None:
        path = pkg.get_impl_lib("cpu")
        assert "CPU" in Path(path).name

    def test_get_impl_lib_htp_tag(self, pkg: UDOPackage) -> None:
        path = pkg.get_impl_lib("htp")
        assert "HTP" in Path(path).name

    def test_get_reg_lib_prefers_recorded_libs(self, pkg: UDOPackage, tmp_path: Path) -> None:
        """When libs dict is populated, get_reg_lib returns that recorded path."""
        reg_path = str(tmp_path / "libSoftmaxUdoPackageReg.so")
        pkg.libs = {"cpu": [reg_path]}
        assert pkg.get_reg_lib("cpu") == reg_path

    def test_get_impl_lib_prefers_recorded_libs(self, pkg: UDOPackage, tmp_path: Path) -> None:
        impl_path = str(tmp_path / "libSoftmaxUdoPackageImplCPU.so")
        pkg.libs = {"cpu": [impl_path]}
        assert pkg.get_impl_lib("cpu") == impl_path


# ---------------------------------------------------------------------------
# CLI command construction (mock mode — inspect logs / no subprocess)
# ---------------------------------------------------------------------------

class TestCLICommands:
    """Verify that in real mode the correct CLI commands would be built.

    We test by constructing manager internals directly and checking the
    helper ``_sdk_bin`` method returns the expected path.
    """

    def test_sdk_bin_returns_correct_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt")
        mgr = UDOManager()
        assert mgr._sdk_bin("snpe-udo-package-generator") == "/opt/qairt/bin/snpe-udo-package-generator"

    def test_sdk_bin_for_quant(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt")
        mgr = UDOManager()
        assert mgr._sdk_bin("snpe-dlc-quant") == "/opt/qairt/bin/snpe-dlc-quant"

    def test_sdk_bin_for_converter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt")
        mgr = UDOManager()
        assert mgr._sdk_bin("snpe-onnx-to-dlc") == "/opt/qairt/bin/snpe-onnx-to-dlc"


# ---------------------------------------------------------------------------
# deploy_to_android / execute_on_android (mock — no-op / string return)
# ---------------------------------------------------------------------------

class TestDeployAndExecute:
    def test_deploy_does_not_raise_in_mock(
        self, mgr: UDOManager, generated_pkg: UDOPackage, tmp_path: Path
    ) -> None:
        # Should complete without error in mock mode
        mgr.deploy_to_android(
            package_dir=generated_pkg.package_dir,
            model_dlc=str(tmp_path / "model.dlc"),
            input_list=str(tmp_path / "inputs.txt"),
            runtime="cpu",
        )

    def test_execute_returns_string(self, mgr: UDOManager) -> None:
        result = mgr.execute_on_android(
            model_dlc="model.dlc",
            input_list="inputs.txt",
            reg_lib="libReg.so",
            runtime="cpu",
        )
        assert isinstance(result, str)

    def test_execute_mock_output_contains_model(self, mgr: UDOManager) -> None:
        result = mgr.execute_on_android(
            model_dlc="my_model.dlc",
            input_list="inputs.txt",
            reg_lib="libReg.so",
            runtime="dsp",
        )
        assert "my_model.dlc" in result

    def test_execute_mock_output_mentions_runtime(self, mgr: UDOManager) -> None:
        result = mgr.execute_on_android(
            model_dlc="model.dlc",
            input_list="inputs.txt",
            reg_lib="libReg.so",
            runtime="gpu",
        )
        assert "gpu" in result.lower()
