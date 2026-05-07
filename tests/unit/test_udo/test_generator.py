"""Tests for QUAD UDO generator — new methods and templates.

Covers:
  - validate_package_structure with mock directory
  - get_implementation_todos returns correct functions per runtime
  - check_environment reports missing vars
  - GPU template renders with OpenCL patterns
  - RegLib template renders with correct API functions
  - Makefile template has all expected targets
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from quad.udo import UDOManager


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure QAIRT_SDK_ROOT is unset so all tests run in mock mode."""
    monkeypatch.delenv("QAIRT_SDK_ROOT", raising=False)
    monkeypatch.delenv("SNPE_UDO_ROOT", raising=False)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.delenv("HEXAGON_SDK_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_NDK_ROOT", raising=False)
    monkeypatch.delenv("CL_LIBRARY_PATH", raising=False)


@pytest.fixture()
def mgr() -> UDOManager:
    return UDOManager()


@pytest.fixture()
def templates_dir() -> Path:
    """Return the path to the UDO templates directory."""
    return Path(__file__).resolve().parents[3] / "templates" / "snpe" / "udo"


# ---------------------------------------------------------------------------
# validate_package_structure
# ---------------------------------------------------------------------------


class TestValidatePackageStructure:
    """Tests for UDOManager.validate_package_structure (mock mode)."""

    def test_returns_dict(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        assert isinstance(result, dict)

    def test_valid_key_present(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        assert "valid" in result

    def test_mock_returns_valid_true(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        assert result["valid"] is True

    def test_found_files_not_empty(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        assert len(result["found_files"]) > 0

    def test_missing_files_empty_in_mock(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        assert result["missing_files"] == []

    def test_needs_implementation_populated(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        assert len(result["needs_implementation"]) > 0

    def test_package_dir_in_result(self, mgr: UDOManager, tmp_path: Path) -> None:
        pkg_dir = str(tmp_path / "TestPkg")
        result = mgr.validate_package_structure(pkg_dir)
        assert result["package_dir"] == pkg_dir

    def test_found_files_contain_makefile(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        assert "Makefile" in result["found_files"]

    def test_found_files_contain_reg_lib(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        reg_files = [f for f in result["found_files"] if "Reg" in f]
        assert reg_files

    def test_needs_implementation_has_cpu(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.validate_package_structure(str(tmp_path / "SoftmaxUdoPackage"))
        cpu_files = [f for f in result["needs_implementation"] if "CPU" in f]
        assert cpu_files


# ---------------------------------------------------------------------------
# get_implementation_todos
# ---------------------------------------------------------------------------


class TestGetImplementationTodos:
    """Tests for UDOManager.get_implementation_todos (mock mode)."""

    def test_returns_list(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        assert isinstance(result, list)

    def test_has_cpu_entry(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        runtimes = [item["runtime"] for item in result]
        assert "CPU" in runtimes

    def test_has_gpu_entry(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        runtimes = [item["runtime"] for item in result]
        assert "GPU" in runtimes

    def test_has_dsp_entry(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        runtimes = [item["runtime"] for item in result]
        assert "DSP_V68" in runtimes

    def test_cpu_functions_include_execute(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        cpu_entry = next(item for item in result if item["runtime"] == "CPU")
        assert "SnpeUdo_executeOp" in cpu_entry["functions"]

    def test_cpu_functions_include_validate(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        cpu_entry = next(item for item in result if item["runtime"] == "CPU")
        assert "SnpeUdo_validateOp" in cpu_entry["functions"]

    def test_gpu_functions_include_set_kernel_info(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        gpu_entry = next(item for item in result if item["runtime"] == "GPU")
        assert "setKernelInfo" in gpu_entry["functions"]

    def test_dsp_functions_include_execute_op(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        dsp_entry = next(item for item in result if item["runtime"] == "DSP_V68")
        assert "QnnOpPackage_executeOp" in dsp_entry["functions"]

    def test_each_entry_has_file_key(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        for item in result:
            assert "file" in item
            assert item["file"]  # non-empty

    def test_each_entry_has_notes(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        for item in result:
            assert "notes" in item
            assert len(item["notes"]) > 10  # meaningful guidance

    def test_gpu_notes_mention_opencl(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        gpu_entry = next(item for item in result if item["runtime"] == "GPU")
        assert "OpenCL" in gpu_entry["notes"] or "cl_mem" in gpu_entry["notes"]

    def test_dsp_notes_mention_quantised(self, mgr: UDOManager, tmp_path: Path) -> None:
        result = mgr.get_implementation_todos(str(tmp_path / "SoftmaxUdoPackage"))
        dsp_entry = next(item for item in result if item["runtime"] == "DSP_V68")
        assert "quantis" in dsp_entry["notes"].lower() or "INT8" in dsp_entry["notes"]


# ---------------------------------------------------------------------------
# check_environment
# ---------------------------------------------------------------------------


class TestCheckEnvironment:
    """Tests for UDOManager.check_environment (mock mode)."""

    def test_returns_dict(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        assert isinstance(result, dict)

    def test_ready_false_in_mock(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        assert result["ready"] is False

    def test_can_generate_false_in_mock(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        assert result["can_generate"] is False

    def test_can_compile_cpu_false_in_mock(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        assert result["can_compile_cpu"] is False

    def test_can_compile_gpu_false_in_mock(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        assert result["can_compile_gpu"] is False

    def test_can_compile_dsp_false_in_mock(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        assert result["can_compile_dsp"] is False

    def test_checks_list_populated(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        assert len(result["checks"]) > 0

    def test_checks_have_var_and_status(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        for check in result["checks"]:
            assert "var" in check
            assert "status" in check
            assert check["status"] in ("set", "missing")

    def test_checks_include_snpe_udo_root(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        vars_checked = [c["var"] for c in result["checks"]]
        assert "SNPE_UDO_ROOT" in vars_checked

    def test_checks_include_hexagon_sdk(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        vars_checked = [c["var"] for c in result["checks"]]
        assert "HEXAGON_SDK_ROOT" in vars_checked

    def test_checks_include_android_ndk(self, mgr: UDOManager) -> None:
        result = mgr.check_environment()
        vars_checked = [c["var"] for c in result["checks"]]
        assert "ANDROID_NDK_ROOT" in vars_checked

    def test_real_mode_with_sdk_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When SDK vars are set, real mode reports can_generate=True."""
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt")
        monkeypatch.setenv("QNN_SDK_ROOT", "/opt/qnn")
        mgr = UDOManager(sdk_root="/opt/qairt")
        result = mgr.check_environment()
        assert result["can_generate"] is True
        assert result["can_compile_cpu"] is True

    def test_real_mode_gpu_needs_ndk_and_cl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GPU compilation requires ANDROID_NDK_ROOT + CL_LIBRARY_PATH."""
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt")
        monkeypatch.delenv("ANDROID_NDK_ROOT", raising=False)
        monkeypatch.delenv("CL_LIBRARY_PATH", raising=False)
        mgr = UDOManager(sdk_root="/opt/qairt")
        result = mgr.check_environment()
        assert result["can_compile_gpu"] is False

    def test_real_mode_dsp_needs_hexagon(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DSP compilation requires HEXAGON_SDK_ROOT."""
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/opt/qairt")
        monkeypatch.delenv("HEXAGON_SDK_ROOT", raising=False)
        mgr = UDOManager(sdk_root="/opt/qairt")
        result = mgr.check_environment()
        assert result["can_compile_dsp"] is False


# ---------------------------------------------------------------------------
# Template rendering — GPU
# ---------------------------------------------------------------------------


class TestGpuTemplate:
    """Tests that the GPU template renders with expected OpenCL patterns."""

    @pytest.fixture()
    def rendered_gpu(self, templates_dir: Path) -> str:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("gpu_impl.cpp.j2")
        return template.render(
            package_name="SoftmaxUdoPackage",
            op_name="Softmax",
            num_inputs=1,
            num_outputs=1,
        )

    def test_contains_opencl_header(self, rendered_gpu: str) -> None:
        assert "CL/cl.h" in rendered_gpu or "OpenCL/opencl.h" in rendered_gpu

    def test_contains_kernel_source(self, rendered_gpu: str) -> None:
        assert "__kernel" in rendered_gpu

    def test_contains_half_type(self, rendered_gpu: str) -> None:
        assert "half" in rendered_gpu

    def test_contains_cl_mem(self, rendered_gpu: str) -> None:
        assert "cl_mem" in rendered_gpu

    def test_contains_set_kernel_info(self, rendered_gpu: str) -> None:
        assert "setKernelInfo" in rendered_gpu

    def test_contains_op_name_in_kernel(self, rendered_gpu: str) -> None:
        assert "softmax_kernel" in rendered_gpu

    def test_contains_clEnqueueNDRangeKernel(self, rendered_gpu: str) -> None:
        assert "clEnqueueNDRangeKernel" in rendered_gpu

    def test_contains_extern_c(self, rendered_gpu: str) -> None:
        assert 'extern "C"' in rendered_gpu

    def test_contains_package_namespace(self, rendered_gpu: str) -> None:
        assert "SoftmaxUdoPackage" in rendered_gpu

    def test_contains_gpu_custom_op_include(self, rendered_gpu: str) -> None:
        assert "GpuCustomOpPackage.hpp" in rendered_gpu

    def test_contains_clSetKernelArg(self, rendered_gpu: str) -> None:
        assert "clSetKernelArg" in rendered_gpu


# ---------------------------------------------------------------------------
# Template rendering — Registration Library
# ---------------------------------------------------------------------------


class TestRegLibTemplate:
    """Tests that the RegLib template renders with correct API functions."""

    @pytest.fixture()
    def rendered_reglib(self, templates_dir: Path) -> str:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("reglib.cpp.j2")
        return template.render(
            package_name="SoftmaxUdoPackage",
            op_name="Softmax",
            supported_runtimes=["CPU", "GPU", "DSP_V68"],
            impl_lib_names={
                "CPU": "libSoftmaxUdoPackageImplCpu.so",
                "GPU": "libSoftmaxUdoPackageImplGpu.so",
                "DSP_V68": "libSoftmaxUdoPackageImplDsp.so",
            },
        )

    def test_contains_get_version(self, rendered_reglib: str) -> None:
        assert "SnpeUdo_getVersion" in rendered_reglib

    def test_contains_get_reg_info(self, rendered_reglib: str) -> None:
        assert "SnpeUdo_getRegInfo" in rendered_reglib

    def test_contains_validate_operation(self, rendered_reglib: str) -> None:
        assert "SnpeUdo_validateOperation" in rendered_reglib

    def test_contains_extern_c(self, rendered_reglib: str) -> None:
        assert 'extern "C"' in rendered_reglib

    def test_contains_package_name(self, rendered_reglib: str) -> None:
        assert "SoftmaxUdoPackage" in rendered_reglib

    def test_contains_op_name(self, rendered_reglib: str) -> None:
        assert "Softmax" in rendered_reglib

    def test_contains_cpu_core_type(self, rendered_reglib: str) -> None:
        assert "SNPE_UDO_CORETYPE_CPU" in rendered_reglib

    def test_contains_gpu_core_type(self, rendered_reglib: str) -> None:
        assert "SNPE_UDO_CORETYPE_GPU" in rendered_reglib

    def test_contains_dsp_core_type(self, rendered_reglib: str) -> None:
        assert "SNPE_UDO_CORETYPE_DSP" in rendered_reglib

    def test_contains_impl_lib_cpu_name(self, rendered_reglib: str) -> None:
        assert "libSoftmaxUdoPackageImplCpu.so" in rendered_reglib

    def test_contains_impl_lib_gpu_name(self, rendered_reglib: str) -> None:
        assert "libSoftmaxUdoPackageImplGpu.so" in rendered_reglib

    def test_contains_udo_base_header(self, rendered_reglib: str) -> None:
        assert "SnpeUdo/UdoBase.h" in rendered_reglib

    def test_contains_version_major(self, rendered_reglib: str) -> None:
        assert "SNPE_UDO_API_VERSION_MAJOR" in rendered_reglib

    def test_validates_op_type(self, rendered_reglib: str) -> None:
        assert "SNPE_UDO_WRONG_OPERATION" in rendered_reglib

    def test_validates_input_count(self, rendered_reglib: str) -> None:
        assert "SNPE_UDO_WRONG_NUMBER_OF_INPUTS" in rendered_reglib


# ---------------------------------------------------------------------------
# Template rendering — Makefile
# ---------------------------------------------------------------------------


class TestMakefileTemplate:
    """Tests that the Makefile template has all expected targets."""

    @pytest.fixture()
    def rendered_makefile(self, templates_dir: Path) -> str:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(templates_dir)))
        template = env.get_template("makefile.j2")
        return template.render(
            package_name="SoftmaxUdoPackage",
            supported_runtimes=["CPU", "GPU", "DSP_V68"],
            dsp_arch="v68",
        )

    def test_has_all_target(self, rendered_makefile: str) -> None:
        assert "all:" in rendered_makefile

    def test_has_cpu_x86_target(self, rendered_makefile: str) -> None:
        assert "cpu_x86:" in rendered_makefile or "cpu_x86 " in rendered_makefile

    def test_has_cpu_android_target(self, rendered_makefile: str) -> None:
        assert "cpu_android:" in rendered_makefile or "cpu_android " in rendered_makefile

    def test_has_gpu_android_target(self, rendered_makefile: str) -> None:
        assert "gpu_android:" in rendered_makefile or "gpu_android " in rendered_makefile

    def test_has_dsp_target(self, rendered_makefile: str) -> None:
        assert "dsp:" in rendered_makefile or "\ndsp " in rendered_makefile

    def test_has_dsp_x86_target(self, rendered_makefile: str) -> None:
        assert "dsp_x86:" in rendered_makefile or "dsp_x86 " in rendered_makefile

    def test_has_dsp_aarch64_target(self, rendered_makefile: str) -> None:
        assert "dsp_aarch64:" in rendered_makefile or "dsp_aarch64 " in rendered_makefile

    def test_has_reg_target(self, rendered_makefile: str) -> None:
        assert "reg:" in rendered_makefile or "\nreg " in rendered_makefile

    def test_has_reg_x86_target(self, rendered_makefile: str) -> None:
        assert "reg_x86:" in rendered_makefile or "reg_x86 " in rendered_makefile

    def test_has_reg_android_target(self, rendered_makefile: str) -> None:
        assert "reg_android:" in rendered_makefile or "reg_android " in rendered_makefile

    def test_has_clean_target(self, rendered_makefile: str) -> None:
        assert "clean:" in rendered_makefile

    def test_contains_package_name(self, rendered_makefile: str) -> None:
        assert "SoftmaxUdoPackage" in rendered_makefile

    def test_contains_dsp_arch(self, rendered_makefile: str) -> None:
        assert "v68" in rendered_makefile

    def test_references_android_ndk(self, rendered_makefile: str) -> None:
        assert "ANDROID_NDK_ROOT" in rendered_makefile

    def test_references_hexagon_sdk(self, rendered_makefile: str) -> None:
        assert "HEXAGON_SDK_ROOT" in rendered_makefile

    def test_references_cl_library_path(self, rendered_makefile: str) -> None:
        assert "CL_LIBRARY_PATH" in rendered_makefile

    def test_phony_targets_declared(self, rendered_makefile: str) -> None:
        assert ".PHONY:" in rendered_makefile
