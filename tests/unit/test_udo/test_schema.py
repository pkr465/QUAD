"""Tests for UDO schema — OpDef, config parsing, backward compatibility."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from quad.udo.schema import (
    UDOBackend,
    UDODataType,
    UDOOpDef,
    UDOPackageDefinition,
    UDOParameter,
    UDOTensor,
    SupplementalOpDef,
    TensorLayout,
    TensorRank,
    TensorShape,
    get_compat_warnings,
    parse_json_config,
    parse_json_file,
)


class TestUDODataType:
    def test_all_types_exist(self) -> None:
        assert UDODataType.FLOAT_16
        assert UDODataType.FLOAT_32
        assert UDODataType.FIXED_4
        assert UDODataType.FIXED_8
        assert UDODataType.FIXED_16
        assert UDODataType.UINT_8
        assert UDODataType.UINT_16
        assert UDODataType.UINT_32
        assert UDODataType.STRING
        assert UDODataType.BACKEND_SPECIFIC


class TestTensorLayout:
    def test_nhwc_is_default(self) -> None:
        t = UDOTensor(name="test")
        assert t.shape.layout == TensorLayout.NHWC

    def test_all_layouts(self) -> None:
        for layout in TensorLayout:
            assert layout.value in ("NHWC", "NCHW", "UNDEFINED", "BACKEND_SPECIFIC")


class TestTensorRank:
    def test_all_ranks(self) -> None:
        assert TensorRank.SCALAR.value == "SCALAR"
        assert TensorRank.D4.value == "4D"
        assert TensorRank.ND.value == "ND"


class TestUDOBackend:
    def test_all_backends(self) -> None:
        assert UDOBackend.CPU.value == "CPU"
        assert UDOBackend.GPU.value == "GPU"
        assert UDOBackend.DSP_V68.value == "DSP_V68"
        assert UDOBackend.HTP.value == "HTP"


class TestUDOTensor:
    def test_per_core_datatype_lookup(self) -> None:
        t = UDOTensor(
            name="input",
            datatype=UDODataType.FLOAT_32,
            per_core_datatypes={"CPU": UDODataType.FLOAT_32, "DSP": UDODataType.UINT_8},
        )
        assert t.get_datatype_for_backend("CPU") == UDODataType.FLOAT_32
        assert t.get_datatype_for_backend("DSP") == UDODataType.UINT_8
        assert t.get_datatype_for_backend("GPU") == UDODataType.FLOAT_32  # fallback

    def test_static_tensor(self) -> None:
        t = UDOTensor(name="weights", is_static=True)
        assert t.is_static is True

    def test_repeated_for_variadic(self) -> None:
        t = UDOTensor(name="concat_input", repeated=True)
        assert t.repeated is True


class TestUDOOpDef:
    def test_supports_backend(self) -> None:
        op = UDOOpDef(
            name="Softmax",
            supported_backends=[UDOBackend.CPU, UDOBackend.GPU, UDOBackend.HTP],
        )
        assert op.supports_backend("CPU") is True
        assert op.supports_backend("HTP") is True
        assert op.supports_backend("DSP_V66") is False

    def test_all_params_combines_scalar_and_tensor(self) -> None:
        op = UDOOpDef(
            name="MyOp",
            scalar_params=[UDOParameter(name="alpha")],
            tensor_params=[UDOParameter(name="weights")],
        )
        assert len(op.all_params) == 2


class TestUDOPackageDefinition:
    def test_package_naming_rule(self) -> None:
        """Actual name = PackageName + Backend suffix."""
        pkg = UDOPackageDefinition(package_name="SoftmaxUdo")
        assert pkg.get_actual_package_name("HTP") == "SoftmaxUdoHtp"
        assert pkg.get_actual_package_name("CPU") == "SoftmaxUdoCpu"
        assert pkg.get_actual_package_name("GPU") == "SoftmaxUdoGpu"
        assert pkg.get_actual_package_name("DSP") == "SoftmaxUdoDsp"

    def test_all_backends(self) -> None:
        pkg = UDOPackageDefinition(
            package_name="Test",
            operators=[UDOOpDef(
                name="Op1",
                supported_backends=[UDOBackend.CPU, UDOBackend.HTP],
            )],
        )
        assert pkg.all_backends == {"CPU", "HTP"}

    def test_get_ops_for_backend(self) -> None:
        pkg = UDOPackageDefinition(
            package_name="Test",
            operators=[
                UDOOpDef(name="Op1", supported_backends=[UDOBackend.CPU, UDOBackend.HTP]),
                UDOOpDef(name="Op2", supported_backends=[UDOBackend.CPU]),
            ],
        )
        htp_ops = pkg.get_ops_for_backend("HTP")
        assert len(htp_ops) == 1
        assert htp_ops[0].name == "Op1"

        cpu_ops = pkg.get_ops_for_backend("CPU")
        assert len(cpu_ops) == 2


class TestParseJSONConfig:
    def test_parse_softmax_config(self) -> None:
        config = {
            "UdoPackage_0": {
                "Operators": [{
                    "type": "Softmax",
                    "inputs": [{"name": "input", "data_type": "FLOAT_32"}],
                    "outputs": [{"name": "output", "data_type": "FLOAT_32"}],
                    "core_types": ["CPU", "GPU", "DSP"],
                    "dsp_arch_types": ["v68"],
                }],
                "UDO_PACKAGE_NAME": "SoftmaxUdoPackage",
            }
        }
        pkg = parse_json_config(config)
        assert pkg.package_name == "SoftmaxUdoPackage"
        assert len(pkg.operators) == 1
        assert pkg.operators[0].name == "Softmax"
        assert UDOBackend.CPU in pkg.operators[0].supported_backends
        assert pkg.operators[0].dsp_arch_types == ["v68"]

    def test_parse_per_core_datatypes(self) -> None:
        config = {
            "UdoPackage_0": {
                "Operators": [{
                    "type": "MyOp",
                    "inputs": [{
                        "name": "x",
                        "per_core_data_types": {"CPU": "FLOAT_32", "DSP": "UINT_8"},
                    }],
                    "outputs": [{"name": "y", "data_type": "FLOAT_32"}],
                    "core_types": ["CPU", "DSP"],
                }],
                "UDO_PACKAGE_NAME": "MyPackage",
            }
        }
        pkg = parse_json_config(config)
        inp = pkg.operators[0].inputs[0]
        assert inp.get_datatype_for_backend("CPU") == UDODataType.FLOAT_32
        assert inp.get_datatype_for_backend("DSP") == UDODataType.UINT_8

    def test_parse_static_tensor(self) -> None:
        config = {
            "UdoPackage_0": {
                "Operators": [{
                    "type": "Conv",
                    "inputs": [
                        {"name": "input", "data_type": "FLOAT_32"},
                        {"name": "weights", "data_type": "FLOAT_32", "static": True},
                    ],
                    "outputs": [{"name": "output", "data_type": "FLOAT_32"}],
                    "core_types": ["CPU"],
                }],
                "UDO_PACKAGE_NAME": "ConvPackage",
            }
        }
        pkg = parse_json_config(config)
        assert pkg.operators[0].inputs[1].is_static is True

    def test_parse_scalar_params(self) -> None:
        config = {
            "UdoPackage_0": {
                "Operators": [{
                    "type": "Op",
                    "inputs": [{"name": "x", "data_type": "FLOAT_32"}],
                    "outputs": [{"name": "y", "data_type": "FLOAT_32"}],
                    "scalar_params": [{"name": "axis", "data_type": "INT_32"}],
                    "core_types": ["CPU"],
                }],
                "UDO_PACKAGE_NAME": "Pkg",
            }
        }
        pkg = parse_json_config(config)
        assert len(pkg.operators[0].scalar_params) == 1
        assert pkg.operators[0].scalar_params[0].name == "axis"

    def test_parse_json_file_mock(self) -> None:
        """Non-existent file returns synthetic config from filename."""
        pkg = parse_json_file("/nonexistent/Softmax_Htp.json")
        assert "Softmax" in pkg.operators[0].name or "Softmax" in pkg.package_name

    def test_parse_json_file_real(self) -> None:
        """Test with actual file on disk."""
        config = {
            "UdoPackage_0": {
                "Operators": [{"type": "ReLU", "inputs": [{"name": "x", "data_type": "FLOAT_32"}],
                               "outputs": [{"name": "y", "data_type": "FLOAT_32"}], "core_types": ["CPU"]}],
                "UDO_PACKAGE_NAME": "ReluPkg",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            pkg = parse_json_file(f.name)
        assert pkg.package_name == "ReluPkg"
        assert pkg.operators[0].name == "ReLU"


class TestCompatWarnings:
    def test_dsp_v68_backward_compat_warning(self) -> None:
        """DSP V68+ UDOs must be recompiled for each SDK release."""
        pkg = UDOPackageDefinition(
            package_name="Test",
            operators=[UDOOpDef(
                name="Op1",
                dsp_arch_types=["v68", "v73"],
                supported_backends=[UDOBackend.HTP],
            )],
        )
        warnings = get_compat_warnings(pkg)
        assert len(warnings) >= 1
        assert "recompile" in warnings[0].lower() or "NOT backward compatible" in warnings[0]

    def test_repeated_inputs_on_dsp_warning(self) -> None:
        """Variadic inputs not supported on DSP/HTP."""
        pkg = UDOPackageDefinition(
            package_name="Test",
            operators=[UDOOpDef(
                name="Concat",
                inputs=[UDOTensor(name="input", repeated=True)],
                supported_backends=[UDOBackend.HTP],
            )],
        )
        warnings = get_compat_warnings(pkg)
        assert any("variadic" in w.lower() or "repeated" in w.lower() for w in warnings)

    def test_no_warnings_for_cpu_only(self) -> None:
        pkg = UDOPackageDefinition(
            package_name="Test",
            operators=[UDOOpDef(
                name="Op1",
                supported_backends=[UDOBackend.CPU],
            )],
        )
        warnings = get_compat_warnings(pkg)
        assert warnings == []
