"""UDO Configuration Schema — full OpDef model for both JSON and XML configs.

Based on SNPE "Defining a UDO" documentation. Supports:
- JSON UDO Config format (UdoPackage_0 / Operators / inputs/outputs/params)
- XML OpDef Config format (OpDefCollection / OpDefList / SupplementalOpDefList)

Key concepts:
- UDO Package = registration library + N implementation libraries (per-runtime)
- Registration runs on ARM CPU; implementation runs on target HW (GPU/DSP/HTP)
- Package naming: actual name = PackageName + Backend (e.g. "MyPackageHtp")
- DSP V68+ UDOs must be recompiled for each SDK release (backward compat limitation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# Data Types
# ══════════════════════════════════════════════════════════════════════════════

class UDODataType(str, Enum):
    """Supported tensor data types for UDO operations."""
    FLOAT_16 = "FLOAT_16"
    FLOAT_32 = "FLOAT_32"
    FIXED_4 = "FIXED_4"
    FIXED_8 = "FIXED_8"
    FIXED_16 = "FIXED_16"
    UINT_8 = "UINT_8"
    UINT_16 = "UINT_16"
    UINT_32 = "UINT_32"
    INT_32 = "INT_32"
    STRING = "STRING"
    BACKEND_SPECIFIC = "BACKEND_SPECIFIC"


class TensorLayout(str, Enum):
    """Tensor data layout conventions."""
    NHWC = "NHWC"       # Batch x Height x Width x Channel (SNPE default)
    NCHW = "NCHW"       # Batch x Channel x Height x Width (PyTorch default)
    UNDEFINED = "UNDEFINED"
    BACKEND_SPECIFIC = "BACKEND_SPECIFIC"


class TensorRank(str, Enum):
    """Tensor rank/dimensionality."""
    SCALAR = "SCALAR"
    D1 = "1D"           # Vector
    D2 = "2D"           # Matrix
    D3 = "3D"           # Image
    D4 = "4D"           # Batched Image
    ND = "ND"           # Generic N-D (N >= 0)


class UDOBackend(str, Enum):
    """Supported UDO backends (hardware targets)."""
    CPU = "CPU"
    GPU = "GPU"
    DSP = "DSP"
    DSP_V65 = "DSP_V65"
    DSP_V66 = "DSP_V66"
    DSP_V68 = "DSP_V68"
    DSP_V69 = "DSP_V69"
    DSP_V73 = "DSP_V73"
    HTP = "HTP"
    AIP = "AIP"


# ══════════════════════════════════════════════════════════════════════════════
# Tensor Definitions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TensorShape:
    """Shape specification for a UDO tensor."""
    rank: TensorRank = TensorRank.ND
    layout: TensorLayout = TensorLayout.NHWC
    text: str = ""  # Optional description


@dataclass
class TensorConstraint:
    """Constraint on a tensor (descriptive, not enforced)."""
    id: str = ""
    type: str = ""  # Number, Shape, Value, Datatype, Description
    description: str = ""


@dataclass
class UDOTensor:
    """Base tensor definition for inputs, outputs, and parameters."""
    name: str
    datatype: UDODataType = UDODataType.FLOAT_32
    per_core_datatypes: dict[str, UDODataType] = field(default_factory=dict)
    shape: TensorShape = field(default_factory=TensorShape)
    mandatory: bool = True
    is_static: bool = False         # True = contains weights/biases from model
    repeated: bool = False          # True = variadic (e.g. Concat inputs)
    default: str = ""               # Default value (for params)
    constraints: list[TensorConstraint] = field(default_factory=list)

    def get_datatype_for_backend(self, backend: str) -> UDODataType:
        """Get the data type for a specific backend.

        Falls back to the generic datatype if no per-core override exists.
        """
        if self.per_core_datatypes:
            return self.per_core_datatypes.get(backend, self.datatype)
        return self.datatype


@dataclass
class UDOParameter(UDOTensor):
    """Scalar or tensor-valued parameter for a UDO operation."""
    enumeration: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Operation Definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UDOOpDef:
    """Complete definition of a User-Defined Operation.

    Corresponds to <OpDef> in XML or an entry in "Operators" in JSON.
    """
    name: str                           # Operation type name
    inputs: list[UDOTensor] = field(default_factory=list)
    outputs: list[UDOTensor] = field(default_factory=list)
    scalar_params: list[UDOParameter] = field(default_factory=list)
    tensor_params: list[UDOParameter] = field(default_factory=list)
    supported_backends: list[UDOBackend] = field(default_factory=list)
    dsp_arch_types: list[str] = field(default_factory=list)  # ["v66", "v68", "v73"]
    description: str = ""
    use_default_translation: bool = False  # True = overrides QNN native op

    @property
    def all_params(self) -> list[UDOParameter]:
        return self.scalar_params + self.tensor_params

    def supports_backend(self, backend: str) -> bool:
        """Check if this op supports a given backend (case-insensitive)."""
        backend_upper = backend.upper()
        return any(b.value.upper() == backend_upper for b in self.supported_backends)


@dataclass
class SupplementalOpDef:
    """Per-backend overrides for an OpDef (from XML SupplementalOpDef)."""
    name: str
    backend: UDOBackend
    inputs: list[UDOTensor] = field(default_factory=list)
    outputs: list[UDOTensor] = field(default_factory=list)
    parameters: list[UDOParameter] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Package Definition (Top-Level)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UDOPackageDefinition:
    """Full UDO package definition — parsed from JSON or XML config.

    Corresponds to OpDefCollection (XML) or UdoPackage_0 (JSON).

    Package naming rule: actual lib name = PackageName + Backend
    e.g. PackageName="MyOps" → "MyOpsHtp", "MyOpsCpu", etc.
    """
    package_name: str
    domain: str = ""
    version: str = "1.0"
    operators: list[UDOOpDef] = field(default_factory=list)
    supplemental_defs: list[SupplementalOpDef] = field(default_factory=list)

    @property
    def all_backends(self) -> set[str]:
        """All backends referenced by any operator."""
        backends: set[str] = set()
        for op in self.operators:
            for b in op.supported_backends:
                backends.add(b.value)
        return backends

    def get_actual_package_name(self, backend: str) -> str:
        """Get the actual library package name for a backend.

        Rule from docs: actual name = PackageName + Backend
        e.g. "SoftmaxUdo" → "SoftmaxUdoHtp" for HTP backend.
        """
        backend_suffix = backend.replace("DSP_", "Dsp").replace("_", "").title()
        if backend.upper() == "HTP":
            backend_suffix = "Htp"
        elif backend.upper() == "CPU":
            backend_suffix = "Cpu"
        elif backend.upper() == "GPU":
            backend_suffix = "Gpu"
        elif "DSP" in backend.upper():
            backend_suffix = "Dsp"
        return f"{self.package_name}{backend_suffix}"

    def get_ops_for_backend(self, backend: str) -> list[UDOOpDef]:
        """Get all operations that support a given backend."""
        return [op for op in self.operators if op.supports_backend(backend)]


# ══════════════════════════════════════════════════════════════════════════════
# Parser: JSON Config
# ══════════════════════════════════════════════════════════════════════════════

def parse_json_config(config: dict[str, Any]) -> UDOPackageDefinition:
    """Parse a JSON UDO config dict into a UDOPackageDefinition.

    Handles both formats:
    - New: {"UdoPackage_0": {"Operators": [...], "UDO_PACKAGE_NAME": "..."}}
    - Legacy: {"UDO_PACKAGE_NAME": "...", "UDO_OPS": [...]}
    """
    # Find the package section (may be keyed as "UdoPackage_0", "UdoPackage", etc.)
    pkg_data: dict[str, Any] = {}
    for key in config:
        if key.startswith("UdoPackage") or key == "UDO_PACKAGE_NAME":
            pkg_data = config[key] if isinstance(config[key], dict) else config
            break
    if not pkg_data:
        pkg_data = config

    package_name = pkg_data.get("UDO_PACKAGE_NAME", "UDOPackage")
    operators_data = pkg_data.get("Operators", pkg_data.get("UDO_OPS", []))

    operators: list[UDOOpDef] = []
    for op_data in operators_data:
        op = UDOOpDef(
            name=op_data.get("type", op_data.get("UDO_OP_NAME", "")),
        )

        # Parse inputs
        for inp in op_data.get("inputs", op_data.get("UDO_INPUT_TENSORS", [])):
            tensor = UDOTensor(
                name=inp.get("name", ""),
                is_static=inp.get("static", False),
            )
            if "per_core_data_types" in inp:
                for core, dt in inp["per_core_data_types"].items():
                    tensor.per_core_datatypes[core] = UDODataType(dt)
            elif "data_type" in inp:
                tensor.datatype = UDODataType(inp["data_type"])
            elif "dtype" in inp:
                tensor.datatype = UDODataType(inp["dtype"])
            if "tensor_layout" in inp:
                tensor.shape.layout = TensorLayout(inp["tensor_layout"])
            op.inputs.append(tensor)

        # Parse outputs
        for out in op_data.get("outputs", op_data.get("UDO_OUTPUT_TENSORS", [])):
            tensor = UDOTensor(name=out.get("name", ""))
            if "per_core_data_types" in out:
                for core, dt in out["per_core_data_types"].items():
                    tensor.per_core_datatypes[core] = UDODataType(dt)
            elif "data_type" in out:
                tensor.datatype = UDODataType(out["data_type"])
            elif "dtype" in out:
                tensor.datatype = UDODataType(out["dtype"])
            op.outputs.append(tensor)

        # Parse scalar params
        for param in op_data.get("scalar_params", op_data.get("UDO_PARAM_TYPES", [])):
            p = UDOParameter(
                name=param.get("name", ""),
                datatype=UDODataType(param.get("data_type", "INT_32")),
            )
            op.scalar_params.append(p)

        # Parse tensor params
        for param in op_data.get("tensor_params", []):
            p = UDOParameter(
                name=param.get("name", ""),
                datatype=UDODataType(param.get("data_type", "FLOAT_32")),
            )
            if "tensor_layout" in param:
                p.shape.layout = TensorLayout(param["tensor_layout"])
            op.tensor_params.append(p)

        # Parse core types
        core_types = op_data.get("core_types", op_data.get("UDO_CORE_TYPES", []))
        for ct in core_types:
            try:
                op.supported_backends.append(UDOBackend(ct.upper()))
            except ValueError:
                op.supported_backends.append(UDOBackend.CPU)

        # Parse DSP arch types
        op.dsp_arch_types = op_data.get("dsp_arch_types", [])

        operators.append(op)

    return UDOPackageDefinition(
        package_name=package_name,
        operators=operators,
    )


def parse_json_file(config_path: str) -> UDOPackageDefinition:
    """Parse a JSON UDO config file."""
    import json
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        # Mock mode: return synthetic definition from filename
        return UDOPackageDefinition(
            package_name=path.stem.replace("_", "") + "Package",
            operators=[UDOOpDef(
                name=path.stem.split("_")[0],
                inputs=[UDOTensor(name="input")],
                outputs=[UDOTensor(name="output")],
                supported_backends=[UDOBackend.CPU, UDOBackend.GPU, UDOBackend.HTP],
            )],
        )

    with open(config_path) as f:
        data = json.load(f)
    return parse_json_config(data)


# ══════════════════════════════════════════════════════════════════════════════
# Backward Compatibility Warnings
# ══════════════════════════════════════════════════════════════════════════════

def get_compat_warnings(pkg_def: UDOPackageDefinition) -> list[str]:
    """Check for UDO backward compatibility issues.

    Key rule from docs: DSP V68+ UDOs compiled for one SDK release
    cannot be used with a different release. Must recompile.
    """
    warnings: list[str] = []

    for op in pkg_def.operators:
        if any(arch in ("v68", "v69", "v73") for arch in op.dsp_arch_types):
            warnings.append(
                f"Op '{op.name}': Uses DSP V68+ architectures ({op.dsp_arch_types}). "
                "UDO libraries compiled for DSP V68+ are NOT backward compatible "
                "across SDK releases. Must recompile with the correct QAIRT SDK version."
            )
            break

    # Check for operations with unknown number of inputs (not supported on DSP/HTP)
    for op in pkg_def.operators:
        if any(t.repeated for t in op.inputs):
            if op.supports_backend("DSP") or op.supports_backend("HTP"):
                warnings.append(
                    f"Op '{op.name}': Has repeated (variadic) inputs. "
                    "Operations with unknown number of inputs/outputs are NOT "
                    "supported on DSP and HTP backends."
                )

    return warnings
