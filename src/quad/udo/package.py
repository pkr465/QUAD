"""UDO Package — dataclass models for User-Defined Operation packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class UDORuntime(str, Enum):
    """Supported runtimes for UDO execution."""

    CPU = "cpu"
    GPU = "gpu"
    DSP_V65 = "dsp_v65"
    DSP_V66 = "dsp_v66"
    DSP_V68 = "dsp_v68"
    HTP = "htp"
    AIP = "aip"


@dataclass
class UDOConfig:
    """Parsed representation of a UDO JSON configuration file."""

    config_path: str
    op_name: str
    package_name: str
    supported_runtimes: list[str]

    @classmethod
    def from_file(cls, config_path: str) -> "UDOConfig":
        """Load and parse a UDO JSON config file.

        In mock mode the file need not exist on disk; a synthetic config
        is returned based on the filename stem.
        """
        import json
        import os

        path = Path(config_path)

        if os.path.exists(config_path):
            with open(config_path) as fh:
                raw = json.load(fh)
            op_name = raw.get("UdoPackage", {}).get("Operators", [{}])[0].get("type", path.stem)
            package_name = raw.get("UdoPackage", {}).get("PackageName", path.stem + "Package")
            runtimes: list[str] = []
            for op in raw.get("UdoPackage", {}).get("Operators", []):
                for rt in op.get("runtimes", []):
                    name = rt.get("runtime", "").lower()
                    if name and name not in runtimes:
                        runtimes.append(name)
        else:
            # Mock: derive sensible defaults from the filename
            stem = path.stem  # e.g. "Softmax_Htp" or "SoftmaxHtp"
            parts = stem.replace("_", " ").split()
            op_name = parts[0] if parts else stem
            package_name = op_name + "UdoPackage"
            runtimes = ["cpu"]
            lower = stem.lower()
            if "htp" in lower:
                runtimes = ["cpu", "htp"]
            elif "gpu" in lower:
                runtimes = ["cpu", "gpu"]
            elif "dsp" in lower:
                runtimes = ["cpu", "dsp_v68"]

        return cls(
            config_path=config_path,
            op_name=op_name,
            package_name=package_name,
            supported_runtimes=runtimes,
        )


@dataclass
class UDOPackage:
    """Represents a generated (and optionally compiled) UDO package.

    Attributes:
        name: Package name, e.g. ``"SoftmaxUdoPackage"``.
        package_dir: Absolute path to the generated package directory.
        config_json: Source UDO JSON configuration file path.
        supported_runtimes: List of runtime strings the package supports.
        libs: Mapping of runtime → list of compiled library paths.
        is_compiled: ``True`` once :meth:`UDOManager.compile_package` succeeds.
    """

    name: str
    package_dir: str
    config_json: str
    supported_runtimes: list[str]
    libs: dict[str, list[str]] = field(default_factory=dict)
    is_compiled: bool = False

    # ------------------------------------------------------------------
    # Library helpers
    # ------------------------------------------------------------------

    def get_reg_lib(self, runtime: str, arch: str = "arm64-v8a") -> str:
        """Return the registration library path for *runtime* / *arch*.

        The registration library follows the SNPE convention::

            <package_dir>/libs/<arch>/lib<name>Reg.so

        If compiled libraries are recorded in :attr:`libs`, the first match
        is returned; otherwise the conventional path is synthesised.

        Args:
            runtime: One of the :class:`UDORuntime` string values.
            arch: Android ABI or host architecture string.

        Returns:
            Absolute path to the registration ``.so`` file.
        """
        rt_key = runtime.lower()
        if rt_key in self.libs:
            for lib in self.libs[rt_key]:
                if "Reg" in Path(lib).name or "reg" in Path(lib).name.lower():
                    return lib

        # Fall back to conventional path
        lib_name = f"lib{self.name}Reg.so"
        return str(Path(self.package_dir) / "libs" / arch / lib_name)

    def get_impl_lib(self, runtime: str, arch: str = "arm64-v8a") -> str:
        """Return the implementation library path for *runtime* / *arch*.

        The implementation library follows the SNPE convention::

            <package_dir>/libs/<arch>/lib<name>Impl<Runtime>.so

        Args:
            runtime: One of the :class:`UDORuntime` string values.
            arch: Android ABI or host architecture string.

        Returns:
            Absolute path to the implementation ``.so`` file.
        """
        rt_key = runtime.lower()
        if rt_key in self.libs:
            for lib in self.libs[rt_key]:
                if "Impl" in Path(lib).name or "impl" in Path(lib).name.lower():
                    return lib

        # Fall back to conventional path
        runtime_tag = runtime.upper().replace("_", "")
        lib_name = f"lib{self.name}Impl{runtime_tag}.so"
        return str(Path(self.package_dir) / "libs" / arch / lib_name)
