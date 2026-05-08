"""UDO Manager — orchestrates the full SNPE UDO workflow.

Supports two execution modes:

* **Mock mode** (default when ``QAIRT_SDK_ROOT`` is unset): all operations
  are simulated and return realistic paths / output without touching the
  filesystem or running any subprocess.
* **Real mode** (when ``QAIRT_SDK_ROOT`` is set): invokes the actual SNPE
  SDK CLI tools (``snpe-udo-package-generator``, ``snpe-dlc-quant``, …) via
  :mod:`subprocess`.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from quad.udo.package import UDOConfig, UDOPackage
from quad.udo.schema import parse_json_file, get_compat_warnings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime → make-target mappings used by snpe-udo-package-generator
# ---------------------------------------------------------------------------
_RUNTIME_MAKE_TARGETS: dict[str, str] = {
    "cpu_x86": "cpu_x86",
    "cpu_android": "cpu_android",
    "dsp_v65": "dsp",      # V65/V66 use generic "dsp" target
    "dsp_v66": "dsp",
    "dsp_v68": "dsp",      # V68+ all use "dsp" target (SDK handles version)
    "dsp_v69": "dsp",
    "dsp_v73": "dsp",
    "dsp_x86": "dsp_x86",  # For offline cache generation (x86 host)
    "dsp_aarch64": "dsp_aarch64",  # For Linux ARM devices
    "gpu_android": "gpu_android",
}

# Canonical library architectures for each runtime target
_RUNTIME_ARCH: dict[str, str] = {
    "cpu_x86": "x86-64_linux_clang",
    "cpu_android": "arm64-v8a",
    "dsp_v65": "dsp",       # V65/V66 output to dsp_v60/ folder
    "dsp_v66": "dsp",
    "dsp_v68": "dsp_v68",   # V68+ output to dsp_v68/ folder
    "dsp_v69": "dsp_v68",
    "dsp_v73": "dsp_v68",
    "dsp_x86": "x86-64_linux_clang",  # Offline cache gen
    "dsp_aarch64": "arm64-v8a",
    "gpu_android": "arm64-v8a",
}


class UDOManager:
    """High-level manager for SNPE User-Defined Operations.

    Wraps the full UDO workflow:

    1. :meth:`generate_package` — run ``snpe-udo-package-generator`` from a
       JSON config to scaffold the C++ package skeleton.
    2. :meth:`convert_model` — run ``snpe-onnx-to-dlc`` (or the appropriate
       converter) to produce a ``.dlc`` that references UDO operators.
    3. :meth:`compile_package` — invoke the package ``Makefile`` to build
       shared libraries for the chosen target(s).
    4. :meth:`quantize_with_udo` — run ``snpe-dlc-quant`` with UDO
       registration libraries loaded.
    5. :meth:`deploy_to_android` — push all assets via ``adb push``.
    6. :meth:`execute_on_android` — run ``snpe-net-run`` on-device.

    Args:
        sdk_root: Path to the QAIRT/SNPE SDK root.  Defaults to the
            ``QAIRT_SDK_ROOT`` environment variable.  When *neither* is set
            the manager operates in **mock mode**.
    """

    def __init__(self, sdk_root: str | None = None) -> None:
        self._sdk_root: str | None = sdk_root or os.environ.get("QAIRT_SDK_ROOT")
        self._mock = self._sdk_root is None
        if self._mock:
            logger.debug("UDOManager: QAIRT_SDK_ROOT not set — running in mock mode")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def is_mock(self) -> bool:
        """``True`` when the manager is operating without a real SDK."""
        return self._mock

    def _sdk_bin(self, tool: str) -> str:
        """Return absolute path to an SDK CLI tool.

        Uses POSIX path separators so the result is identical on Windows
        and Linux — the SDK CLI tools that consume these paths
        ultimately run on POSIX targets (Linux / Android).
        """
        if self._sdk_root is None:
            return tool  # unreachable in normal mock flow
        return (Path(self._sdk_root) / "bin" / tool).as_posix()

    def _run(self, cmd: list[str], cwd: str | None = None) -> str:
        """Execute *cmd*, return combined stdout+stderr."""
        logger.debug("UDOManager._run: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"Command failed (exit {result.returncode}):\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  output: {output.strip()}"
            )
        return output

    # ------------------------------------------------------------------
    # Step 1 — generate_package
    # ------------------------------------------------------------------

    def generate_package(
        self,
        config_json: str,
        output_dir: str,
        gen_cmakelists: bool = False,
    ) -> UDOPackage:
        """Scaffold a UDO C++ package from a JSON config.

        Wraps ``snpe-udo-package-generator -p <config_json> -o <output_dir>``.

        Args:
            config_json: Path to the UDO JSON configuration file (e.g.
                ``Softmax_Htp.json``).
            output_dir: Directory where the package skeleton is written.
            gen_cmakelists: Pass ``--gen_cmakelists`` (required on Windows).

        Returns:
            :class:`~quad.udo.package.UDOPackage` describing the generated
            package.
        """
        cfg = UDOConfig.from_file(config_json)
        package_dir = str(Path(output_dir) / cfg.package_name)

        if self._mock:
            logger.debug(
                "UDOManager.generate_package [mock]: config=%s → %s",
                config_json,
                package_dir,
            )
            return UDOPackage(
                name=cfg.package_name,
                package_dir=package_dir,
                config_json=config_json,
                supported_runtimes=cfg.supported_runtimes,
            )

        # Real mode -------------------------------------------------------
        cmd = [
            self._sdk_bin("snpe-udo-package-generator"),
            "-p", config_json,
            "-o", output_dir,
        ]
        if gen_cmakelists:
            cmd.append("--gen_cmakelists")
        self._run(cmd)

        return UDOPackage(
            name=cfg.package_name,
            package_dir=package_dir,
            config_json=config_json,
            supported_runtimes=cfg.supported_runtimes,
        )

    # ------------------------------------------------------------------
    # Step 2 — convert_model
    # ------------------------------------------------------------------

    def convert_model(
        self,
        model_path: str,
        output_dlc: str,
        udo_config: str,
        source_format: str = "onnx",
    ) -> str:
        """Convert a source model to ``.dlc`` with UDO operator references.

        Wraps ``snpe-onnx-to-dlc`` (or ``snpe-tensorflow-to-dlc``, etc.)
        with ``--udo_config_paths <udo_config>``.

        Args:
            model_path: Path to the source model (``.onnx``, ``.pb``, …).
            output_dlc: Desired output ``.dlc`` path.
            udo_config: Path to the UDO JSON config that describes the custom
                operators present in the model.
            source_format: Source model format — ``"onnx"``, ``"tensorflow"``,
                ``"tflite"``, or ``"pytorch"``.

        Returns:
            Absolute path to the generated ``.dlc`` file.
        """
        fmt = source_format.lower()
        converter_map: dict[str, str] = {
            "onnx": "snpe-onnx-to-dlc",
            "tensorflow": "snpe-tensorflow-to-dlc",
            "tflite": "snpe-tflite-to-dlc",
            "pytorch": "snpe-pytorch-to-dlc",
        }
        if fmt not in converter_map:
            raise ValueError(
                f"Unsupported source_format '{source_format}'. "
                f"Choose from: {list(converter_map)}"
            )

        if self._mock:
            logger.debug(
                "UDOManager.convert_model [mock]: %s → %s (udo=%s)",
                model_path,
                output_dlc,
                udo_config,
            )
            return str(Path(output_dlc).with_suffix(".dlc"))

        cmd = [
            self._sdk_bin(converter_map[fmt]),
            "--input_network", model_path,
            "--output_path", output_dlc,
            "--udo_config_paths", udo_config,
        ]
        self._run(cmd)
        return output_dlc

    # ------------------------------------------------------------------
    # Step 3 — compile_package
    # ------------------------------------------------------------------

    def compile_package(
        self,
        package_dir: str,
        runtime: str = "cpu",
        android_ndk_root: str = "",
        hexagon_sdk_root: str = "",
    ) -> dict[str, str]:
        """Build shared libraries for the UDO package.

        Invokes the ``Makefile`` inside *package_dir* with the appropriate
        target and environment variables.

        Supported *runtime* values:
        ``"cpu_x86"``, ``"cpu_android"``, ``"dsp_v65"``, ``"dsp_v66"``,
        ``"dsp_v68"``, ``"gpu_android"``.

        Args:
            package_dir: Root of the generated UDO package (e.g.
                ``SoftmaxUdoPackage/``).
            runtime: Build target.
            android_ndk_root: Path to Android NDK (required for
                Android targets).
            hexagon_sdk_root: Path to Hexagon SDK (required for DSP
                targets).

        Returns:
            Mapping of ``{library_basename: absolute_path}`` for every
            ``.so`` produced.
        """
        rt = runtime.lower()
        arch = _RUNTIME_ARCH.get(rt, "arm64-v8a")
        pkg_name = Path(package_dir).name

        if self._mock:
            logger.debug(
                "UDOManager.compile_package [mock]: %s runtime=%s", package_dir, runtime
            )
            libs_dir = Path(package_dir) / "libs" / arch
            reg_lib = str(libs_dir / f"lib{pkg_name}Reg.so")
            runtime_tag = rt.upper().replace("_", "")
            impl_lib = str(libs_dir / f"lib{pkg_name}Impl{runtime_tag}.so")
            return {
                f"lib{pkg_name}Reg.so": reg_lib,
                f"lib{pkg_name}Impl{runtime_tag}.so": impl_lib,
            }

        # Real mode -------------------------------------------------------
        make_target = _RUNTIME_MAKE_TARGETS.get(rt)
        if make_target is None:
            raise ValueError(
                f"Unknown runtime '{runtime}'. "
                f"Choose from: {list(_RUNTIME_MAKE_TARGETS)}"
            )

        env = os.environ.copy()
        if android_ndk_root:
            env["ANDROID_NDK_ROOT"] = android_ndk_root
        if hexagon_sdk_root:
            env["HEXAGON_SDK_ROOT"] = hexagon_sdk_root
        if self._sdk_root:
            env["SNPE_ROOT"] = self._sdk_root

        cmd = ["make", make_target]
        result = subprocess.run(
            cmd,
            cwd=package_dir,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"compile_package make failed:\n{result.stdout}\n{result.stderr}"
            )

        # Collect produced .so files
        libs_dir = Path(package_dir) / "libs" / arch
        produced: dict[str, str] = {}
        if libs_dir.exists():
            for so in libs_dir.glob("*.so"):
                produced[so.name] = str(so)
        return produced

    # ------------------------------------------------------------------
    # Step 4 — quantize_with_udo
    # ------------------------------------------------------------------

    def quantize_with_udo(
        self,
        input_dlc: str,
        output_dlc: str,
        input_list: str,
        reg_lib_path: str,
        enable_htp: bool = False,
        htp_socs: str = "sm8750",
    ) -> str:
        """Quantize a DLC that contains UDO operators.

        Wraps ``snpe-dlc-quant`` with UDO registration library pre-loaded so
        the quantizer knows how to handle custom op types.

        Args:
            input_dlc: Path to the floating-point ``.dlc``.
            output_dlc: Desired output path for the quantized ``.dlc``.
            input_list: Path to a text file listing representative input
                tensors (one per line).
            reg_lib_path: Path to the UDO registration library
                (``libUdo<Pkg>Reg.so``).
            enable_htp: Pass ``--enable_htp`` (required for AIP/HTP
                quantization).
            htp_socs: Comma-separated SoC targets, e.g. ``"sm8350,sm8450"``.

        Returns:
            Absolute path to the quantized ``.dlc``.
        """
        if self._mock:
            logger.debug(
                "UDOManager.quantize_with_udo [mock]: %s → %s", input_dlc, output_dlc
            )
            stem = Path(output_dlc).stem
            suffix = Path(output_dlc).suffix or ".dlc"
            parent = Path(output_dlc).parent
            quantized = str(parent / f"{stem}_quantized{suffix}")
            return quantized

        cmd = [
            self._sdk_bin("snpe-dlc-quant"),
            "--input_dlc", input_dlc,
            "--output_dlc", output_dlc,
            "--input_list", input_list,
            # NOTE: --udo_package_path expects the REGISTRATION LIBRARY PATH
            # (not a directory). The reg lib must be in a location where
            # LD_LIBRARY_PATH can find the corresponding impl libs.
            "--udo_package_path", reg_lib_path,
        ]
        if enable_htp:
            cmd.extend(["--enable_htp", "--htp_socs", htp_socs])

        self._run(cmd)
        return output_dlc

    # ------------------------------------------------------------------
    # Step 5 — deploy_to_android
    # ------------------------------------------------------------------

    def deploy_to_android(
        self,
        package_dir: str,
        model_dlc: str,
        input_list: str,
        runtime: str = "cpu",
        device_dir: str = "/data/local/tmp/snpeexample",
        dsp_arch: str = "hexagon-v68",
    ) -> None:
        """Push UDO libraries and model assets to an Android device via ADB.

        Pushes:

        * SNPE runtime libraries from ``<sdk_root>/lib/aarch64-android/``
        * ``snpe-net-run`` binary
        * UDO shared libraries from the compiled package
        * The ``.dlc`` model file
        * The input data list

        Args:
            package_dir: Root of the compiled UDO package.
            model_dlc: Local path to the model ``.dlc``.
            input_list: Local path to the input list file.
            runtime: Target runtime — ``"cpu"``, ``"gpu"``, ``"dsp"``,
                ``"aip"``.
            device_dir: Destination directory on the Android device.
            dsp_arch: Hexagon architecture string (``"hexagon-v68"``,
                ``"hexagon-v73"``, etc.) used to select the correct DSP
                library subdirectory.
        """
        if self._mock:
            logger.debug(
                "UDOManager.deploy_to_android [mock]: package=%s dlc=%s runtime=%s",
                package_dir,
                model_dlc,
                runtime,
            )
            return

        # Real mode -------------------------------------------------------
        def _adb_push(src: str, dst: str) -> None:
            self._run(["adb", "push", src, dst])

        def _adb_shell(cmd_str: str) -> None:
            self._run(["adb", "shell", cmd_str])

        assert self._sdk_root is not None  # guaranteed in real mode

        # Create device dir
        _adb_shell(f"mkdir -p {device_dir}")

        # Push SNPE runtime libs
        snpe_lib_dir = Path(self._sdk_root) / "lib" / "aarch64-android"
        for lib in snpe_lib_dir.glob("libSNPE*.so"):
            _adb_push(str(lib), device_dir)

        # Push snpe-net-run
        net_run = Path(self._sdk_root) / "bin" / "aarch64-android" / "snpe-net-run"
        if net_run.exists():
            _adb_push(str(net_run), device_dir)

        # Push DSP stub libs if needed
        if runtime in ("dsp", "aip"):
            dsp_lib_dir = Path(self._sdk_root) / "lib" / dsp_arch / "unsigned"
            if dsp_lib_dir.exists():
                for lib in dsp_lib_dir.glob("libSNPE*.so"):
                    _adb_push(str(lib), device_dir)

        # Push UDO libs
        udo_libs_dir = Path(package_dir) / "libs" / "arm64-v8a"
        if udo_libs_dir.exists():
            for lib in udo_libs_dir.glob("*.so"):
                _adb_push(str(lib), device_dir)

        # Push model + input list
        _adb_push(model_dlc, device_dir)
        _adb_push(input_list, device_dir)

    # ------------------------------------------------------------------
    # Step 6 — execute_on_android
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Validation & introspection helpers
    # ------------------------------------------------------------------

    def validate_package_structure(self, package_dir: str) -> dict:
        """Validate that a generated UDO package has the expected directory structure.

        Checks for the canonical files and directories produced by
        ``snpe-udo-package-generator``.  Works in both mock and real mode:
        in mock mode returns a realistic synthetic response; in real mode
        actually walks the filesystem.

        Returns:
            {
                "valid": bool,
                "package_dir": str,
                "found_files": list[str],
                "missing_files": list[str],
                "needs_implementation": list[str],  # Files with "add code here"
            }
        """
        pkg_name = Path(package_dir).name

        # Expected files relative to the package root
        expected_files = [
            "Makefile",
            f"jni/src/reg/{pkg_name}RegLib.cpp",
            f"jni/inc/{pkg_name}ImplLibCpu.h",
            "jni/src/CPU/CpuImpl.cpp",
        ]

        if self._mock:
            # Return realistic mock response — assume package was generated
            # correctly but implementation files need user code.
            found = list(expected_files)
            needs_impl = [
                "jni/src/CPU/CpuImpl.cpp",
                "jni/src/GPU/GpuImpl.cpp",
                "jni/src/DSP/DspImpl.cpp",
            ]
            return {
                "valid": True,
                "package_dir": package_dir,
                "found_files": found + [
                    "jni/src/GPU/GpuImpl.cpp",
                    "jni/src/DSP/DspImpl.cpp",
                    f"jni/inc/{pkg_name}ImplLibGpu.h",
                    f"jni/inc/{pkg_name}ImplLibDsp.h",
                ],
                "missing_files": [],
                "needs_implementation": needs_impl,
            }

        # Real mode — check filesystem
        pkg_path = Path(package_dir)
        found: list[str] = []
        missing: list[str] = []
        needs_impl: list[str] = []

        for rel in expected_files:
            full = pkg_path / rel
            if full.exists():
                found.append(rel)
                # Check if file contains placeholder markers
                try:
                    content = full.read_text(errors="replace")
                    if "add code here" in content.lower() or "TODO" in content:
                        needs_impl.append(rel)
                except OSError:
                    pass
            else:
                missing.append(rel)

        # Also scan for additional source files
        for pattern in ("jni/src/**/*.cpp", "jni/inc/**/*.h"):
            for f in pkg_path.glob(pattern):
                rel_str = str(f.relative_to(pkg_path))
                if rel_str not in found:
                    found.append(rel_str)
                    try:
                        content = f.read_text(errors="replace")
                        if "add code here" in content.lower() or "TODO" in content:
                            needs_impl.append(rel_str)
                    except OSError:
                        pass

        return {
            "valid": len(missing) == 0,
            "package_dir": package_dir,
            "found_files": found,
            "missing_files": missing,
            "needs_implementation": needs_impl,
        }

    def get_implementation_todos(self, package_dir: str) -> list[dict]:
        """List files that need user implementation after package generation.

        The ``snpe-udo-package-generator`` creates skeleton C++ files with
        placeholder comments that the developer must fill in.  This method
        identifies those files and describes what functions need implementation
        per runtime.

        Returns list of:
            {
                "file": str,          # Relative path within package
                "runtime": str,       # CPU, GPU, DSP_V68, etc.
                "functions": list[str],  # Functions to implement
                "notes": str,         # Runtime-specific guidance
            }
        """
        pkg_name = Path(package_dir).name

        if self._mock:
            return [
                {
                    "file": "jni/src/CPU/CpuImpl.cpp",
                    "runtime": "CPU",
                    "functions": [
                        "SnpeUdo_validateOp",
                        "SnpeUdo_executeOp",
                        "SnpeUdo_terminateOp",
                    ],
                    "notes": (
                        "CPU path uses float32 tensors in NHWC layout. "
                        "Input/output pointers come from SnpeUdo_TensorParam_t."
                    ),
                },
                {
                    "file": "jni/src/GPU/GpuImpl.cpp",
                    "runtime": "GPU",
                    "functions": [
                        "setKernelInfo",
                        f"{pkg_name}Operation",
                    ],
                    "notes": (
                        "GPU path uses OpenCL half (FP16) buffers. "
                        "Write an OpenCL kernel and set it via setKernelInfo(). "
                        "Input/output are cl_mem objects."
                    ),
                },
                {
                    "file": "jni/src/DSP/DspImpl.cpp",
                    "runtime": "DSP_V68",
                    "functions": [
                        "QnnOpPackage_executeOp",
                        "QnnOpPackage_validateOpConfig",
                        "QnnOpPackage_createOpImpl",
                        "QnnOpPackage_terminateOp",
                    ],
                    "notes": (
                        "DSP/HTP path uses INT8 quantised tensors with "
                        "affine encoding (scale + zero_point). Dequantise to "
                        "float32 for computation, then requantise output."
                    ),
                },
            ]

        # Real mode — scan for placeholder functions
        todos: list[dict] = []
        pkg_path = Path(package_dir)

        runtime_map = {
            "CPU": ("jni/src/CPU", [
                "SnpeUdo_validateOp",
                "SnpeUdo_executeOp",
                "SnpeUdo_terminateOp",
            ]),
            "GPU": ("jni/src/GPU", [
                "setKernelInfo",
                f"{pkg_name}Operation",
            ]),
            "DSP_V68": ("jni/src/DSP", [
                "QnnOpPackage_executeOp",
                "QnnOpPackage_validateOpConfig",
                "QnnOpPackage_createOpImpl",
                "QnnOpPackage_terminateOp",
            ]),
        }

        notes_map = {
            "CPU": (
                "CPU path uses float32 tensors in NHWC layout. "
                "Input/output pointers come from SnpeUdo_TensorParam_t."
            ),
            "GPU": (
                "GPU path uses OpenCL half (FP16) buffers. "
                "Write an OpenCL kernel and set it via setKernelInfo(). "
                "Input/output are cl_mem objects."
            ),
            "DSP_V68": (
                "DSP/HTP path uses INT8 quantised tensors with "
                "affine encoding (scale + zero_point). Dequantise to "
                "float32 for computation, then requantise output."
            ),
        }

        for runtime, (src_dir, functions) in runtime_map.items():
            full_dir = pkg_path / src_dir
            if full_dir.exists():
                for cpp_file in full_dir.glob("*.cpp"):
                    rel = str(cpp_file.relative_to(pkg_path))
                    todos.append({
                        "file": rel,
                        "runtime": runtime,
                        "functions": functions,
                        "notes": notes_map[runtime],
                    })

        return todos

    def check_environment(self) -> dict:
        """Validate that required environment variables are set for UDO generation.

        Checks the following environment variables:
          - ``SNPE_UDO_ROOT`` or ``QAIRT_SDK_ROOT`` — Core SDK path
          - ``QNN_SDK_ROOT`` — QNN SDK (required for HTP/DSP ops)
          - ``HEXAGON_SDK_ROOT`` — Hexagon toolchain (required for DSP compilation)
          - ``ANDROID_NDK_ROOT`` — Android NDK (required for on-device targets)

        Returns:
            {
                "ready": bool,
                "checks": list[{"var": str, "status": "set"|"missing", "path": str}],
                "can_generate": bool,    # SNPE_UDO_ROOT or QNN_SDK_ROOT
                "can_compile_cpu": bool,  # + nothing extra needed
                "can_compile_gpu": bool,  # + ANDROID_NDK_ROOT + CL_LIBRARY_PATH
                "can_compile_dsp": bool,  # + HEXAGON_SDK_ROOT
            }
        """
        env_vars = [
            "SNPE_UDO_ROOT",
            "QAIRT_SDK_ROOT",
            "QNN_SDK_ROOT",
            "HEXAGON_SDK_ROOT",
            "ANDROID_NDK_ROOT",
            "CL_LIBRARY_PATH",
        ]

        if self._mock:
            # Return a realistic mock response showing typical missing state
            checks = [
                {"var": "SNPE_UDO_ROOT", "status": "missing", "path": ""},
                {"var": "QAIRT_SDK_ROOT", "status": "missing", "path": ""},
                {"var": "QNN_SDK_ROOT", "status": "missing", "path": ""},
                {"var": "HEXAGON_SDK_ROOT", "status": "missing", "path": ""},
                {"var": "ANDROID_NDK_ROOT", "status": "missing", "path": ""},
                {"var": "CL_LIBRARY_PATH", "status": "missing", "path": ""},
            ]
            return {
                "ready": False,
                "checks": checks,
                "can_generate": False,
                "can_compile_cpu": False,
                "can_compile_gpu": False,
                "can_compile_dsp": False,
            }

        # Real mode — actually inspect environment
        checks: list[dict] = []
        for var in env_vars:
            val = os.environ.get(var, "")
            status = "set" if val else "missing"
            checks.append({"var": var, "status": status, "path": val})

        # Determine capabilities
        has_snpe = bool(os.environ.get("SNPE_UDO_ROOT") or os.environ.get("QAIRT_SDK_ROOT"))
        has_qnn = bool(os.environ.get("QNN_SDK_ROOT"))
        has_hexagon = bool(os.environ.get("HEXAGON_SDK_ROOT"))
        has_ndk = bool(os.environ.get("ANDROID_NDK_ROOT"))
        has_cl = bool(os.environ.get("CL_LIBRARY_PATH"))

        can_generate = has_snpe or has_qnn
        can_compile_cpu = can_generate  # CPU needs only the SDK headers
        can_compile_gpu = can_generate and has_ndk and has_cl
        can_compile_dsp = can_generate and has_hexagon

        return {
            "ready": can_generate,
            "checks": checks,
            "can_generate": can_generate,
            "can_compile_cpu": can_compile_cpu,
            "can_compile_gpu": can_compile_gpu,
            "can_compile_dsp": can_compile_dsp,
        }

    # ------------------------------------------------------------------
    # Step 6 — execute_on_android
    # ------------------------------------------------------------------

    def execute_on_android(
        self,
        model_dlc: str,
        input_list: str,
        reg_lib: str,
        runtime: str = "cpu",
        device_dir: str = "/data/local/tmp/inception_v3_udo",
    ) -> str:
        """Run ``snpe-net-run`` on the Android device.

        Args:
            model_dlc: Model ``.dlc`` filename on the device (basename).
            input_list: Input list filename on the device (basename).
            reg_lib: UDO registration library filename on the device (basename).
            runtime: One of ``"cpu"``, ``"gpu"``, ``"dsp"``, ``"aip"``.
            device_dir: Working directory on the device.

        Returns:
            Combined stdout+stderr from ``snpe-net-run``.
        """
        _runtime_flag: dict[str, str] = {
            "cpu": "cpu",
            "gpu": "gpu",
            "dsp": "dsp",
            "aip": "aip",
        }
        rt_flag = _runtime_flag.get(runtime.lower(), "cpu")

        if self._mock:
            mock_output = (
                f"[mock] snpe-net-run --container {model_dlc} "
                f"--input_list {input_list} "
                f"--udo_package_path {reg_lib} "
                f"--use_{rt_flag}\n"
                "Total inference time: 12.34 ms\n"
                "Output: output.raw\n"
            )
            logger.debug("UDOManager.execute_on_android [mock]")
            return mock_output

        # Real mode -------------------------------------------------------
        cmd_str = (
            f"cd {device_dir} && "
            f"export LD_LIBRARY_PATH={device_dir}:$LD_LIBRARY_PATH && "
            f"./snpe-net-run "
            f"--container {model_dlc} "
            f"--input_list {input_list} "
            f"--udo_package_path {reg_lib} "
            f"--use_{rt_flag}"
        )
        return self._run(["adb", "shell", cmd_str])
