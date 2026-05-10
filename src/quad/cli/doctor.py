"""QUAD CLI — Environment Diagnostics (quad doctor).

Checks the development environment for correct setup: Python version,
package installation, configuration, device detection, and more.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: str  # "pass", "warn", "fail"
    message: str


@dataclass
class DoctorReport:
    """Aggregated report from all diagnostic checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.status == "pass" for c in self.checks)

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]


def run_doctor(real_mode: bool = False) -> DoctorReport:
    """Run all environment diagnostic checks.

    Checks performed:
        1. Python version >= 3.10
        2. QUAD package installed and importable
        3. quad.toml exists and is valid
        4. Required dependencies available
        5. Device detection works
        6. Template directory exists
        7. Tests can import successfully
        8. DLC / WER compatibility
        9. SDK env vars, CLI tools, DSP env, Android tools
       10. QHAS prerequisites
       11. Adapter-mode consistency (real-mode pre-flight)

    Args:
        real_mode: If True, escalate SDK-related warnings to failures and
            add an adapter-mode consistency check. Use this for a strict
            pre-flight before running on physical hardware.

    Returns:
        DoctorReport with all check results.
    """
    report = DoctorReport()

    report.checks.append(_check_python_version())
    report.checks.append(_check_quad_importable())
    report.checks.append(_check_quad_toml())
    report.checks.append(_check_dependencies())
    report.checks.append(_check_device_detection())
    report.checks.append(_check_template_directory())
    report.checks.append(_check_test_imports())
    report.checks.append(_check_dlc_compatibility())
    report.checks.append(_check_wer_support())
    # SDK environment checks
    report.checks.append(_check_sdk_env_vars())
    report.checks.append(_check_sdk_tools_in_path())
    report.checks.append(_check_dsp_env())
    report.checks.append(_check_android_tools())
    report.checks.append(_check_qhas_prerequisites())

    # Optional integrations — informational, not strict pre-flight
    report.checks.append(_check_aimet_integration())
    report.checks.append(_check_aihub_integration())
    report.checks.append(_check_python_arch_vs_os())
    report.checks.append(_check_psutil_for_profiling())
    report.checks.append(_check_diagview_for_profiling())
    report.checks.append(_check_powercfg_for_power_estimation())

    if real_mode:
        report.checks.append(_check_adapter_mode_real())
        # Escalate any SDK-related warnings to failures
        sdk_check_names = {
            "SDK env vars",
            "SDK tools in PATH",
            "DSP env (ADSP_LIBRARY_PATH)",
            "QHAS prerequisites",
            "DLC compatibility",
        }
        for check in report.checks:
            if check.name in sdk_check_names and check.status == "warn":
                check.status = "fail"
                check.message = f"[real-mode strict] {check.message}"

    return report


def _check_adapter_mode_real() -> CheckResult:
    """Confirm adapter_mode='real' is configured and the SDK is reachable.

    Only included when ``run_doctor(real_mode=True)`` is called. Reports
    PASS only if the factory would actually return a real adapter.
    """
    try:
        from quad.adapters.factory import AdapterFactory
        from quad.config import load_config
    except Exception as e:
        return CheckResult(
            "Adapter mode (real)",
            "fail",
            f"Cannot load configuration to verify adapter mode: {e}",
        )

    try:
        cfg = load_config()
    except Exception as e:
        return CheckResult(
            "Adapter mode (real)",
            "fail",
            f"load_config() failed: {e}. Check quad.toml syntax.",
        )

    factory = AdapterFactory(cfg)
    ready, reason = factory.real_mode_ready()

    if ready:
        return CheckResult("Adapter mode (real)", "pass", reason)
    if cfg.adapter_mode != "real":
        return CheckResult(
            "Adapter mode (real)",
            "fail",
            f"adapter_mode is {cfg.adapter_mode!r}, not 'real'. "
            "Set adapter_mode = \"real\" in quad.toml or export QUAD_ADAPTER_MODE=real.",
        )
    return CheckResult("Adapter mode (real)", "fail", reason)


def _check_python_version() -> CheckResult:
    """Check Python version >= 3.10."""
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"

    if (major, minor) >= (3, 10):
        return CheckResult("Python version", "pass", f"Python {version_str} (>= 3.10)")
    elif (major, minor) >= (3, 9):
        return CheckResult("Python version", "warn", f"Python {version_str} — 3.10+ recommended")
    else:
        return CheckResult("Python version", "fail", f"Python {version_str} — requires 3.10+")


def _check_quad_importable() -> CheckResult:
    """Check that the quad package is importable."""
    try:
        import quad

        version = getattr(quad, "__version__", "unknown")
        return CheckResult("QUAD package", "pass", f"quad v{version} importable")
    except ImportError as e:
        return CheckResult("QUAD package", "fail", f"Cannot import quad: {e}")


def _check_quad_toml() -> CheckResult:
    """Check that quad.toml exists and is parseable."""
    # Look in CWD and parent directories
    search_paths = [Path.cwd(), Path.cwd().parent]
    for base in search_paths:
        toml_path = base / "quad.toml"
        if toml_path.exists():
            try:
                content = toml_path.read_text()
                if len(content.strip()) == 0:
                    return CheckResult("quad.toml", "warn", f"Found {toml_path} but it is empty")
                return CheckResult("quad.toml", "pass", f"Found valid {toml_path}")
            except Exception as e:
                return CheckResult("quad.toml", "fail", f"Error reading {toml_path}: {e}")

    return CheckResult("quad.toml", "warn", "No quad.toml found (optional but recommended)")


def _check_dependencies() -> CheckResult:
    """Check that required dependencies are available."""
    required = ["typer", "numpy"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return CheckResult("Dependencies", "pass", "All required packages available")
    else:
        return CheckResult("Dependencies", "fail", f"Missing packages: {', '.join(missing)}")


def _check_device_detection() -> CheckResult:
    """Check that device detection works."""
    try:
        from quad.runtime import list_devices

        devices = list_devices()
        names = [d.name for d in devices]
        return CheckResult("Device detection", "pass", f"Found {len(devices)} device(s): {', '.join(names)}")
    except Exception as e:
        return CheckResult("Device detection", "fail", f"Device detection failed: {e}")


def _check_template_directory() -> CheckResult:
    """Check that the template/codegen directory exists."""
    try:
        import quad

        pkg_dir = Path(quad.__file__).parent
        templates = pkg_dir / "codegen" / "templates"
        if templates.exists():
            return CheckResult("Template directory", "pass", f"Found {templates}")
        # Also check for codegen module
        codegen_dir = pkg_dir / "codegen"
        if codegen_dir.exists():
            return CheckResult("Template directory", "pass", f"Codegen module at {codegen_dir}")
        return CheckResult("Template directory", "warn", "Template directory not found (codegen may be limited)")
    except Exception as e:
        return CheckResult("Template directory", "fail", f"Error checking templates: {e}")


def _check_test_imports() -> CheckResult:
    """Check that test modules can be imported."""
    try:
        from quad.cli.quickstart import QuickstartResult
        from quad.cli.benchmark import BenchmarkReport

        return CheckResult("Test imports", "pass", "CLI modules importable")
    except ImportError as e:
        return CheckResult("Test imports", "fail", f"Import error: {e}")


def _check_dlc_compatibility() -> CheckResult:
    """Check SNPE SDK version and warn about known DLC migration issues.

    Key change at SNPE 1.16.0: batch dimension handling changed.
    Pre-1.16: {1,3,224,224} → {224,224,3}  (3D, batch dropped)
    Post-1.16: {1,3,224,224} → {1,224,224,3} (4D, batch preserved)
    """
    import os
    from quad.adapters.dlc_compat import parse_snpe_version, BATCH_DIM_CHANGE_VERSION, is_post_116

    sdk_root = (
        os.environ.get("QAIRT_SDK_ROOT")
        or os.environ.get("SNPE_ROOT")
        or os.environ.get("QNN_SDK_ROOT")
    )

    if not sdk_root:
        return CheckResult(
            "DLC compatibility",
            "warn",
            "QAIRT_SDK_ROOT not set — cannot check SDK version for DLC compatibility",
        )

    # Try to determine SDK version from path
    import re
    version_match = re.search(r"(\d+\.\d+\.\d+)", sdk_root)
    if not version_match:
        return CheckResult(
            "DLC compatibility",
            "pass",
            "QAIRT SDK detected. Run `quad doctor` with QAIRT_SDK_ROOT set to a versioned path for DLC compat check.",
        )

    version_str = version_match.group(1)
    if is_post_116(version_str):
        return CheckResult(
            "DLC compatibility",
            "pass",
            f"SNPE {version_str}: Post-1.16 batch dimension handling (4D tensors). "
            f"Tensor shapes include batch dimension (e.g. {{1,224,224,3}} not {{224,224,3}}).",
        )
    else:
        return CheckResult(
            "DLC compatibility",
            "warn",
            f"SNPE {version_str}: Pre-1.16 batch dimension handling (3D tensors, batch dropped). "
            f"Re-convert models with current SDK for correct 4D tensor support. "
            f"Application code using 3D tensors will need updating after migration.",
        )


def _check_wer_support() -> CheckResult:
    """Check if SDK version supports Windows Error Reporting (WER).

    WER auto-reports critical DSP errors from SNPE 2.28.0+ on Windows.
    No app code needed — always enabled, submission controlled by OS.
    """
    import os
    from quad.adapters.wer import get_wer_status, WER_MIN_SDK_VERSION
    from quad.adapters.dlc_compat import parse_snpe_version

    sdk_root = (
        os.environ.get("QAIRT_SDK_ROOT")
        or os.environ.get("SNPE_ROOT")
        or ""
    )
    if not sdk_root:
        return CheckResult(
            "WER support",
            "warn",
            "QAIRT_SDK_ROOT not set — cannot check WER availability",
        )

    import re
    match = re.search(r"(\d+\.\d+\.\d+)", sdk_root)
    if not match:
        return CheckResult("WER support", "pass", "QAIRT SDK detected (set versioned path for WER check)")

    status = get_wer_status(match.group(1))
    if status.wer_available:
        return CheckResult("WER support", "pass", status.note)
    return CheckResult("WER support", "warn", status.note)


def _check_sdk_env_vars() -> CheckResult:
    """Check that critical SDK environment variables are set and paths exist."""
    import os

    checks = [
        ("QAIRT_SDK_ROOT", "QAIRT/SNPE primary SDK"),
        ("QNN_SDK_ROOT", "QNN SDK (alternative to QAIRT)"),
        ("SNPE_ROOT", "Legacy SNPE SDK"),
    ]

    found = []
    missing = []
    bad_path = []

    for var, desc in checks:
        val = os.environ.get(var, "")
        if val:
            if Path(val).exists():
                found.append(f"{var}={val}")
            else:
                bad_path.append(f"{var}={val} (path does not exist)")
        else:
            missing.append(var)

    if bad_path:
        return CheckResult(
            "SDK env vars",
            "fail",
            f"SDK env vars with invalid paths: {'; '.join(bad_path)}. "
            "Run: source activate_qairt.sh",
        )
    if found:
        return CheckResult(
            "SDK env vars",
            "pass",
            f"SDK env vars set: {', '.join(found)}",
        )
    return CheckResult(
        "SDK env vars",
        "warn",
        f"No SDK env vars set ({', '.join(missing)}). "
        "Real mode disabled — running mock-only. "
        "Set QAIRT_SDK_ROOT to enable real hardware: export QAIRT_SDK_ROOT=/path/to/qairt",
    )


def _check_sdk_tools_in_path() -> CheckResult:
    """Check that QAIRT/SNPE CLI tools are available in PATH."""
    import shutil

    critical_tools = [
        ("qairt-converter", "model conversion"),
        ("snpe-net-run", "inference + profiling"),
    ]
    optional_tools = [
        ("qairt-quantizer", "INT8/INT4 quantization"),
        ("snpe-diagview", "profiling log viewer"),
        ("qnn-profile-viewer", "QHAS chrometrace generation"),
        ("snpe-dlc-graph-prepare", "graph caching + QHAS step 1"),
        ("snpe-platform-validator", "hardware capability check"),
    ]

    missing_critical = [t for t, _ in critical_tools if not shutil.which(t)]
    missing_optional = [t for t, _ in optional_tools if not shutil.which(t)]
    found_critical = [t for t, _ in critical_tools if shutil.which(t)]

    if missing_critical:
        return CheckResult(
            "SDK tools in PATH",
            "fail" if not found_critical else "warn",
            f"Critical tools missing: {', '.join(missing_critical)}. "
            "Run: source activate_qairt.sh  (or add SDK/bin to PATH). "
            + (f"Optional tools also missing: {', '.join(missing_optional[:3])}" if missing_optional else ""),
        )
    if missing_optional:
        return CheckResult(
            "SDK tools in PATH",
            "warn",
            f"Core tools found. Optional tools missing: {', '.join(missing_optional)}. "
            "Some profiling features will be unavailable.",
        )
    return CheckResult(
        "SDK tools in PATH",
        "pass",
        f"All SDK tools found: {', '.join(found_critical + [t for t, _ in optional_tools])}",
    )


def _check_dsp_env() -> CheckResult:
    """Check DSP/HTP library path environment variable (ADSP_LIBRARY_PATH).

    Required for HTP/DSP execution. Must include the directory containing
    libSnpeHtpV*Skel.so (and libQnnHtpV*Skel.so for QNN HTP backend).
    """
    import os

    adsp_path = os.environ.get("ADSP_LIBRARY_PATH", "")
    if not adsp_path:
        return CheckResult(
            "DSP env (ADSP_LIBRARY_PATH)",
            "warn",
            "ADSP_LIBRARY_PATH not set. HTP/DSP execution will fail at runtime. "
            "Set to the directory containing libSnpeHtpV*Skel.so: "
            "export ADSP_LIBRARY_PATH=/path/to/qairt/lib/aarch64-android;/vendor/lib/rfsa/adsp",
        )

    # Verify at least one directory in the path exists
    sep = ";" if ";" in adsp_path else ":"
    dirs = [d.strip() for d in adsp_path.split(sep) if d.strip()]
    existing = [d for d in dirs if Path(d).exists()]

    if not existing:
        return CheckResult(
            "DSP env (ADSP_LIBRARY_PATH)",
            "fail",
            f"ADSP_LIBRARY_PATH set but no directories exist: {adsp_path}",
        )
    return CheckResult(
        "DSP env (ADSP_LIBRARY_PATH)",
        "pass",
        f"ADSP_LIBRARY_PATH set with {len(existing)}/{len(dirs)} valid directories",
    )


def _check_android_tools() -> CheckResult:
    """Check Android development tools (ADB, NDK) for Phase 3 mobile support."""
    import os
    import shutil

    adb = shutil.which("adb")
    ndk_root = os.environ.get("ANDROID_NDK_ROOT", "")
    android_home = os.environ.get("ANDROID_HOME", "") or os.environ.get("ANDROID_SDK_ROOT", "")

    issues = []
    found = []

    if adb:
        found.append(f"adb={adb}")
    else:
        issues.append("adb not found (required for Android device communication)")

    if ndk_root:
        if Path(ndk_root).exists():
            found.append(f"ANDROID_NDK_ROOT={ndk_root}")
        else:
            issues.append(f"ANDROID_NDK_ROOT set but path missing: {ndk_root}")
    else:
        issues.append("ANDROID_NDK_ROOT not set (required for Android AAR generation)")

    if not found and issues:
        return CheckResult(
            "Android tools",
            "warn",
            "Android tools not found — Phase 3 (mobile) unavailable. "
            + "; ".join(issues),
        )
    if issues:
        return CheckResult(
            "Android tools",
            "warn",
            f"Partial Android setup ({', '.join(found)}). Missing: {'; '.join(issues)}",
        )
    return CheckResult(
        "Android tools",
        "pass",
        f"Android tools available: {', '.join(found)}",
    )


def _check_qhas_prerequisites() -> CheckResult:
    """Check QHAS profiling prerequisites.

    QHAS chrometrace generation requires:
    1. snpe-dlc-graph-prepare (step 1 — schematic generation)
    2. qnn-profile-viewer (step 3 — chrometrace generation)
    3. libQnnHtpOptraceProfilingReader.so in SDK lib directory
    """
    import os
    import shutil

    from quad.profiler.qhas import QHAS_READER_LIB

    issues = []
    found = []

    if shutil.which("snpe-dlc-graph-prepare"):
        found.append("snpe-dlc-graph-prepare")
    else:
        issues.append("snpe-dlc-graph-prepare not in PATH (QHAS step 1)")

    if shutil.which("qnn-profile-viewer"):
        found.append("qnn-profile-viewer")
    else:
        issues.append("qnn-profile-viewer not in PATH (QHAS step 3 — chrometrace)")

    # Check for reader .so in SDK
    sdk_root = os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("QNN_SDK_ROOT") or ""
    if sdk_root:
        reader_candidates = list(Path(sdk_root).rglob(QHAS_READER_LIB))
        if reader_candidates:
            found.append(f"{QHAS_READER_LIB} ({reader_candidates[0]})")
        else:
            issues.append(
                f"{QHAS_READER_LIB} not found under {sdk_root} "
                "(required for QHAS chrometrace on-device)"
            )
    else:
        issues.append("SDK root not set — cannot check for reader library")

    if not issues:
        return CheckResult("QHAS prerequisites", "pass", f"QHAS ready: {', '.join(found)}")
    if found:
        return CheckResult(
            "QHAS prerequisites",
            "warn",
            f"Partial QHAS setup ({', '.join(found)}). "
            f"Issues: {'; '.join(issues)}",
        )
    return CheckResult(
        "QHAS prerequisites",
        "warn",
        f"QHAS profiling not available: {'; '.join(issues)}. "
        "Install QAIRT SDK and ensure tools are in PATH.",
    )


def _check_aimet_integration() -> CheckResult:
    """Check AIMET adapter availability for INT8/INT4 quantization (T1.5)."""
    try:
        from quad.adapters.aimet_adapter import AIMETAdapter
    except Exception as e:
        return CheckResult(
            "AIMET integration",
            "fail",
            f"Cannot import AIMETAdapter: {e}",
        )

    a = AIMETAdapter(backend="auto")
    info = a.doctor()
    if info["aimet_torch_installed"] or info["aimet_onnx_installed"]:
        which = "torch" if info["aimet_torch_installed"] else "onnx"
        return CheckResult(
            "AIMET integration",
            "pass",
            f"aimet_{which} available; backend={info['backend']}. "
            f"INT8/INT4 quantization workflows enabled.",
        )
    return CheckResult(
        "AIMET integration",
        "warn",
        "aimet not installed. INT8/INT4 quantization will use the qairt-quantizer "
        "fallback (less accurate without proper calibration data). Install via: "
        "pip install aimet-torch  (or aimet-onnx for ONNX-only flows).",
    )


def _check_aihub_integration() -> CheckResult:
    """Check Qualcomm AI Hub cloud integration (T1.6)."""
    try:
        from quad.adapters.aihub_adapter import AIHubAdapter
    except Exception as e:
        return CheckResult(
            "AI Hub integration",
            "fail",
            f"Cannot import AIHubAdapter: {e}",
        )

    a = AIHubAdapter(backend="auto")
    info = a.doctor()
    if info["backend"] == "qai_hub":
        return CheckResult(
            "AI Hub integration",
            "pass",
            "qai_hub installed and authenticated. Cloud profiling and compilation enabled.",
        )
    if info["qai_hub_installed"] and not info["auth_configured"]:
        return CheckResult(
            "AI Hub integration",
            "warn",
            "qai_hub installed but auth not configured. Run: "
            "qai-hub configure --api_token <token>  "
            "or set QAI_HUB_API_KEY env var. Get a token at https://app.aihub.qualcomm.com",
        )
    return CheckResult(
        "AI Hub integration",
        "warn",
        "qai_hub not installed. Cloud profiling and compilation disabled. "
        "Install via: pip install qai-hub  (or set QUAD_AIHUB_BACKEND=mock for tests).",
    )


def _check_python_arch_vs_os() -> CheckResult:
    """Catch the Snapdragon X Elite x86_64-Python pitfall.

    On ARM64 Windows (Copilot+ PCs), an x86_64 Python runs through Prism
    emulation. QAIRT's ``qti.aisw.dlc_utils.__init__`` keys off
    ``platform.processor()`` — which returns "ARMv8…" on the host CPU
    regardless of the Python's bitness — and tries to load the
    ``windows-arm64ec/`` .pyd. That .pyd is ARM64-native and won't load
    into emulated x86_64 Python; the user sees:

        ImportError: DLL load failed while importing libDlModelToolsPy

    Native ARM64 Python loads ``windows-arm64ec/`` correctly. We surface
    a clear warning so the user can install python-arm64 from python.org
    instead of debugging an opaque DLL error.
    """
    import platform
    import sys
    import sysconfig

    # uname.machine reflects the OS architecture; sysconfig.get_platform()
    # reflects what Python was built for.
    os_arch = (platform.uname().machine or "").upper()
    py_arch = sysconfig.get_platform().lower()

    if os_arch in ("ARM64", "AARCH64") and ("amd64" in py_arch or "x86" in py_arch):
        return CheckResult(
            "Python arch vs OS",
            "warn",
            "Detected x86_64 Python on ARM64 Windows (Prism emulation). "
            "QAIRT host tools (qairt-converter, qairt-quantizer, *-onnx-converter) "
            "may fail with 'libDlModelToolsPy ImportError' because QAIRT's "
            "qti.aisw.dlc_utils.__init__ picks the wrong .pyd path. "
            "Install native ARM64 Python from python.org/downloads/windows "
            "(look for 'Windows arm64' installer) and recreate the venv. "
            "QUAD's runtime side (snpe-net-run, qnn-platform-validator) is "
            "unaffected — only the host conversion path needs native Python.",
        )
    return CheckResult(
        "Python arch vs OS",
        "pass",
        f"Python ({py_arch}) matches OS arch ({os_arch}); "
        "QAIRT host tools should load the correct .pyd.",
    )


def _check_psutil_for_profiling() -> CheckResult:
    """Verify psutil is importable — used for RSS sampling + CPU%."""
    try:
        import psutil
    except ImportError:
        return CheckResult(
            "psutil (RSS + CPU%)",
            "warn",
            "psutil not installed. Memory + CPU utilisation will report "
            "as not_measured. Install via: pip install -e .[real]",
        )
    return CheckResult(
        "psutil (RSS + CPU%)",
        "pass",
        f"psutil {getattr(psutil, '__version__', '?')} ready for RSS sampling + cpu_percent.",
    )


def _check_diagview_for_profiling() -> CheckResult:
    """snpe-diagview is required to extract structured metrics from the
    binary diaglog produced by snpe-net-run."""
    from quad.profiler.diagview import find_diagview

    tool = find_diagview()
    if not tool:
        return CheckResult(
            "snpe-diagview",
            "warn",
            "snpe-diagview not on PATH. Per-layer + accurate latency "
            "metrics will fall back to stdout parsing only. Source the "
            "QAIRT envsetup or activate.ps1 to add it.",
        )
    return CheckResult(
        "snpe-diagview",
        "pass",
        f"snpe-diagview at {tool} (used to convert SNPEDiag_*.bin to CSV).",
    )


def _check_powercfg_for_power_estimation() -> CheckResult:
    """powercfg.exe is the no-extra-tooling Windows power source."""
    from quad.profiler.host_power import srumutil_available

    import os
    if os.name != "nt":
        return CheckResult(
            "powercfg / SRUM",
            "pass",
            "Non-Windows host — power estimation uses the host_thermal_model.",
        )
    if not srumutil_available():
        return CheckResult(
            "powercfg / SRUM",
            "warn",
            "powercfg.exe not on PATH (unusual on Windows). Power values "
            "will fall back to the host_thermal_model estimate only.",
        )
    return CheckResult(
        "powercfg / SRUM",
        "pass",
        "powercfg.exe ready — SRUM Energy Estimation can supplement the "
        "host_thermal_model on Windows ARM64.",
    )
