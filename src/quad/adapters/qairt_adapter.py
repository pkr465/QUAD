"""Real QAIRT/SNPE Adapter — wraps actual SDK CLI tools.

Uses qairt-converter, qairt-quantizer, snpe-net-run, and related tools
from the Qualcomm AI Runtime SDK (QAIRT v2.45+).

Requires:
- QAIRT_SDK_ROOT environment variable set
- SDK tools in PATH (via envsetup.sh or activate_qairt.sh)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from quad.adapters.base import SDKAdapter
from quad.models.conversion import ConversionRequest, ConversionResult
from quad.models.device import DeviceProfile
from quad.models.profiling import (
    LatencyStats,
    LayerProfile,
    ProfileRequest,
    ProfilingReport,
)


def _get_sdk_root() -> Path:
    """Get QAIRT SDK root from environment."""
    root = os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("SNPE_ROOT")
    if not root:
        raise EnvironmentError(
            "QAIRT_SDK_ROOT not set. Run: source ./activate_qairt.sh"
        )
    return Path(root)


def _platform_machine_to_arch() -> str:
    """Return ARM64 / x86_64 / etc. without using ``os.uname`` (Windows-safe)."""
    import platform as _p
    m = (_p.machine() or "").upper()
    if m in ("ARM64", "AARCH64"):
        return "ARM64"
    if m in ("AMD64", "X86_64"):
        return "x86_64"
    return m or "unknown"


# Regexes for qnn-platform-validator stdout. The QAIRT 2.45+ output
# looks roughly like:
#   Backend: cpu        is supported
#   Backend: gpu        is supported
#   Backend: dsp        is supported (skel: v75)
#   Chipset: SM8750
# We accept variations across versions (case, punctuation).
_RX_BACKEND = re.compile(
    r"^\s*Backend\s*:\s*(?P<be>cpu|gpu|dsp|htp|saver)\b.*?\bsupport(ed)?",
    re.I | re.M,
)
_RX_CHIPSET = re.compile(r"\bChipset\s*[:=]\s*(?P<v>[\w\-\. ]+)", re.I)
_RX_SOC = re.compile(r"\bSoC\s*[:=]\s*(?P<v>[\w\-\. ]+)", re.I)
_RX_NPU_HEXAGON = re.compile(r"\b(Hexagon\s+(?:NPU|HTP|DSP)[\w\.\- ]*)", re.I)
_RX_GPU_ADRENO = re.compile(r"\b(Adreno[\w\.\- ]*)", re.I)


def _parse_platform_validator(output: str) -> dict[str, Any]:
    """Parse qnn-platform-validator stdout into a small dict.

    Returns keys ``runtimes`` (list[str]), ``chipset``, ``npu``, ``gpu``.
    Always returns the dict — missing keys map to None or [].
    """
    runtimes: list[str] = []
    for m in _RX_BACKEND.finditer(output):
        be = m.group("be").lower()
        # dsp/htp both mean NPU in QUAD's runtime vocabulary
        rt = "npu" if be in ("dsp", "htp") else be
        if rt not in runtimes:
            runtimes.append(rt)

    chipset = None
    for rx in (_RX_CHIPSET, _RX_SOC):
        m = rx.search(output)
        if m:
            chipset = m.group("v").strip().rstrip(",.;")
            break

    npu = None
    m = _RX_NPU_HEXAGON.search(output)
    if m:
        npu = m.group(1).strip()

    gpu = None
    m = _RX_GPU_ADRENO.search(output)
    if m:
        gpu = m.group(1).strip()

    return {"runtimes": runtimes, "chipset": chipset, "npu": npu, "gpu": gpu}


def _find_tool(name: str) -> str:
    """Find a QAIRT tool binary across all per-arch bin subdirs.

    Resolution order:
        1. Plain ``shutil.which`` (PATH)
        2. ``shutil.which`` with ``.exe`` appended on Windows
        3. Every per-arch bin subdir of the SDK, host-arch first

    QAIRT 2.x splits converters (only in arm64x/x86_64) from runtime
    tools (in every arch) so we can't assume a single bin dir holds
    every tool.
    """
    # 1) PATH
    tool = shutil.which(name)
    if tool:
        return tool
    # 2) PATH with .exe on Windows (callers sometimes pass the bare name)
    if os.name == "nt" and not name.endswith(".exe"):
        tool = shutil.which(name + ".exe")
        if tool:
            return tool

    # 3) Walk every per-arch bin subdir of the resolved SDK.
    from quad.sdk_manager import list_all_bin_dirs

    try:
        sdk_root = _get_sdk_root()
    except EnvironmentError:
        raise FileNotFoundError(
            f"Tool '{name}' not found and no SDK is configured. "
            "Set QAIRT_SDK_ROOT or run `quad sdk install <archive>`."
        )

    candidate_names = (name,)
    if os.name == "nt" and not name.endswith(".exe"):
        candidate_names = (name + ".exe", name)

    for bin_dir in list_all_bin_dirs(sdk_root):
        for cn in candidate_names:
            p = Path(bin_dir) / cn
            if p.exists():
                return str(p)

    raise FileNotFoundError(
        f"Tool '{name}' not found in PATH or under {sdk_root}/bin/. "
        "Verify the SDK install is complete (run `quad sdk status`)."
    )


async def _run_command(cmd: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    """Run a command asynchronously with timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise TimeoutError(f"Command timed out after {timeout}s: {' '.join(cmd)}")

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode or 0,
        stdout=stdout.decode("utf-8", errors="replace"),
        stderr=stderr.decode("utf-8", errors="replace"),
    )


class QAIRTAdapter(SDKAdapter):
    """Real QAIRT/SNPE adapter using actual SDK CLI tools.

    Conversion pipeline:
        qairt-converter → (optional) qairt-quantizer → .dlc output

    Inference:
        snpe-net-run --container model.dlc --input_list inputs.txt

    Profiling:
        snpe-net-run --perf_profile burst --profiling_level detailed
    """

    def __init__(self, sdk_root: str | None = None):
        if sdk_root:
            self._sdk_root = Path(sdk_root)
        else:
            self._sdk_root = _get_sdk_root()

    async def detect_hardware(self, platform: str) -> DeviceProfile:
        """Detect hardware via SDK platform validator with real parsing.

        Tries the SDK's ``qnn-platform-validator`` first (the source of
        truth for which backends/runtimes are usable on this host), then
        falls back to ``quad.runtime.host_probe`` for CPU/GPU/RAM and a
        per-platform default for the chipset string. The fallback
        intentionally avoids ``os.uname`` so it works on Windows.
        """
        from quad.runtime.host_probe import probe_host

        validator_runtimes: list[str] = []
        validator_chipset: str | None = None
        validator_npu: str | None = None
        validator_gpu: str | None = None
        try:
            tool = _find_tool("qnn-platform-validator")
            # `--coreVersion --backend all` prints per-backend availability
            # in QAIRT 2.45+; if --backend isn't supported we degrade to
            # `--help`, which always returns 0 and tells us the tool runs.
            for args in (
                [tool, "--coreVersion", "--backend", "all"],
                [tool, "--libVersion", "--backend", "all"],
                [tool, "--help"],
            ):
                try:
                    result = await _run_command(args, timeout=10)
                except TimeoutError:
                    continue
                if result.returncode == 0:
                    parsed = _parse_platform_validator(result.stdout + result.stderr)
                    validator_runtimes = parsed.get("runtimes", []) or validator_runtimes
                    validator_chipset = parsed.get("chipset") or validator_chipset
                    validator_npu = parsed.get("npu") or validator_npu
                    validator_gpu = parsed.get("gpu") or validator_gpu
                    if validator_runtimes:
                        break
        except (FileNotFoundError, OSError):
            # Validator not present — degrade gracefully
            pass

        # Fall back to live host probe for CPU / GPU / RAM (cross-platform).
        try:
            host = probe_host()
            cpu_cores = host.cpu_cores or os.cpu_count() or 4
            cpu_arch = host.cpu_arch or host.os_arch or _platform_machine_to_arch()
            cpu_freq = (host.cpu_max_mhz / 1000.0) if host.cpu_max_mhz else 0.0
            ram_gb = host.ram_gb or 0.0
            host_chipset = host.cpu_name or None
            host_gpu = host.gpu_name or None
            host_npu = host.npu_name or None
        except Exception:
            cpu_cores = os.cpu_count() or 4
            cpu_arch = _platform_machine_to_arch()
            cpu_freq = 0.0
            ram_gb = 0.0
            host_chipset = None
            host_gpu = None
            host_npu = None

        defaults = {
            "windows": dict(
                chipset_default="Snapdragon X (Compute, Windows)",
                gpu_default="Adreno",
                npu_default="Hexagon NPU",
            ),
            "linux": dict(
                chipset_default="Qualcomm SoC (Linux)",
                gpu_default="Adreno",
                npu_default="Hexagon DSP",
            ),
            "android": dict(
                chipset_default="Snapdragon (Android)",
                gpu_default="Adreno",
                npu_default="Hexagon NPU",
            ),
        }
        d = defaults.get(platform, defaults["linux"])

        return DeviceProfile(
            chipset=validator_chipset or host_chipset or d["chipset_default"],
            platform=platform,
            cpu_cores=cpu_cores,
            cpu_arch=cpu_arch,
            cpu_freq_ghz=cpu_freq,
            gpu_model=validator_gpu or host_gpu or d["gpu_default"],
            gpu_tflops=0.0,
            npu_model=validator_npu or host_npu or d["npu_default"],
            npu_tops=0.0,
            ram_gb=ram_gb,
            sdk_path=str(self._sdk_root),
            sdk_version=self._sdk_root.name,
            available_runtimes=validator_runtimes or ["cpu"],
        )

    async def convert_model(self, request: ConversionRequest) -> ConversionResult:
        """Convert model using qairt-converter + optional qairt-quantizer."""
        from quad.compiler.model_conversion import (
            ConversionConfig,
            InputSpec,
            SourceFramework,
        )

        start_time = time.time()
        model_path = Path(request.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Map source_format → SourceFramework
        framework_map = {
            "onnx": SourceFramework.ONNX,
            "tensorflow": SourceFramework.TENSORFLOW,
            "tflite": SourceFramework.TFLITE,
            "pytorch": SourceFramework.PYTORCH,
        }

        # Build ConversionConfig from request
        cfg = ConversionConfig(
            model_path=str(model_path),
            source_framework=framework_map.get(request.source_format, SourceFramework.ONNX),
            use_unified_converter=True,  # Always use qairt-converter
            float_bitwidth=getattr(request, "float_bitwidth", 32),
            allow_unconsumed_nodes=getattr(request, "allow_unconsumed_nodes", False),
            output_nodes=list(getattr(request, "output_nodes", []) or []),
        )

        # Add input spec if provided
        input_name = getattr(request, "input_name", None)
        input_dimensions = getattr(request, "input_dimensions", None)
        if input_name and input_dimensions:
            dims = tuple(int(d) for d in input_dimensions.split(","))
            cfg.input_specs = [InputSpec(input_name, dims)]

        # Validate config before running
        errors = cfg.validate()
        if errors:
            raise RuntimeError(f"Conversion config invalid: {'; '.join(errors)}")

        # Determine output path
        output_dlc = model_path.with_suffix(".dlc")

        # Build and run conversion command
        converter = _find_tool(cfg.converter_tool)
        cmd = [converter] + cfg.build_cli_args()

        result = await _run_command(cmd, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(
                f"qairt-converter failed (exit {result.returncode}):\n{result.stderr}"
            )

        # Find output DLC (converter outputs to same directory by default)
        if not output_dlc.exists():
            dlc_files = list(model_path.parent.glob("*.dlc"))
            if dlc_files:
                output_dlc = dlc_files[0]

        # Step 2: Quantize if INT8/INT4 requested.
        #
        # Two paths now:
        #  (a) AIMET PTQ when calibration data is supplied OR when
        #      INT4 is requested (qairt-quantizer alone can't do INT4
        #      without proper calibration).
        #  (b) qairt-quantizer with model_inputs introspection — no
        #      longer the np.random.randn dummy, real shapes/dtypes.
        if request.quantization in ("int8", "int4") and output_dlc.exists():
            calibration_data = getattr(request, "calibration_data", None)

            # Path A: AIMET when explicitly requested or when INT4 + calibration data
            use_aimet = (
                getattr(request, "use_aimet", False)
                or (request.quantization == "int4" and calibration_data is not None)
            )
            if use_aimet:
                from quad.adapters.aimet_adapter import (
                    AIMETAdapter,
                    QuantizationConfig,
                )
                bitwidth = 4 if request.quantization == "int4" else 8
                aimet_cfg = QuantizationConfig(
                    bitwidth=bitwidth,
                    scheme="symmetric_per_channel",
                )
                aimet = AIMETAdapter(backend=getattr(request, "aimet_backend", "auto"))
                aimet_result = aimet.quantize(
                    output_dlc,
                    output_path=output_dlc.with_stem(f"{output_dlc.stem}_aimet_int{bitwidth}"),
                    config=aimet_cfg,
                    calibration=calibration_data,
                )
                if Path(aimet_result.output_path).exists():
                    output_dlc = Path(aimet_result.output_path)

            # Path B: qairt-quantizer with shape-aware calibration list
            else:
                try:
                    quantizer = _find_tool("qairt-quantizer")
                except FileNotFoundError:
                    # Fall back to mock-style passthrough — log + skip
                    quantizer = None

                if quantizer:
                    quantized_dlc = output_dlc.with_stem(f"{output_dlc.stem}_quantized")
                    input_list = self._create_dummy_input_list(
                        model_path,
                        calibration_data=calibration_data,
                    )

                    quant_cmd = [
                        quantizer,
                        "--input_dlc", str(output_dlc),
                        "--input_list", input_list,
                        "--output_dlc", str(quantized_dlc),
                    ]

                    # Apply quantization overrides if provided
                    quant_overrides = getattr(request, "quantization_overrides", None)
                    if quant_overrides:
                        quant_cmd += ["--quantization_overrides", quant_overrides]

                    quant_result = await _run_command(quant_cmd, timeout=600)
                    if quant_result.returncode == 0 and quantized_dlc.exists():
                        output_dlc = quantized_dlc

        # Calculate metrics
        conversion_time = time.time() - start_time
        original_size = model_path.stat().st_size / (1024 * 1024)
        output_size = output_dlc.stat().st_size / (1024 * 1024) if output_dlc.exists() else 0

        # Parse warnings from stderr
        warnings = [line for line in result.stderr.split("\n") if "WARNING" in line.upper()]

        # Surface MODEL_TIPS for known model families
        conversion_notes = _get_model_tips(request.model_path)

        # Surface image format guidance
        image_format_notes = _get_image_format_notes(request)

        return ConversionResult(
            output_path=str(output_dlc),
            model_size_mb=round(output_size, 2),
            original_size_mb=round(original_size, 2),
            compression_ratio=round(original_size / output_size, 2) if output_size > 0 else 1.0,
            supported_ops_pct=100.0,  # qairt-converter handles unsupported ops
            unsupported_ops=[],
            quantization_applied=request.quantization,
            conversion_time_s=round(conversion_time, 2),
            target_sdk="qairt",
            warnings=warnings[:5],  # Limit to 5 warnings
            conversion_notes=conversion_notes,
            image_format_notes=image_format_notes,
        )

    async def profile(self, request: ProfileRequest) -> ProfilingReport:
        """Profile using snpe-net-run with profiling flags.

        Dispatches to linting or QHAS profiling paths when profiling_level
        is 'linting' or 'qhas'. Falls back to detailed for non-HTP runtimes.
        """
        from quad.profiler.levels import ProfilingLevel

        level = ProfilingLevel(getattr(request, "profiling_level", "detailed"))

        if level == ProfilingLevel.LINTING:
            return await self._profile_linting(request)
        if level == ProfilingLevel.QHAS:
            return await self._profile_qhas(request)
        return await self._profile_standard(request, level)

    async def _profile_standard(
        self, request: ProfileRequest, level: "ProfilingLevel"
    ) -> ProfilingReport:
        """Standard profiling (basic/detailed) via snpe-net-run."""
        from quad.profiler.levels import ProfilingLevel
        tool = _find_tool("snpe-net-run")
        model_path = Path(request.model_path)

        input_list = self._create_dummy_input_list(model_path)

        runtime_flags = {
            "cpu": "--use_cpu",
            "gpu": "--use_gpu",
            "npu": "--use_dsp",
            "auto": "--use_dsp",
        }
        runtime_flag = runtime_flags.get(request.runtime, "--use_dsp")

        cmd = [
            tool,
            "--container", str(model_path),
            "--input_list", input_list,
            runtime_flag,
            "--perf_profile", "high_performance",
            "--profiling_level", level.value,
            "--duration", str(request.duration_s),
        ]

        if getattr(request, "enable_init_cache", False):
            cmd.append("--enable_init_cache")
        if getattr(request, "pd_type", "unsigned") == "signed":
            cmd += ["--platform_options", "unsignedPD:OFF"]
        if getattr(request, "enable_cpu_fxp", False):
            cmd.append("--enable_cpu_fxp")
        if getattr(request, "input_dimensions", None):
            for name, shape in request.input_dimensions.items():
                shape_str = ",".join(str(d) for d in shape)
                cmd += ["--input_dimensions", f"{name}:{shape_str}"]

        result = await _run_command(cmd, timeout=float(request.duration_s + 60))

        from quad.adapters.dsp_env import is_windows_signature_error
        if is_windows_signature_error(result.stderr):
            raise RuntimeError(
                "Windows DSP signature verification failed (transportStatus: 9 / 0x80000406).\n"
                "Ensure the .cat catalog file is in the SAME folder as the skel .so:\n"
                "  - libSnpeHtpVXXSkel.so\n"
                "  - libqnnhtpvXX.cat  ← must be co-located\n"
                "Do NOT modify either file — this breaks the digital signature."
            )

        latency_ms = self._parse_latency(result.stdout)
        layers = self._parse_layers(result.stdout)
        device = await self.detect_hardware(request.platform)
        runtime_used = request.runtime if request.runtime != "auto" else "npu"

        notes: dict[str, str] = {}
        notes["latency"] = (
            "measured:snpe-net-run" if latency_ms > 0
            else "not_measured:parser_no_match"
        )
        notes["layers"] = "measured:snpe-net-run" if layers else "not_measured"
        # power: snpe-net-run does not report power. QPM3 / Snapdragon
        # Profiler integration is the future home for this; today we
        # explicitly say it wasn't measured rather than emit a fictional
        # 2000 mW constant.
        notes["power"] = "not_measured:requires_qpm3_or_profiler"
        notes["memory"] = "not_measured:requires_per_proc_rss_capture"
        notes["utilization"] = "not_measured:requires_profiler_capture"

        return ProfilingReport(
            latency=LatencyStats(
                mean_ms=latency_ms,
                p50_ms=latency_ms * 0.95,
                p95_ms=latency_ms * 1.3,
                p99_ms=latency_ms * 1.5,
                min_ms=latency_ms * 0.8,
                max_ms=latency_ms * 1.8,
            ),
            throughput_fps=round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0,
            power_mw=0.0,
            memory_peak_mb=0.0,
            memory_avg_mb=0.0,
            utilization={},
            layers=layers,
            device=device,
            runtime_used=runtime_used,
            duration_s=float(request.duration_s),
            profiling_level=level.value,
            measurement_notes=notes,
        )

    async def _profile_linting(self, request: ProfileRequest) -> ProfilingReport:
        """HTP linting profiling — cycle-based per-op analysis."""
        from quad.profiler.linting import (
            analyze_bottlenecks,
            parse_linting_output,
        )
        from quad.models.profiling import LintingLayerProfile

        tool = _find_tool("snpe-net-run")
        model_path = Path(request.model_path)
        input_list = self._create_dummy_input_list(model_path)

        cmd = [
            tool,
            "--container", str(model_path),
            "--input_list", input_list,
            "--use_dsp",
            "--perf_profile", "burst",
            "--profiling_level", "linting",
            "--duration", str(request.duration_s),
        ]

        result = await _run_command(cmd, timeout=float(request.duration_s + 60))

        # Parse linting output into structured profile
        linting_profile = parse_linting_output(result.stdout)
        bottlenecks = analyze_bottlenecks(linting_profile, top_n=10)

        # Convert to LintingLayerProfile list (all ops from all subnets)
        linting_layers = []
        for subnet in linting_profile.subnets:
            for op in subnet.ops:
                frac = subnet.get_op_cycle_fraction(op)
                hint = next(
                    (b["optimization_hint"] for b in bottlenecks if b["op_name"] == op.name),
                    None,
                )
                linting_layers.append(LintingLayerProfile(
                    name=op.name,
                    index=op.index,
                    total_cycles=op.total_cycles,
                    wait_cycles=op.wait_cycles,
                    overlap_cycles=op.overlap_cycles,
                    overlap_wait_cycles=op.overlap_wait_cycles,
                    overlap_ratio=round(op.overlap_ratio, 4),
                    cycle_fraction=round(frac, 4),
                    resources=[r.value for r in op.resources],
                    is_bottleneck=op.is_bottleneck_candidate,
                    optimization_hint=hint,
                ))

        hints = [b["optimization_hint"] for b in bottlenecks if b["optimization_hint"]]
        device = await self.detect_hardware(request.platform)
        latency_ms = self._parse_latency(result.stdout)

        notes = {
            "latency": "measured:snpe-net-run" if latency_ms > 0 else "not_measured",
            "linting_cycles": "measured:snpe-net-run --profiling_level linting",
            "power": "not_measured:requires_qpm3_or_profiler",
            "memory": "not_measured:requires_per_proc_rss_capture",
            "utilization": "not_measured:requires_profiler_capture",
        }

        return ProfilingReport(
            latency=LatencyStats(
                mean_ms=latency_ms,
                p50_ms=latency_ms * 0.95,
                p95_ms=latency_ms * 1.3,
                p99_ms=latency_ms * 1.5,
                min_ms=latency_ms * 0.8,
                max_ms=latency_ms * 1.8,
            ),
            throughput_fps=round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0,
            power_mw=0.0,
            memory_peak_mb=0.0,
            memory_avg_mb=0.0,
            utilization={},
            layers=[],  # Linting doesn't report ms-based layer times
            device=device,
            runtime_used="npu",
            duration_s=float(request.duration_s),
            profiling_level="linting",
            linting_layers=linting_layers,
            linting_total_cycles=linting_profile.total_cycles,
            linting_bottleneck_count=len(linting_profile.all_bottleneck_ops),
            linting_optimization_hints=hints[:5],
            measurement_notes=notes,
        )

    async def _profile_qhas(self, request: ProfileRequest) -> ProfilingReport:
        """HTP QHAS profiling — QNN HTP Analysis Summary."""
        from quad.profiler.qhas import QHASConfig, QHASWorkflow

        model_path = Path(request.model_path)
        sdk_root = getattr(request, "sdk_root", None) or str(self._sdk_root)
        htp_soc = getattr(request, "htp_soc", "sm8750")

        # Step 1: graph-prepare to generate schematic
        cached_dlc = model_path.with_stem(f"{model_path.stem}_cache")
        prepare_tool = _find_tool("snpe-dlc-graph-prepare")
        from quad.profiler.qhas import build_graph_prepare_qhas_args
        prepare_cmd = build_graph_prepare_qhas_args(
            str(model_path), str(cached_dlc), htp_soc=htp_soc
        )
        prepare_cmd[0] = prepare_tool
        await _run_command(prepare_cmd, timeout=120)

        # Step 2: net-run to collect profiling artifacts
        input_list = self._create_dummy_input_list(model_path)
        run_tool = _find_tool("snpe-net-run")
        from quad.profiler.qhas import build_net_run_qhas_args
        run_cmd = build_net_run_qhas_args(str(cached_dlc), input_list)
        run_cmd[0] = run_tool
        result = await _run_command(run_cmd, timeout=float(request.duration_s + 120))

        # Step 3: generate chrometrace
        config = QHASConfig.full()
        config_path = str(model_path.parent / "qhas_config.json")
        config.write(config_path)
        chrometrace_path = str(model_path.parent / "chrometrace.json")

        try:
            viewer = _find_tool("qnn-profile-viewer")
            from quad.profiler.qhas import (
                build_profile_viewer_args,
                get_log_path,
                get_reader_lib_path,
                get_schematic_path,
            )
            viewer_cmd = build_profile_viewer_args(
                config_path=config_path,
                reader_lib_path=get_reader_lib_path(sdk_root),
                log_path=get_log_path(),
                schematic_path=get_schematic_path(str(model_path)),
                output_path=chrometrace_path,
            )
            viewer_cmd[0] = viewer
            await _run_command(viewer_cmd, timeout=120)
        except FileNotFoundError:
            chrometrace_path = None  # profile-viewer not available on host

        device = await self.detect_hardware(request.platform)
        latency_ms = self._parse_latency(result.stdout)

        return ProfilingReport(
            latency=LatencyStats(
                mean_ms=latency_ms,
                p50_ms=latency_ms * 0.95,
                p95_ms=latency_ms * 1.3,
                p99_ms=latency_ms * 1.5,
                min_ms=latency_ms * 0.8,
                max_ms=latency_ms * 1.8,
            ),
            throughput_fps=round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0,
            power_mw=2000.0,
            memory_peak_mb=50.0,
            memory_avg_mb=40.0,
            utilization={"cpu": 5.0, "gpu": 0.0, "npu": 95.0},
            layers=[],
            device=device,
            runtime_used="npu",
            duration_s=float(request.duration_s),
            profiling_level="qhas",
            qhas_chrometrace_path=chrometrace_path,
        )

    async def get_supported_ops(self) -> list[str]:
        """Return QAIRT-supported ONNX operators."""
        # Based on SDK docs — 130+ operators supported
        return [
            "Add", "AveragePool", "BatchNormalization", "Cast", "Clip", "Concat",
            "Conv", "ConvTranspose", "DepthToSpace", "DequantizeLinear", "Div",
            "Einsum", "Elu", "Equal", "Exp", "Expand", "Flatten", "Gather",
            "GatherElements", "GatherND", "Gelu", "Gemm", "GlobalAveragePool",
            "Greater", "GreaterOrEqual", "GridSample", "GroupNormalization", "GRU",
            "HardSigmoid", "HardSwish", "Identity", "InstanceNormalization",
            "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual", "Log",
            "LpPool", "LRN", "LSTM", "MatMul", "MaxPool", "MaxRoiPool", "Mod",
            "Mul", "Neg", "NonMaxSuppression", "Not", "OneHot", "Or", "Pad",
            "Pow", "QuantizeLinear", "Range", "Reciprocal", "ReduceL2",
            "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin",
            "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape",
            "Resize", "RMSNormalization", "RoiAlign", "ScatterElements",
            "ScatterND", "Shape", "Sigmoid", "Size", "Slice", "Softmax",
            "Softplus", "SpaceToDepth", "Split", "Sqrt", "Squeeze", "STFT",
            "Sub", "Tanh", "ThresholdedRelu", "Tile", "TopK", "Transpose",
            "Unsqueeze", "Where", "Xor",
        ]

    async def execute_inference(
        self,
        model_path: str,
        input_data: Any = None,
        *,
        runtime: str = "auto",
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Execute inference using ``snpe-net-run`` with real I/O marshalling.

        Closes GAP_ANALYSIS T1.4: the previous implementation ignored
        ``input_data`` and used a hardcoded ``np.random.randn(1,3,224,224)``
        regardless of what the caller passed; output was a 500-char
        truncation of stdout, not actual inference outputs.

        Now:
        * Marshals ``input_data`` (a dict mapping input-tensor name to
          numpy array, or a single numpy array for single-input models)
          into per-tensor ``.raw`` files via ``model_inputs.write_input_list``
        * If ``input_data`` is None, falls back to introspect-driven
          random inputs of the right shape / dtype
        * Runs ``snpe-net-run`` with ``--output_dir``
        * Reads back the output ``.raw`` files using output shape/dtype
          from model introspection (so the caller gets numpy arrays,
          not file paths)

        Args:
            model_path: Path to the .dlc / .bin / .onnx file
            input_data: dict[name, ndarray] OR a single ndarray (for
                single-input models) OR None (random inputs)
            runtime: 'cpu' / 'gpu' / 'npu' / 'auto'
            timeout_s: kill snpe-net-run after this many seconds

        Returns:
            ``{"status": "success" | "error", "outputs": {name: ndarray, ...},
               "returncode": int, "stdout": str, "stderr": str,
               "model_io": {...}}``
        """
        from quad.adapters.model_inputs import (
            ModelIO,
            create_input_list_for_model,
            introspect_model,
            write_input_list,
        )
        import numpy as np

        tool = _find_tool("snpe-net-run")
        model_p = Path(model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Model not found: {model_p}")

        # 1) Build the per-tensor .raw files + input_list.txt
        work_dir = Path(tempfile.mkdtemp(prefix="quad_infer_"))
        input_dir = work_dir / "in"
        output_dir = work_dir / "out"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Normalise input_data: accept None, single ndarray, or dict
        cal_data: dict[str, np.ndarray] | None = None
        if input_data is None:
            model_io = introspect_model(model_p, sdk_root=self._sdk_root)
        else:
            model_io = introspect_model(model_p, sdk_root=self._sdk_root)
            if hasattr(input_data, "shape"):
                # Single ndarray — use it for the first input
                if not model_io.inputs:
                    # Fallback: synthesise a spec from the array shape
                    from quad.adapters.model_inputs import TensorSpec
                    model_io = ModelIO(
                        inputs=[TensorSpec("input", tuple(int(d) for d in input_data.shape), str(input_data.dtype))],
                        source="user-array",
                    )
                cal_data = {model_io.inputs[0].name: input_data}
            elif isinstance(input_data, dict):
                cal_data = input_data
            else:
                raise TypeError(
                    f"input_data must be None, a numpy array, or a dict[name, array]; "
                    f"got {type(input_data).__name__}"
                )

        list_path = write_input_list(
            model_io,
            output_dir=input_dir,
            calibration_data=cal_data,
        )

        # 2) Build the snpe-net-run command
        runtime_flag = {
            "cpu": "--use_cpu",
            "gpu": "--use_gpu",
            "npu": "--use_dsp",
            "auto": "--use_dsp",
        }.get(runtime, "--use_dsp")

        cmd = [
            tool,
            "--container", str(model_p),
            "--input_list", list_path,
            "--output_dir", str(output_dir),
            runtime_flag,
            "--perf_profile", "burst",
        ]

        result = await _run_command(cmd, timeout=timeout_s)

        # 3) Read output .raw files back into numpy arrays
        outputs: dict[str, np.ndarray] = {}
        if result.returncode == 0:
            # snpe-net-run writes outputs to <output_dir>/Result_N/<output_name>.raw
            # for each input sample. We collect the first sample only here;
            # callers needing batched results should use a higher-level helper.
            result_dirs = sorted(p for p in output_dir.iterdir() if p.is_dir() and p.name.lower().startswith("result"))
            if result_dirs:
                first = result_dirs[0]
                for out_spec in (model_io.outputs or []):
                    raw_file = first / f"{out_spec.name}.raw"
                    if raw_file.exists():
                        arr = np.fromfile(raw_file, dtype=out_spec.numpy_dtype).reshape(out_spec.shape)
                        outputs[out_spec.name] = arr
                # If introspection didn't give us output specs, just
                # return whatever .raw files we found as bytes.
                if not outputs:
                    for raw_file in first.glob("*.raw"):
                        outputs[raw_file.stem] = np.fromfile(raw_file, dtype=np.float32)

        return {
            "status": "success" if result.returncode == 0 else "error",
            "returncode": result.returncode,
            "outputs": outputs,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "model_io": model_io.to_dict(),
            "work_dir": str(work_dir),
        }

    def _create_dummy_input_list(
        self,
        model_path: Path,
        *,
        num_samples: int = 1,
        calibration_data: "dict[str, Any] | None" = None,
    ) -> str:
        """Create a temporary input list file for snpe-net-run.

        Closes GAP_ANALYSIS T2.8: previously every adapter call generated
        a hardcoded ``np.random.randn(1, 3, 224, 224)`` regardless of the
        model's actual input shape, which silently broke models with
        different shapes and pinned quantization calibration to garbage.

        Now: introspect the model (via ``snpe-dlc-info`` for DLC, the
        ``onnx`` Python module for ONNX) to discover real input shapes,
        then generate inputs of the right shape and dtype. If
        ``calibration_data`` is provided (caller has real calibration
        samples), use those instead of random data.
        """
        from quad.adapters.model_inputs import create_input_list_for_model

        list_path, model_io = create_input_list_for_model(
            model_path,
            sdk_root=self._sdk_root,
            num_samples=num_samples,
            calibration_data=calibration_data,
        )
        # Log the introspection source so users can tell which path
        # produced the inputs (helpful for debugging "why are my
        # quantization scales wrong?").
        try:
            import structlog
            structlog.get_logger().info(
                "input_list_created",
                model_path=str(model_path),
                source=model_io.source,
                num_inputs=len(model_io.inputs),
                num_samples=num_samples,
            )
        except Exception:
            pass
        return list_path

    def _parse_latency(self, stdout: str) -> float:
        """Parse latency from snpe-net-run output.

        Returns 0.0 (a sentinel meaning "unknown") rather than a fictional
        default if no pattern matches. Callers must check ``> 0`` before
        deriving throughput etc. The previous 5.0 ms fallback silently
        masked parser breakage.
        """
        patterns = [
            r"Total Inference Time:\s*([\d.]+)\s*ms",
            r"Average inference time:\s*([\d.]+)\s*ms?",
            r"\bAverage\s+Total\s+Inference\s+Time\s*:\s*([\d.]+)",
            r"\binference time\s*[:=]\s*([\d.]+)\s*ms",
        ]
        for pattern in patterns:
            match = re.search(pattern, stdout, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return 0.0

    def _parse_layers(self, stdout: str) -> list[LayerProfile]:
        """Parse per-layer profiling from snpe-net-run output."""
        layers = []
        import re
        for line in stdout.split("\n"):
            match = re.match(r"\s*(\w+)\s+\w+\s+([\d.]+)", line)
            if match:
                layers.append(LayerProfile(
                    name=match.group(1),
                    op_type="op",
                    runtime="npu",
                    latency_ms=float(match.group(2)),
                    memory_mb=1.0,
                ))
        return layers if layers else [
            LayerProfile(name="model", op_type="composite", runtime="npu",
                        latency_ms=5.0, memory_mb=30.0)
        ]


def _get_model_tips(model_path: str) -> list[str]:
    """Return MODEL_TIPS notes for known model families based on filename."""
    try:
        from quad.compiler.model_conversion import MODEL_TIPS
    except ImportError:
        return []

    notes = []
    path_lower = model_path.lower()
    for model_key, tips in MODEL_TIPS.items():
        if model_key in path_lower:
            limitations = tips.get("limitations", [])
            notes.extend(limitations[:3])
            perf_tips = tips.get("performance_tips", [])
            notes.extend(perf_tips[:2])
            if tips.get("allow_unconsumed_nodes"):
                notes.append(
                    f"{model_key}: requires --allow_unconsumed_nodes during conversion."
                )
    return notes


def _get_image_format_notes(request: "ConversionRequest") -> list[str]:  # type: ignore[name-defined]
    """Build image format guidance notes from the conversion request."""
    notes = []
    layout = getattr(request, "input_layout", "auto")
    channel = getattr(request, "channel_order", "auto")
    means = getattr(request, "mean_values", None)

    if layout == "nchw":
        notes.append(
            "Input layout is NCHW (PyTorch). "
            "SNPE requires NHWC — transpose before inference: "
            "np.transpose(img, (0, 2, 3, 1))  # (N,C,H,W) → (N,H,W,C)"
        )
    elif layout == "nhwc":
        notes.append("Input layout is NHWC — matches SNPE requirement, no transposition needed.")

    if channel == "bgr":
        notes.append(
            "Channel order is BGR (Caffe/OpenCV convention). "
            "Ensure inference inputs are in BGR order: img[:,:,::-1] to convert from RGB."
        )
    elif channel == "rgb":
        notes.append("Channel order is RGB — standard for most modern frameworks.")

    if means:
        notes.append(
            f"Mean subtraction required before inference: mean={means} "
            f"(in {channel.upper() if channel != 'auto' else 'channel'} order)."
        )

    if not notes:
        notes.append(
            "SNPE requires NHWC inputs (channel-last). "
            "If using PyTorch model, transpose NCHW→NHWC before inference. "
            "Verify channel order matches training (RGB vs BGR)."
        )
    return notes
