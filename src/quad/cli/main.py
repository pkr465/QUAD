"""QUAD CLI — Unified command-line interface for the QUAD toolchain.

Usage:
    quad configure      # Interactive configuration wizard (run first!)
    quad quickstart     # Interactive getting-started wizard
    quad doctor         # Diagnose environment
    quad benchmark      # Run standard benchmarks
    quad compile        # Compile model
    quad optimize       # Optimize model
    quad profile        # Profile workload
    quad serve          # Start inference server
    quad detect         # Detect hardware
    quad version        # Show version info
"""

from __future__ import annotations

from typing import Optional

import typer

app = typer.Typer(
    name="quad",
    help="QUAD — Qualcomm Unified AI Developer toolchain.",
    no_args_is_help=True,
)


@app.command()
def configure(
    config: str = typer.Option("quad.toml", help="Path to quad.toml"),
    env: str = typer.Option(".env", help="Path to .env file"),
    non_interactive: bool = typer.Option(False, "--yes", "-y", help="Use defaults (non-interactive)"),
) -> None:
    """Interactive configuration wizard — writes quad.toml and .env.

    Run this first when setting up QUAD or changing hardware/SDK paths.
    Asks questions about your SDK installation, target device, and API keys.
    """
    from pathlib import Path
    from quad.cli.configure import run_configure
    run_configure(Path(config), Path(env), non_interactive)


@app.command()
def quickstart() -> None:
    """Interactive getting-started wizard."""
    from quad.cli.quickstart import run_quickstart

    result = run_quickstart()
    typer.echo(f"\nQuickstart completed in {result.total_time_s:.1f}s")
    typer.echo(f"  Device detected: {result.device_detected}")
    typer.echo(f"  Model compiled:  {result.model_compiled}")
    typer.echo(f"  Profile created: {result.profile_generated}")
    typer.echo(f"  Code generated:  {result.code_generated}")


@app.command()
def doctor(
    real_mode: bool = typer.Option(
        False,
        "--real-mode",
        help="Strict pre-flight for real-hardware mode. Exits non-zero if any "
        "SDK/runtime check fails or warns. Use this in CI before running on hardware.",
    ),
) -> None:
    """Diagnose environment and report issues.

    Use ``--real-mode`` for a strict pre-flight check before running on
    physical hardware: warnings about missing SDK env vars, missing CLI
    tools, or missing DSP libraries are escalated to errors and the
    process exits with status 1.
    """
    from quad.cli.doctor import run_doctor

    report = run_doctor(real_mode=real_mode)

    for check in report.checks:
        icon = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}[check.status]
        typer.echo(f"  {icon} {check.name}: {check.message}")

    typer.echo("")
    if report.all_passed:
        typer.echo("All checks passed.")
        return

    if report.warnings:
        typer.echo(f"Warnings: {len(report.warnings)}")
    if report.errors:
        typer.echo(f"Errors: {len(report.errors)}")

    if real_mode and (report.errors or report.warnings):
        typer.echo("\nReal-mode pre-flight failed. Fix the issues above before running on hardware.")
        raise typer.Exit(code=1)


@app.command()
def mode(
    set_to: Optional[str] = typer.Option(
        None,
        "--set",
        help="Set adapter mode for the current shell: 'mock' or 'real'. "
        "Prints the export command — wrap in $(quad mode --set real) to apply.",
    ),
) -> None:
    """Show the active adapter mode and whether real-hardware mode is ready.

    Reads ``adapter_mode`` from quad.toml and ``QUAD_ADAPTER_MODE`` from
    the environment, then reports whether real mode would actually work
    (i.e. whether the SDK root + tools are reachable).
    """
    if set_to is not None:
        if set_to not in {"mock", "real"}:
            typer.echo(f"Invalid mode {set_to!r}. Must be 'mock' or 'real'.", err=True)
            raise typer.Exit(code=2)
        # Print as a shell-evalable export so users can do:
        #   eval "$(quad mode --set real)"
        typer.echo(f"export QUAD_ADAPTER_MODE={set_to}")
        return

    from quad.adapters.factory import AdapterFactory
    from quad.config import load_config
    from quad.sdk_manager import resolve_sdk_root, apply_to_environment

    cfg = load_config()

    # Run SDK discovery so env vars are populated before checking readiness.
    # This mirrors what the MCP server does at startup, so `quad mode`
    # reports the same answer the server would see.
    sdk = resolve_sdk_root()
    if sdk is not None:
        apply_to_environment(sdk)

    factory = AdapterFactory(cfg)

    typer.echo(f"adapter_mode:    {factory.mode}")
    typer.echo(f"strict:          {factory.strict}")
    if sdk is not None:
        typer.echo(f"sdk:             {sdk.flavor} {sdk.version}  ({sdk.source})")
    else:
        typer.echo(f"sdk:             (none — run `quad sdk status` for guidance)")
    ready, reason = factory.real_mode_ready()
    status = "READY" if ready else "NOT READY"
    typer.echo(f"real-mode:       {status}")
    typer.echo(f"  reason:        {reason}")
    typer.echo("")
    if not ready and factory.mode == "real":
        typer.echo("Hint: run `quad doctor --real-mode` for a full pre-flight.")
    elif factory.mode == "mock":
        typer.echo("Hint: set QUAD_ADAPTER_MODE=real (or adapter_mode=\"real\" in quad.toml) to enable hardware.")


@app.command()
def benchmark(
    device: str = typer.Option("auto", help="Target device (auto, npu, gpu, cpu)"),
    models: Optional[list[str]] = typer.Option(None, help="Specific models to benchmark"),
) -> None:
    """Run standard benchmark suite."""
    from quad.cli.benchmark import run_benchmark

    report = run_benchmark(device=device, models=models)

    typer.echo(f"\nBenchmark Report — Device: {report.device} @ {report.timestamp}")
    typer.echo("-" * 70)
    typer.echo(f"{'Model':<20} {'Latency(ms)':<14} {'Throughput(FPS)':<17} {'Power(mW)':<12} {'vs Baseline'}")
    typer.echo("-" * 70)
    for r in report.results:
        typer.echo(
            f"{r.model_name:<20} {r.latency_ms:<14.2f} {r.throughput_fps:<17.1f} "
            f"{r.power_mw:<12.0f} {r.vs_baseline_pct:+.1f}%"
        )


@app.command()
def compile(
    model_path: str = typer.Argument(..., help="Path to model file (.onnx, .pt)"),
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output .qbin path"),
    targets: str = typer.Option(
        "all",
        help="Comma-separated target list (e.g. 'qnpu_v3,qdsp_v66') or 'all'",
    ),
    portable: bool = typer.Option(
        False, "--portable", help="IR-only build; JIT each target at load time"
    ),
    quantization: str = typer.Option(
        "fp32", help="Quantization scheme: fp32 | int8 | int4"
    ),
    backend: str = typer.Option(
        "auto", help="Backend: auto (default) | qairt | stub"
    ),
    coverage_only: bool = typer.Option(
        False, "--coverage-only", help="IR + per-target op-coverage report only"
    ),
) -> None:
    """Compile a model to QUAD binary format."""
    from quad.compiler.pipeline import compile_model

    target_list: list[str] | str
    if targets == "all":
        target_list = "all"
    else:
        target_list = [t.strip() for t in targets.split(",") if t.strip()]

    qbin = compile_model(
        model_path=model_path,
        output_path=output,
        targets=target_list,  # type: ignore[arg-type]
        portable=portable,
        backend=backend,  # type: ignore[arg-type]
        quantization=quantization,  # type: ignore[arg-type]
        coverage_only=coverage_only,
    )
    typer.echo(f"Compiled: {qbin.path}")


@app.command()
def optimize(
    model_path: str = typer.Argument(..., help="Path to model or .qbin"),
    target: str = typer.Option("qnpu_v3", help="Target capability (e.g. qnpu_v3)"),
    quantization: str = typer.Option("int8", help="Quantization: fp32 | int8 | int4 | none"),
    power_budget_mw: float | None = typer.Option(
        None, "--power-budget", help="Power budget in milliwatts"
    ),
) -> None:
    """Run the graph-optimization pipeline on a model."""
    from quad.optimizer import optimize_model

    result = optimize_model(
        model_path=model_path,
        target=target,
        quantization=quantization,
        power_budget_mw=power_budget_mw,
    )
    typer.echo(
        f"Nodes: {result.original_nodes} -> {result.optimized_nodes} "
        f"({len(result.passes_applied)} passes)"
    )
    typer.echo(f"Estimated speedup:        {result.estimated_speedup:.2f}x")
    typer.echo(f"Estimated power savings:  {result.estimated_power_reduction_pct:.0f}%")
    typer.echo(f"Quantization applied:     {result.quantization_applied}")
    typer.echo(f"Passes:                   {', '.join(result.passes_applied)}")


@app.command()
def profile(
    model_path: str = typer.Argument(..., help="Path to compiled model (.qbin / .dlc / .onnx)"),
    level: str = typer.Option("kernel", help="Profile depth: system | kernel | deep"),
    device: str = typer.Option("npu", help="Target device: npu | gpu | cpu"),
) -> None:
    """Profile a model — system / kernel / deep (power + memory)."""
    from quad.profiler import profile_model

    summary = profile_model(model_path=model_path, level=level, device=device)  # type: ignore[arg-type]
    typer.echo(
        f"Profile depth: {level}  device: {device}  "
        f"duration: {summary.profile_duration_ms} ms"
    )
    if summary.kernel_report is not None:
        top = (
            summary.kernel_report.top_kernels(5)
            if hasattr(summary.kernel_report, "top_kernels")
            else []
        )
        if top:
            typer.echo("Top kernels:")
            for k in top:
                typer.echo(f"  {k.name}: {k.latency_us}us ({k.bottleneck})")
    if level == "deep" and summary.power_trace is not None:
        typer.echo(f"Average power: {summary.power_trace.avg_power_mw:.0f} mW")


@app.command()
def serve(
    model_path: str = typer.Argument(..., help="Path to compiled model (.qbin / .bin / .dlc)"),
    name: str = typer.Option("default", help="Model name for the /infer endpoint"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, help="Bind port"),
    device: str = typer.Option("npu", help="Device for the model (cpu/gpu/npu)"),
) -> None:
    """Start an HTTP inference server for a compiled model.

    Closes GAP_ANALYSIS T1.2: ``quad serve`` previously only printed
    a message — now it spins up a real FastAPI app via uvicorn with
    /infer, /health, /metrics, /models endpoints.

    Requires the [real] extras: ``pip install -e .[real]`` for fastapi
    + uvicorn (or just ``pip install fastapi uvicorn``).
    """
    try:
        from quad.serve.http import start_http
        from quad.serve.server import ModelServer
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Loading {model_path} as model '{name}' on {device}…")
    server = ModelServer.from_env(host=host, port=port)
    server.start()
    server.load_model(name, model_path, device=device)
    typer.echo(f"Runtime: {server._runtime} "
               + ("(real inference via QAIRT)" if server._runtime == "qairt"
                  else "(mock — set QUAD_SERVE_RUNTIME=qairt for real inference)"))

    typer.echo(f"Serving on http://{host}:{port}")
    typer.echo("Endpoints:")
    typer.echo("  POST /infer           - run inference")
    typer.echo("  POST /infer/batch     - batch inference")
    typer.echo("  GET  /health          - liveness")
    typer.echo("  GET  /metrics         - perf stats")
    typer.echo("  GET  /models          - list loaded models")
    typer.echo("Press Ctrl+C to stop.")

    try:
        start_http(server, host=host, port=port)
    except ImportError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    except KeyboardInterrupt:
        typer.echo("\nShutting down…")
        server.stop()


@app.command()
def detect(
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-probe local hardware (skip the discovery cache)"
    ),
) -> None:
    """Detect available hardware devices on the local machine.

    Performs a real OS-level probe (PowerShell on Windows, /proc on
    Linux, sysctl on macOS, adb on Android when ANDROID_SERIAL is set)
    and reports the actual CPU / GPU / NPU present, not a hardcoded list.
    """
    from quad.runtime import list_devices
    from quad.runtime.host_probe import probe_host

    info = probe_host()
    devices = list_devices(refresh=refresh)

    typer.echo("")
    typer.echo("=== Host probe ===")
    typer.echo(f"  OS:           {info.os_name or 'unknown'} ({info.os_arch or 'unknown'})")
    if info.cpu_name:
        typer.echo(
            f"  CPU:          {info.cpu_name} "
            f"({info.cpu_cores or '?'} cores"
            + (f" / {info.cpu_threads} threads" if info.cpu_threads != info.cpu_cores else "")
            + (f" @ {info.cpu_max_mhz} MHz" if info.cpu_max_mhz else "")
            + ")"
        )
    if info.gpu_name:
        typer.echo(f"  GPU:          {info.gpu_name}")
    if info.npu_name:
        typer.echo(f"  NPU:          {info.npu_name}")
    if info.ram_gb:
        typer.echo(f"  RAM:          {info.ram_gb} GB")
    typer.echo(f"  Probe source: {info.source}")

    typer.echo("")
    typer.echo("=== Compute units ===")
    for d in devices:
        # Device exposes .type (not .device_type) — this previously crashed
        marker = {"npu": "[NPU]", "gpu": "[GPU]", "cpu": "[CPU]"}.get(d.type, f"[{d.type}]")
        if d.is_npu:
            typer.echo(f"  {marker}  {d.name}  ({d.tops} TOPS, {d.memory_mb // 1024} GB)")
        elif d.is_gpu:
            typer.echo(f"  {marker}  {d.name}  ({d.tflops} TFLOPS, {d.memory_mb // 1024} GB)")
        else:
            freq = f", {d.cores * 1.0 if not d.cores else 0} GHz" if 0 else ""
            typer.echo(f"  {marker}  {d.name}  ({d.cores} cores, {d.memory_mb // 1024} GB)")

    typer.echo("")
    if info.is_qualcomm:
        typer.echo("Qualcomm hardware detected. Real-mode workflows are supported on this machine.")
        typer.echo("Tip: run `quad mode` to confirm the SDK is wired, or `quad doctor --real-mode` for a full pre-flight.")
    else:
        typer.echo("Note: this doesn't look like Qualcomm hardware. QUAD will run in mock mode by default;")
        typer.echo("      mock mode is fully functional for development without real silicon.")


@app.command()
def version() -> None:
    """Show QUAD version and build info."""
    import quad

    typer.echo(f"QUAD v{quad.__version__}")


sdk_app = typer.Typer(
    name="sdk",
    help="Manage the Qualcomm AI Runtime SDK (QAIRT/SNPE) used by QUAD.",
    no_args_is_help=True,
)
app.add_typer(sdk_app)


# ── quad models — production ONNX provisioning ───────────────────────────

models_app = typer.Typer(
    name="models",
    help="Provision production ONNX models for plans. "
         "Sources are declared in src/quad/model_registry/registry.yaml.",
    no_args_is_help=True,
)
app.add_typer(models_app)


@models_app.command("list")
def models_list(plan: Optional[str] = typer.Option(None, "--plan", help="Filter by plan id")) -> None:
    """List every model in the registry, with cache state per entry."""
    from quad.model_registry import list_for_plan, list_models
    from quad.model_registry.fetcher import _cached_path  # type: ignore[attr-defined]
    import os

    entries = list_for_plan(plan) if plan else list_models()
    if not entries:
        typer.echo(f"No models found{' for plan ' + plan if plan else ''}.")
        return

    typer.echo(f"{'name':<26} {'plan':<8} {'src':<8} {'state':<10} {'size_mb':>8}  description")
    typer.echo("-" * 110)
    for e in entries:
        if e.url:
            src = "url"
            cached = _cached_path(e)
            state = "cached" if cached.exists() else "missing"
        else:
            src = "env"
            state = "set" if os.environ.get(e.path_env_var or "") else "unset"
        typer.echo(f"{e.name:<26} {e.plan:<8} {src:<8} {state:<10} {e.size_mb:>8.1f}  {e.description}")


@models_app.command("fetch")
def models_fetch(
    name: str = typer.Argument(..., help="Registry entry name (see `quad models list`)"),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached"),
) -> None:
    """Download (or resolve) a registered model and print its absolute path."""
    from quad.model_registry import fetch_model, ModelFetchError

    def _progress(written: int, total: int) -> None:
        if total:
            pct = written / total * 100
            typer.echo(f"  {written/1_048_576:7.2f} MB / {total/1_048_576:7.2f} MB ({pct:5.1f}%)\r", nl=False)
        else:
            typer.echo(f"  {written/1_048_576:7.2f} MB\r", nl=False)

    try:
        path = fetch_model(name, force=force, progress=_progress)
    except ModelFetchError as exc:
        typer.echo(f"\n[error] {exc}", err=True)
        raise typer.Exit(code=2)
    except KeyError as exc:
        typer.echo(f"\n[error] {exc}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"\n{path}")


@models_app.command("path")
def models_path(name: str = typer.Argument(...)) -> None:
    """Print the local path the registry will use for a model (no download)."""
    from quad.model_registry import resolve_model_path, ModelFetchError

    try:
        typer.echo(str(resolve_model_path(name)))
    except ModelFetchError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(code=2)


@models_app.command("verify")
def models_verify(name: str = typer.Argument(...)) -> None:
    """Verify the SHA-256 of a cached entry (no-op for env-var entries)."""
    from quad.model_registry import resolve_entry
    from quad.model_registry.fetcher import _cached_path, _sha256  # type: ignore[attr-defined]

    entry = resolve_entry(name)
    if entry.path_env_var:
        typer.echo(f"{name}: user-supplied (no SHA verification).")
        return
    cached = _cached_path(entry)
    if not cached.exists():
        typer.echo(f"{name}: not cached. Run `quad models fetch {name}`.", err=True)
        raise typer.Exit(code=2)
    actual = _sha256(cached)
    if entry.sha256:
        ok = actual.lower() == entry.sha256.lower()
        typer.echo(f"{name}: SHA-256 = {actual}  [{ 'OK' if ok else 'MISMATCH' }]")
        if not ok:
            raise typer.Exit(code=3)
    else:
        typer.echo(f"{name}: SHA-256 = {actual}  (no expected value in registry)")


@sdk_app.command("status")
def sdk_status() -> None:
    """Show the SDK QUAD will use (or report that none is installed)."""
    from quad.sdk_manager import (
        QAIRT_PRODUCT_URL,
        SNPE_PRODUCT_URL,
        discover_sdks,
        missing_sdk_message,
        resolve_sdk_root,
    )

    info = resolve_sdk_root()
    if info is None:
        typer.echo("No QAIRT/SNPE SDK detected.")
        typer.echo("")
        typer.echo(missing_sdk_message())
        raise typer.Exit(code=1)

    typer.echo(f"Active SDK:")
    typer.echo(f"  flavor:   {info.flavor}")
    typer.echo(f"  version:  {info.version}")
    typer.echo(f"  root:     {info.root}")
    typer.echo(f"  bin:      {info.bin_dir or '(none — SDK has no bin/<arch>/ tools)'}")
    typer.echo(f"  source:   {info.source}")
    typer.echo(
        f"  tools:    qairt-converter={'yes' if info.has_qairt_converter else 'no'}  "
        f"snpe-net-run={'yes' if info.has_snpe_net_run else 'no'}"
    )

    others = discover_sdks()
    if len(others) > 1:
        typer.echo("")
        typer.echo(f"Also found ({len(others) - 1} additional install{'s' if len(others) > 2 else ''}):")
        for o in others[1:]:
            typer.echo(f"  - {o.flavor} {o.version} @ {o.root} [{o.source}]")

    typer.echo("")
    typer.echo(f"Update sources:  QAIRT: {QAIRT_PRODUCT_URL}")
    typer.echo(f"                 SNPE:  {SNPE_PRODUCT_URL}")


@sdk_app.command("discover")
def sdk_discover() -> None:
    """Scan all known locations and list every SDK found."""
    from quad.sdk_manager import DEFAULT_SCAN_PATHS, discover_sdks

    sdks = discover_sdks()
    if not sdks:
        typer.echo("No QAIRT/SNPE SDKs found in any of:")
        for p in DEFAULT_SCAN_PATHS:
            typer.echo(f"  {p}")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(sdks)} SDK install(s):")
    for i, info in enumerate(sdks):
        marker = "*" if i == 0 else " "
        typer.echo(
            f"  {marker} {info.flavor} {info.version}  ({info.source})  {info.root}"
        )
    typer.echo("")
    typer.echo("(* = the one QUAD will use — first match wins)")


client_app = typer.Typer(
    name="client",
    help="Manage IDE/agent client provisioning (.claude/settings.json + skills).",
    no_args_is_help=True,
)
app.add_typer(client_app)


@client_app.command("install")
def client_install(
    client: str = typer.Option("claude_code", help="Which MCP client to install for"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    adapter_mode: str = typer.Option("mock", help="Adapter mode written into settings.json"),
) -> None:
    """Install MCP client config + skills under the current project.

    For Claude Code: writes ``.claude/settings.json`` and copies the
    bundled skill files into ``.claude/skills/``. The MCP server
    itself is not affected.
    """
    from pathlib import Path

    from quad.client import get_provisioner

    try:
        prov = get_provisioner(client)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    result = prov.install(Path.cwd(), force=force, adapter_mode=adapter_mode)
    typer.echo(f"Provisioned {result.client} client at {result.settings_path}")
    typer.echo(f"  Skills:    {result.skills_dir}")
    typer.echo(f"  Written:   {len(result.files_written)} file(s)")
    if result.files_skipped:
        typer.echo(f"  Skipped:   {len(result.files_skipped)} file(s) (use --force to overwrite)")
    for note in result.notes:
        typer.echo(f"  Note: {note}")


@client_app.command("uninstall")
def client_uninstall(
    client: str = typer.Option("claude_code", help="Which MCP client to uninstall"),
) -> None:
    """Remove the bundled client config + skills."""
    from pathlib import Path

    from quad.client import get_provisioner

    try:
        prov = get_provisioner(client)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    result = prov.uninstall(Path.cwd())
    typer.echo(f"Uninstalled {result.client} from {result.skills_dir}")
    for note in result.notes:
        typer.echo(f"  {note}")


@client_app.command("status")
def client_status(
    client: str = typer.Option("claude_code", help="Which MCP client to check"),
) -> None:
    """Show what's currently installed for a given client."""
    from pathlib import Path

    from quad.client import get_provisioner

    try:
        prov = get_provisioner(client)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    s = prov.status(Path.cwd())
    typer.echo(f"Client:           {s['client']}")
    typer.echo(f"settings.json:    {'present' if s['settings_exists'] else 'missing'} @ {s['settings_path']}")
    typer.echo(f"Skills dir:       {'present' if s['skills_dir_exists'] else 'missing'} @ {s['skills_dir']}")
    typer.echo(f"Bundled skills:   {s['bundled_skill_count']}")
    typer.echo(f"Installed skills: {s['installed_skill_count']}")
    if s["missing_skills"]:
        typer.echo(f"Missing:          {', '.join(s['missing_skills'])}")
    if s["extra_user_skills"]:
        typer.echo(f"User-added:       {', '.join(s['extra_user_skills'])}")


@client_app.command("preview")
def client_preview(
    client: str = typer.Option("claude_code", help="Which MCP client config to preview"),
    adapter_mode: str = typer.Option("mock", help="Adapter mode token to render"),
) -> None:
    """Print the settings.json content that ``install`` would write."""
    if client != "claude_code":
        typer.echo("preview is currently only supported for client=claude_code", err=True)
        raise typer.Exit(code=2)

    from quad.client.claude_code import ClaudeCodeProvisioner

    prov = ClaudeCodeProvisioner()
    typer.echo(prov.render_settings_preview(adapter_mode=adapter_mode))


@sdk_app.command("install")
def sdk_install(
    archive: str = typer.Argument(
        ..., help="Path to a downloaded QAIRT/SNPE archive (.zip / .tar.gz / .tgz)"
    ),
    target: Optional[str] = typer.Option(
        None, "--target", help="Override extraction directory (default: ./sdks/<archive-stem>)"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace target directory if it already exists"
    ),
) -> None:
    """Unpack a downloaded QAIRT/SNPE archive into ./sdks/.

    Download the archive yourself (Qualcomm requires a developer-account
    login + EULA acceptance — there is no public direct link), then run:

        quad sdk install ~/Downloads/qairt-2.45.0.260326.zip

    The unpacked SDK is gitignored and auto-detected on next server start.
    """
    from quad.sdk_manager import install_archive

    try:
        result = install_archive(archive, target_dir=target, overwrite=overwrite)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)
    except FileExistsError as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo("Pass --overwrite to replace the existing directory.", err=True)
        raise typer.Exit(code=2)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=2)

    typer.echo(f"Installed {result.flavor} {result.version} at:")
    typer.echo(f"  {result.target_dir}")
    typer.echo(
        f"Extracted {result.files_extracted} files "
        f"({result.bytes_extracted / 1024 / 1024:.1f} MB)"
    )
    typer.echo("")
    typer.echo("Next: run `quad sdk status` to confirm, then `quad mode` to flip to real mode.")


def main() -> None:
    """Entry point for the quad CLI."""
    # On Windows the default console code page is cp1252; the help text
    # and output use em-dashes and other non-ASCII glyphs. Reconfigure
    # stdio to UTF-8 with 'replace' so we degrade to '?' on legacy
    # terminals instead of crashing.
    import contextlib
    import sys
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    app()


if __name__ == "__main__":
    main()
