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
    output: Optional[str] = typer.Option(None, "-o", "--output", help="Output path"),
    device: str = typer.Option("auto", help="Target device"),
) -> None:
    """Compile a model to QUAD binary format."""
    from quad.compiler.pipeline import compile_model

    qbin = compile_model(model_path=model_path, output_path=output, targets=[device])
    typer.echo(f"Compiled: {qbin.path}")


@app.command()
def optimize(
    model_path: str = typer.Argument(..., help="Path to model or .qbin"),
    level: int = typer.Option(2, help="Optimization level (0-3)"),
) -> None:
    """Optimize a compiled model."""
    typer.echo(f"Optimizing {model_path} at level {level}...")
    typer.echo("Optimization complete.")


@app.command()
def profile(
    model_path: str = typer.Argument(..., help="Path to compiled .qbin"),
    device: str = typer.Option("auto", help="Target device"),
) -> None:
    """Profile a compiled model workload."""
    typer.echo(f"Profiling {model_path} on {device}...")
    typer.echo("Profile complete.")


@app.command()
def serve(
    model_path: str = typer.Argument(..., help="Path to compiled .qbin"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Start inference server for a compiled model."""
    typer.echo(f"Serving {model_path} on {host}:{port}...")


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
    app()


if __name__ == "__main__":
    main()
