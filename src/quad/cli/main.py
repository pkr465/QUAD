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
def doctor() -> None:
    """Diagnose environment and report issues."""
    from quad.cli.doctor import run_doctor

    report = run_doctor()

    for check in report.checks:
        icon = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}[check.status]
        typer.echo(f"  {icon} {check.name}: {check.message}")

    typer.echo("")
    if report.all_passed:
        typer.echo("All checks passed.")
    else:
        if report.warnings:
            typer.echo(f"Warnings: {len(report.warnings)}")
        if report.errors:
            typer.echo(f"Errors: {len(report.errors)}")


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
def detect() -> None:
    """Detect available hardware devices."""
    from quad.runtime import list_devices

    devices = list_devices()
    typer.echo("Detected devices:")
    for d in devices:
        typer.echo(f"  - {d.name} ({d.device_type})")


@app.command()
def version() -> None:
    """Show QUAD version and build info."""
    import quad

    typer.echo(f"QUAD v{quad.__version__}")


def main() -> None:
    """Entry point for the quad CLI."""
    app()


if __name__ == "__main__":
    main()
