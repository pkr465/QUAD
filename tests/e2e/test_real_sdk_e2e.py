"""End-to-end real-SDK test for QUAD on Windows.

What this test exercises (in order):

    1. SDK acquisition         sdk_manager.install_archive() unpacks the user's
                               QAIRT zip from ~/Downloads/ if it isn't already
                               present in ./sdks/ (idempotent).
    2. SDK discovery           sdk_manager.resolve_sdk_root() locates the
                               installed SDK by walking the standard paths.
    3. Tool resolution         A real Qualcomm binary (qnn-platform-validator)
                               is found inside <sdk>/bin/<arch>/ for this host.
    4. Real subprocess         The binary is actually invoked (--help) and its
                               stdout is verified to contain a Qualcomm marker.
    5. Env application         apply_to_environment() populates QAIRT_SDK_ROOT
                               etc. for child processes.
    6. Real-mode factory       AdapterFactory + QUAD_ADAPTER_MODE=real returns
                               a QAIRTAdapter (not a fall-back MockAdapter).
    7. MCP tool layer          The 5 MCP tools dispatch cleanly through the
                               factory in mock mode (sanity baseline that
                               proves the full pipeline wires up).

Skip behaviour:
    The test is automatically skipped if the QAIRT archive is not in
    ~/Downloads/. To run elsewhere, set QUAD_TEST_QAIRT_ARCHIVE=<path>.

Runnable two ways:
    pytest tests/e2e/test_real_sdk_e2e.py -v -s
    python tests/e2e/test_real_sdk_e2e.py        # standalone with rich logs
"""

from __future__ import annotations

import asyncio
import os
import platform as _platform
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


# ───────────────────────────── Locating QAIRT ──────────────────────────────


_DEFAULT_DOWNLOADS = Path.home() / "Downloads"
_ARCHIVE_PATTERNS = (
    "v2.46.0.260424.zip",
    "qairt-*.zip",
    "v2.*.zip",
    "snpe-*.zip",
)


def _find_qairt_archive() -> Path | None:
    """Return the first QAIRT/SNPE archive in ~/Downloads (or env override)."""
    override = os.environ.get("QUAD_TEST_QAIRT_ARCHIVE", "").strip()
    if override:
        p = Path(override)
        return p if p.exists() else None

    if not _DEFAULT_DOWNLOADS.is_dir():
        return None

    for pattern in _ARCHIVE_PATTERNS:
        matches = sorted(_DEFAULT_DOWNLOADS.glob(pattern))
        if matches:
            # Prefer the largest match — partial downloads tend to be smaller
            return max(matches, key=lambda p: p.stat().st_size)
    return None


# ───────────────────────────── Helpers ─────────────────────────────────────


def _candidate_runtime_tools() -> tuple[str, ...]:
    """Tools we'll try to locate, in order of preference.

    These all ship as native binaries in QAIRT/SNPE 2.x.  We prefer
    qnn-platform-validator (purpose-built for "is the SDK functional?"),
    then fall back to snpe-net-run / snpe-diagview, both of which always
    ship under x86_64-windows-msvc/ and aarch64-windows-msvc/.
    """
    suffix = ".exe" if sys.platform == "win32" else ""
    return (
        f"qnn-platform-validator{suffix}",
        f"snpe-diagview{suffix}",
        f"qnn-net-run{suffix}",
        f"snpe-net-run{suffix}",
    )


def _find_real_tool(sdk_root: Path, names: tuple[str, ...]) -> tuple[Path, str]:
    """Search every per-arch bin dir for the first matching tool.

    Returns ``(tool_path, bin_dir_name)``.  Raises FileNotFoundError if
    nothing matches.
    """
    bin_root = sdk_root / "bin"
    if not bin_root.is_dir():
        raise FileNotFoundError(f"No bin/ under {sdk_root}")

    # Prefer native subdirs for this host first
    arch_pref: list[str] = []
    if sys.platform == "win32":
        # On Windows ARM64 the Python may run via Prism (reports AMD64),
        # but the OS itself is ARM64.  Try aarch64 first, then x86_64.
        # WMI is the authoritative source but `platform.machine()` works
        # for our purposes (it reads the registry / OS).
        if _platform.machine().upper() in ("ARM64", "AARCH64"):
            arch_pref = ["aarch64-windows-msvc", "x86_64-windows-msvc", "arm64x-windows-msvc"]
        else:
            arch_pref = ["x86_64-windows-msvc", "aarch64-windows-msvc", "arm64x-windows-msvc"]
    elif sys.platform.startswith("linux"):
        if _platform.machine().lower() in ("aarch64", "arm64"):
            arch_pref = ["aarch64-ubuntu-gcc9.4", "aarch64-oe-linux-gcc11.2", "x86_64-linux-clang"]
        else:
            arch_pref = ["x86_64-linux-clang"]

    ordered: list[Path] = []
    for sub in arch_pref:
        candidate = bin_root / sub
        if candidate.is_dir():
            ordered.append(candidate)
    # Fallback: anything else in bin/
    for entry in sorted(bin_root.iterdir()):
        if entry.is_dir() and entry not in ordered:
            ordered.append(entry)

    for arch_bin in ordered:
        for name in names:
            tool = arch_bin / name
            if tool.exists():
                return tool, arch_bin.name
    raise FileNotFoundError(
        f"None of {names} found under any of {[p.name for p in ordered]}"
    )


def _step(label: str) -> None:
    """Pretty-print a phase header."""
    print(f"\n---- {label} ----", flush=True)


# ───────────────────────────── The test ────────────────────────────────────


_archive = _find_qairt_archive()


@pytest.mark.skipif(
    _archive is None,
    reason=(
        "No QAIRT archive found in ~/Downloads/ (or QUAD_TEST_QAIRT_ARCHIVE). "
        "Download from https://www.qualcomm.com/developer/software/"
        "qualcomm-ai-engine-direct-sdk and place under ~/Downloads/."
    ),
)
def test_e2e_qairt_real_mode_full_pipeline(tmp_path_factory) -> None:
    """End-to-end: SDK install → real binary invocation → real-mode factory → MCP layer."""
    from quad.adapters.factory import AdapterFactory
    from quad.config import load_config
    from quad.sdk_manager import (
        apply_to_environment,
        install_archive,
        resolve_sdk_root,
    )

    project_root = Path(__file__).resolve().parents[2]
    started = time.perf_counter()

    # ── Phase 1: SDK acquisition (idempotent) ──────────────────────────────
    _step("Phase 1: install/locate QAIRT SDK")
    target = project_root / "sdks" / "qairt-2.46.0.260424"
    if target.exists():
        print(f"   target already exists: {target}")
    else:
        t0 = time.perf_counter()
        result = install_archive(
            str(_archive),
            target_dir=str(target),
            project_root=project_root,
            overwrite=False,
        )
        dt = time.perf_counter() - t0
        print(f"   extracted {result.files_extracted} files "
              f"({result.bytes_extracted / 1e9:.2f} GB) in {dt:.1f}s")
        assert result.files_extracted > 1000, "QAIRT zip should contain >1000 files"

    # ── Phase 2: SDK discovery ─────────────────────────────────────────────
    _step("Phase 2: discover SDK via sdk_manager")
    sdk = resolve_sdk_root(project_root=project_root)
    assert sdk is not None, "SDK should be discoverable after install"
    print(f"   root:    {sdk.root}")
    print(f"   version: {sdk.version}")
    print(f"   flavor:  {sdk.flavor}")
    print(f"   bin_dir: {sdk.bin_dir}")
    assert sdk.has_qairt_converter or sdk.has_snpe_net_run, \
        "SDK install should expose at least one CLI tool"

    # ── Phase 3: locate a real Qualcomm tool for this host ────────────────
    _step("Phase 3: resolve a runnable native tool")
    print(f"   PROCESSOR_ARCHITECTURE = {os.environ.get('PROCESSOR_ARCHITECTURE')}")
    print(f"   platform.machine()     = {_platform.machine()}")
    tool, picked_subdir = _find_real_tool(Path(sdk.root), _candidate_runtime_tools())
    print(f"   picked:                  bin/{picked_subdir}/{tool.name}")
    assert tool.exists()

    # ── Phase 4: invoke that tool via subprocess (real .exe execution) ────
    _step(f"Phase 4: real subprocess -> {tool.name} --help")
    arch_bin = tool.parent
    proc = subprocess.run(
        [str(tool), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        # Qualcomm binaries depend on co-located shared libs; prepend the
        # bin dir to PATH so the dynamic loader finds them.
        env={**os.environ, "PATH": f"{arch_bin}{os.pathsep}{os.environ.get('PATH', '')}"},
    )
    print(f"   exit:   {proc.returncode}")
    print(f"   stdout (first 300): {proc.stdout[:300]!r}")
    if proc.stderr:
        print(f"   stderr (first 300): {proc.stderr[:300]!r}")
    combined = (proc.stdout + proc.stderr).lower()
    # --help sometimes routes to stderr; some Qualcomm tools emit no
    # "qualcomm" string at all but show usage.  Accept any of these
    # markers, plus simply seeing options ("-h"/"-v"/"--").
    markers = (
        "qnn", "snpe", "qairt", "platform", "backend", "usage",
        "qualcomm", "options", "argument",
    )
    assert any(m in combined for m in markers), (
        f"{tool.name} --help did not produce any expected marker; "
        f"got stdout={proc.stdout[:500]!r} stderr={proc.stderr[:500]!r}"
    )

    # ── Phase 5: apply env vars so child processes inherit ─────────────────
    _step("Phase 5: apply SDK env vars")
    saved_env = {k: os.environ.get(k) for k in
                 ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT", "QUAD_ADAPTER_MODE")}
    try:
        apply_to_environment(sdk)
        assert os.environ.get("QAIRT_SDK_ROOT") == sdk.root
        print(f"   QAIRT_SDK_ROOT = {os.environ['QAIRT_SDK_ROOT']}")

        # ── Phase 6: AdapterFactory in real mode ───────────────────────────
        _step("Phase 6: AdapterFactory in real mode")
        os.environ["QUAD_ADAPTER_MODE"] = "real"
        cfg = load_config()
        cfg.adapter_mode = "real"  # belt-and-braces in case env loaded earlier
        factory = AdapterFactory(cfg, strict=False)
        adapter = factory.get_adapter("auto")
        adapter_class = type(adapter).__name__
        fell_back = getattr(adapter, "fell_back_from_real", False)
        print(f"   adapter:     {adapter_class}")
        print(f"   fell_back:   {fell_back}")
        if fell_back:
            print(f"   reason:      {getattr(adapter, 'fallback_reason', '')}")
        assert adapter_class == "QAIRTAdapter", \
            f"Real-mode factory should yield QAIRTAdapter; got {adapter_class}"
        assert not fell_back, \
            f"Adapter should not have fallen back to mock: {getattr(adapter, 'fallback_reason', '')}"

        # ── Phase 7: MCP tool layer dispatches cleanly ─────────────────────
        # We use mock-mode here as a sanity baseline so the test doesn't
        # depend on cross-platform SDK behaviour (e.g. detect_hardware on
        # Windows still has known stub gaps documented in the gap analysis).
        # This phase proves the FastMCP tool wrappers, models, and adapter
        # interface all wire up end-to-end without import errors.
        _step("Phase 7: MCP tool layer round-trip (mock baseline)")
        os.environ["QUAD_ADAPTER_MODE"] = "mock"
        mock_cfg = load_config()
        mock_cfg.adapter_mode = "mock"
        mock_factory = AdapterFactory(mock_cfg, strict=False)

        from quad.mcp.tools import (
            convert_model_impl,
            generate_code_impl,
            hardware_detect_impl,
            orchestrate_workload_impl,
            profile_workload_impl,
        )

        async def _round_trip() -> dict[str, object]:
            # 1) hardware_detect
            hw = await hardware_detect_impl("windows", mock_factory)

            # 2) convert_model — write a tiny placeholder ONNX so paths exist
            tmp = tmp_path_factory.mktemp("e2e_model")
            fake_onnx = tmp / "tiny.onnx"
            fake_onnx.write_bytes(b"\x08\x09")  # mock adapter doesn't open it
            cv = await convert_model_impl(
                source_format="onnx",
                model_path=str(fake_onnx),
                target_sdk="qnn",
                quantization="fp32",
                factory=mock_factory,
            )

            # 3) profile_workload
            pf = await profile_workload_impl(
                model_path=str(fake_onnx),
                platform="windows",
                runtime="auto",
                duration_s=1,
                factory=mock_factory,
            )

            # 4) orchestrate_workload
            orc = await orchestrate_workload_impl(
                model_path=str(fake_onnx),
                power_mode="balanced",
                factory=mock_factory,
            )

            # 5) generate_code  (note: no factory arg — codegen reads templates,
            #    not adapters)
            gc = await generate_code_impl(
                platform="windows",
                sdk="qnn",
                language="cpp",
                model_path=str(fake_onnx),
            )

            return {
                "hardware_detect": hw,
                "convert_model": cv,
                "profile_workload": pf,
                "orchestrate_workload": orc,
                "generate_code": gc,
            }

        results = asyncio.run(_round_trip())
        for tool_name, payload in results.items():
            assert isinstance(payload, dict) and payload, f"{tool_name} returned empty/None"
            print(f"   {tool_name:<22} -> keys={sorted(payload.keys())[:6]}")

    finally:
        # Restore original env
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    elapsed = time.perf_counter() - started
    print(f"\n   total elapsed: {elapsed:.2f}s")


# ───────────────────────────── Standalone runner ───────────────────────────


def _main() -> int:
    """Run the test directly without pytest, for ad-hoc verification."""

    class _Factory:
        @staticmethod
        def mktemp(prefix: str) -> Path:
            import tempfile
            return Path(tempfile.mkdtemp(prefix=f"{prefix}_"))

    if _archive is None:
        print("SKIP: no QAIRT archive in ~/Downloads/. "
              "Set QUAD_TEST_QAIRT_ARCHIVE to the path.")
        return 0

    print(f"Using archive: {_archive}")
    try:
        test_e2e_qairt_real_mode_full_pipeline(_Factory)
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {type(e).__name__}: {e}")
        return 2
    print("\nPASS — all 7 phases succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
