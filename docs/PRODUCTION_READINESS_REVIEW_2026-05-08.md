# Production-Readiness Review — 2026-05-08

> **Scope:** independent end-to-end review of `QUAD` (server) and
> `QUAD-Client` (provisioner) for production readiness. Goes beyond the
> existing `docs/GAP_ANALYSIS.md` and `docs/IMPLEMENTATION_PROGRESS.md`
> by cross-checking code-vs-doc claims, exercising the install path on
> a real Snapdragon X-series Windows ARM64 host, and running a new
> end-to-end test against the QAIRT 2.46 SDK.

> **Status:** v0.4.0 is **not yet production-ready**, but the architecture
> is sound and the gap surface is well-understood. Most P0 items are
> 1-2 days each; the remaining heavy lifts (compiler backend, AIMET
> real backend, runtime SDK calls) are already correctly classified as
> deferred behind honest `NotImplementedError` raises.

---

## What this review changes vs. the existing GAP_ANALYSIS.md

| New finding | Where surfaced |
|---|---|
| `_populate_sdkinfo` picks `bin_dir` by alphabetical iteration order, not host arch — on a Windows ARM64 box it lands on `x86_64-windows-msvc/` even though native binaries exist in `aarch64-windows-msvc/`. | `src/quad/sdk_manager.py:150–172`, observed in `sdks/qairt-2.46.0.260424` resolution |
| Flavor classification is order-dependent and incorrect — a QAIRT install is classified as `flavor="snpe"` because the first bin subdir alphabetically (`aarch64-android`) ships `snpe-net-run` but no `qairt-converter`. | `src/quad/sdk_manager.py:117–124` (`_looks_like_sdk_root`) — observed live |
| `_SDK_DIR_RE` requires a `qairt`/`snpe` prefix; archive dirs named `v2.46.0.260424` (the actual portal naming) yield `version="unknown"`. | `src/quad/sdk_manager.py:87`, `_version_from_dir_name` |
| `QAIRTAdapter.detect_hardware("windows")` is unimplemented — falls through to a hardcoded `"linux"` branch that calls `os.uname()`, which raises `AttributeError` on Windows. | `src/quad/adapters/qairt_adapter.py:98–128` |
| `_parse_latency` silently falls back to `5.0` ms when no regex matches, so a parser break is invisible. The same shape is true for power/memory which are hardcoded constants (2000 mW, 50 MB peak). | `src/quad/adapters/qairt_adapter.py:744–758`, `_profile_standard` lines ~370–375 |
| `bin_dir` mismatch means `_find_tool` will pick x86_64 binaries on ARM64 — they run via Prism (slow) and `qnn-platform-validator.exe` is **not** present in `x86_64-windows-msvc/` for QAIRT 2.46, so the resolver fails to find it. | Direct ZIP listing of QAIRT 2.46 |
| Microsoft Store Python is x86_64 — under Prism on ARM64 it reports `PROCESSOR_ARCHITECTURE=AMD64` while `platform.machine()` reports `ARM64`. Any host-detection logic that uses only `PROCESSOR_ARCHITECTURE` will mis-classify ARM64 hosts. | Observed live in this venv |
| `quad-client` `stdio-local` probe only checks `shutil.which("python")` — does not verify Python version, does not verify `quad.mcp.server` is importable. The `stdio-ssh` probe correctly does both. | `QUAD-Client/src/quad_mcp_client/connection.py:82–162` |
| `sse-http` probe does HTTP reachability only — no MCP/SSE handshake. A 200 from a static site validates as success. | `QUAD-Client/src/quad_mcp_client/connection.py:313` |

---

## What we actually verified works (live, on this host)

The new test `tests/e2e/test_real_sdk_e2e.py` ran end-to-end on this
Snapdragon X-series Windows ARM64 box in 5.8s and exercised:

```
Phase 1  install QAIRT 2.46 zip via sdk_manager.install_archive()
Phase 2  discover SDK via sdk_manager.resolve_sdk_root()
Phase 3  pick the right per-arch bin (aarch64-windows-msvc)
Phase 4  real subprocess to qnn-platform-validator.exe --help
         -> exit 0, stdout starts: "DESCRIPTION:\nQNN-platform-validator
            is a tool to check the capabilities of a device."
Phase 5  apply_to_environment populates QAIRT_SDK_ROOT
Phase 6  AdapterFactory in real mode returns a real QAIRTAdapter
         (no silent fallback)
Phase 7  Round-trip all 5 MCP tools through mock-mode factory
```

Confirms:
* SDK install + discovery is real
* The `bin/<arch>/` layout assumption holds for QAIRT 2.46
* The factory wires up a real `QAIRTAdapter` when env vars are set
* All 5 MCP tool implementations dispatch cleanly through `quad.mcp.tools`

---

## Tier P0 — Production blockers

These prevent claiming "v1.0 production-ready" even with the SDK
installed. Each is small (≤ 2 days).

| # | Issue | File(s) | Fix |
|---|---|---|---|
| **P0-1** | `QAIRTAdapter.detect_hardware("windows")` crashes (`os.uname` AttributeError on Windows). | `src/quad/adapters/qairt_adapter.py:98–128` | Add a `windows` branch that uses `platform.machine()` + `psutil` (already a real-extra dep). Reuse the live host probe from `quad/runtime/host_probe.py`. |
| **P0-2** | `bin_dir` selection ignores host arch — picks alphabetical-last, mis-routes to x86_64 on ARM64 hosts where required tools are not present. | `src/quad/sdk_manager.py:150–172` | Rank candidate bin subdirs by `(matches_host_arch, has_qairt_converter, has_snpe_net_run)` rather than insertion order. Native arch first, then x86_64-via-emulation. |
| **P0-3** | Flavor classification depends on iteration order — QAIRT installs end up tagged `snpe`. | `src/quad/sdk_manager.py:117–124` | Search every bin subdir before deciding; prefer `qairt` when both markers are present. |
| **P0-4** | Real-mode `qnn-platform-validator` output is read but discarded — `detect_hardware` always returns the fallback hardcoded chipset string. | `src/quad/adapters/qairt_adapter.py:103–107` | Parse the validator's `--coreVersion --backend all` output (this is the documented invocation; QAIRT 2.46 supports it). Map results to `DeviceProfile`. |
| **P0-5** | Inference HTTP server is documented as "real FastAPI" but `ModelServer.infer()` returns `np.random` and there is no `start()` HTTP binding. | `src/quad/serve/server.py`, `src/quad/serve/http.py` | Either delete the README claim and mark `quad serve` experimental, or wire FastAPI + a real model loader. (README says "real FastAPI server" — must match reality.) |
| **P0-6** | Generated QNN C++ template `inference_bin.cpp.j2` has the right SDK API shape but several user-supplied placeholders need a verified compile path. The validator detects TODO/FIXME but no test compiles a real generated file. | `templates/qnn/inference_bin.cpp.j2`, `src/quad/codegen/validators.py` | Add a CI job that runs `generate_code` for `mobilenetv2.onnx` then compiles the output with the QAIRT-bundled toolchain (`bin/x86_64-windows-msvc/`). Ship the result as an artifact. |
| **P0-7** | PyPI dry-run not in CI — packaging breakage would be discovered at release time. The package name in `pyproject.toml` (`quad-agent`) does not match the install instructions in the README. | `pyproject.toml`, `.github/workflows/ci.yml` | Add a CI step that runs `python -m build` then `pip install dist/*.whl` in a clean venv and exercises `quad mode`. Decide on `quad-agent` vs `qualcomm-ai-toolkit` — pick one and update both. |
| **P0-8** | QUAD-Client `stdio-local` probe never verifies `python -c "import quad.mcp.server"` works. The package can be installed against an ancient Python or against a venv missing QUAD entirely and the probe still passes. | `QUAD-Client/src/quad_mcp_client/connection.py:82–162` | Run `python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1); import quad.mcp.server"` in the probe (this is what the SSH probe already does). |
| **P0-9** | QUAD-Client `sse-http` probe accepts any 200 response — does not validate the MCP/SSE handshake. | `QUAD-Client/src/quad_mcp_client/connection.py:313` | Issue an OPTIONS or read the first SSE event with a short timeout; require `text/event-stream` content-type. Fall back gracefully on 4xx with hint. |

## Tier P1 — Correctness gaps

| # | Issue | Fix |
|---|---|---|
| **P1-1** | `_parse_latency` silently returns 5.0 ms on parse failure. Per-layer/power/memory profile fields are hardcoded constants in real mode. | Distinguish "no data" (return `None`) from "5 ms"; propagate the raw stdout in the response so callers can debug. Replace the 2000 mW / 50 MB constants with real `snpe-net-run --perf_profile` parsing or `None`. |
| **P1-2** | `_SDK_DIR_RE` ignores QAIRT's actual portal naming (`v2.46.0.260424.zip`). Version becomes `"unknown"` and downstream UI says `qairt unknown`. | Extend regex to `^(?:qairt|snpe|v)?[-_ ]?(\d+\.\d+(?:\.\d+){0,2})$`. |
| **P1-3** | `QAIRTAdapter.execute_inference` exists and looks correct, but no test exercises the `snpe-net-run`-based round-trip — it is silently not run anywhere. | Add a test that runs `execute_inference` against a tiny `.dlc` sample (the SDK ships some under `examples/Models/`) and asserts shape of returned `outputs`. |
| **P1-4** | Generated `windows/python` and `linux/arduino` templates are not exercised by the validator's optional `gcc -fsyntax-only` step (Python templates aren't C++; Arduino templates need different syntax checking). | Per-language validator dispatch: `pyflakes` for Python templates, skipped for Arduino with a TODO. |
| **P1-5** | Backward-compat shim at `src/quad/server/__init__.py` is correct but the README's "Quick Start" still mentions `python -m quad.server` while `.claude/settings.json` uses `quad.mcp.server`. | Pick one, update README and the troubleshooting table to match. |
| **P1-6** | QUAD-Client `bootstrap.ps1` does not fall back to WSL if Git Bash is missing and `winget install` fails. | Try `wsl bash` then `bash` from `$env:Path` before failing. |
| **P1-7** | QUAD-Client doesn't validate `--ssh-key` file existence in CLI; only fails at probe time. | Validate in `cli.py:install` before calling the probe. |
| **P1-8** | TODO.md asserts several gaps that are already closed (templates packaging, AIMET adapter, Windows CI). Drift between docs and reality. | Sweep TODO.md against `docs/IMPLEMENTATION_PROGRESS.md` and remove stale items. |

## Tier P2 — Heavy lifts (already correctly classified as deferred)

These are honestly stubbed today (raise `NotImplementedError`) and will
take a sprint each. They are **not blockers for v1.0** as long as the
README and `quad doctor --real-mode` report their absence clearly.

* T1.1 — Real compiler backend (IR → SDK-compiled binary). 2-3 weeks.
  Path A (qairt-converter shell-out + cache) is the fastest credible win.
* T1.3 — Runtime ctypes/cffi to QNN C API. 1-2 weeks. Only required for
  the in-process inference server (P0-5); shelling to `snpe-net-run`
  is a viable interim.
* T1.5 — AIMET PTQ for `aimet_torch` / `aimet_onnx`. 2 weeks. Mock path
  works today; real path is gated by needing PyTorch + AIMET in dev env.
* T2.5 — Phase 2/3 platform adapters wired through the factory.
* T3.x — VS Code extension / Arduino / Android Studio plugins; model
  registry; deploy.sh QNN backend; recommended-ops dynamic from SDK.

## Tier P3 — Nice-to-have

* Auto-generated API docs (mkdocs).
* Cross-platform performance dashboard.
* Hosted MCP service (sse-http transport already supported by client).
* AIMET INT4 pipeline.
* Multi-model pipeline chaining.

---

## Phased plan

### Sprint 1 — "honest v0.5" (1 week)

Close P0-1 through P0-9 and P1-1, P1-2, P1-5, P1-8. None of these is
hard; they are mostly correctness. After this sprint:

* `quad doctor --real-mode` passes on Windows ARM64 without crashing.
* SDK is detected as `qairt 2.46.0.260424` (not `snpe unknown`).
* `qnn-platform-validator` output is parsed, populating DeviceProfile.
* CI builds the wheel, installs it cleanly, runs `quad mode`.
* QUAD-Client probes do not pass for misconfigured hosts.
* README and TODO.md no longer drift from reality.

**Exit criterion:** the e2e test in `tests/e2e/test_real_sdk_e2e.py`
passes on Linux x86_64 and Windows ARM64 in CI, with no skipped
phases. Add a Linux runner.

### Sprint 2 — "first real inference" (2 weeks)

* P0-5 (HTTP server bindings) — wire FastAPI + a real `snpe-net-run`-
  backed `infer()` for QAIRTAdapter. This unblocks `quad serve`.
* P1-3 (test `execute_inference` against a real `.dlc`).
* P0-6 (CI compile of generated C++ for `mobilenetv2.onnx`).

**Exit criterion:** `quad serve mobilenetv2.dlc` accepts a POST and
returns a real classification vector on Snapdragon X.

### Sprint 3 — "real compiler backend (Path A)" (2 weeks)

* T1.1 — Wire `quad.compiler.pipeline` to shell out to `qairt-converter`
  + `qairt-quantizer` and emit a real `.bin` / `.dlc`. Cache by IR hash.
* Backfill tests for the cache layer.

**Exit criterion:** `quad compile model.onnx --target qnn` produces a
real binary that runs on `snpe-net-run` without manual fixup.

### Sprint 4 — "AIMET real" (2 weeks)

* T1.5 — `_quantize_aimet_torch` / `_quantize_aimet_onnx` using
  the real `aimet-torch` / `aimet-onnx` packages. Calibration data
  pipe-through.

**Exit criterion:** `convert_model(... quantization='int8', use_aimet=True)`
produces a model whose accuracy is within 1% of fp32 on a small
classification benchmark.

### Sprint 5 — "publish + plug-ins" (1-2 weeks)

* Decide on PyPI name (`quad-agent` vs `qualcomm-ai-toolkit`).
* PyPI publication workflow.
* VS Code extension finishing.
* Real-hardware CI on Snapdragon X (self-hosted runner).

---

## What this means for "production"

| Claim today | Reality | After Sprint 1 | After Sprint 2-3 |
|---|---|---|---|
| "5 MCP tools work" (mock) | True | True | True |
| "5 MCP tools work" (real) | Partial — convert + execute_inference real; profile partly real; detect mocked; orchestrate metadata-only | All five honest, with parsed real outputs | Real inference end-to-end |
| "TTFI < 10 min" | Mock-mode yes; real-mode crashes on Windows | Real-mode meets 10 min on Linux x86_64 | Real-mode meets it on all 3 platforms |
| "Real FastAPI inference server" | False (returns np.random) | Marked experimental | True |
| "PyPI install in 5 min" | Untested | Tested in CI | Published |
| "Snapdragon X Elite validated" | CPU yes, NPU no | NPU detection yes | NPU inference yes |

---

## The new e2e test

`tests/e2e/test_real_sdk_e2e.py` is the first test in the suite that:

* Uses the actual QAIRT zip from the user's Downloads folder
* Performs a real subprocess call to a Qualcomm `.exe`
* Validates the AdapterFactory's real-mode path end-to-end

It is the single reproducible production-realism check for this repo
on Windows ARM64. Run with:

```powershell
.venv\Scripts\python.exe -m pytest tests/e2e/test_real_sdk_e2e.py -v -s
```

It auto-skips if no QAIRT archive is in `~/Downloads/`. Override path
with `QUAD_TEST_QAIRT_ARCHIVE=<path>`.

This test should join the CI matrix as soon as a self-hosted Snapdragon
runner is available. Until then, it serves as the local smoke check
before each release tag.
