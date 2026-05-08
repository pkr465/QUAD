# QUAD Gap-Closure Implementation Plan

> Companion to [`docs/GAP_ANALYSIS.md`](GAP_ANALYSIS.md). This document
> is the execution plan that closes the Tier-1 / Tier-2 gaps in
> sequenced phases. Each phase is independently commit-able and
> shippable; the gap analysis estimated 6-10 person-weeks of total
> work — this plan groups it into phases so progress is visible after
> each phase completes.

## Phases at a glance

| Phase | Scope | Closes | Effort |
|---|---|---|---|
| **A** | Quick wins | T1.7, T2.4, T2.6, T2.8, T3.6, T1.4 (partial) | ~half day |
| **B** | Codegen quality | T1.8, validator strengthening | ~1 day |
| **C** | AIMET + AI Hub adapters | T1.5, T1.6 | ~2 days |
| **D** | Real Inference Server + Runtime backends | T1.2, T1.3 (partial) | ~2 days |
| **E** | ONNX frontend in compiler | T1.1 (partial — frontend only, SDK call still stubbed) | ~1 day |
| **F** | Claude Code skills + UX | net new — slash commands, summaries, tips | ~1 day |
| **G** | Docs + mkdocs + final polish | T3.2, README updates, GAP_ANALYSIS update | ~half day |

After each phase: full test suite runs, commit with detail, push to main.

---

## Phase A — Quick wins (foundation)

These are small but unblock everything else.

1. **T1.7 — Templates package_data**
   - Move `templates/` under `src/quad/templates/` (or add to package_data)
   - Update `pyproject.toml` with `[tool.setuptools.package-data]`
   - Update `CodegenEngine` to find templates relative to package
   - Acceptance: `pip install --no-deps -e .` then `quad quickstart` works in mock mode without the source repo

2. **T2.6 — Windows CI + 8 path-assertion fixes**
   - Add `windows-latest` to `.github/workflows/ci.yml` matrix
   - Fix the 8 known failures (one-line `Path.as_posix()` per assertion)
   - Skip the symlink test on Windows
   - Acceptance: full test suite green on both Linux and Windows

3. **T2.4 — `orchestrate_workload` empty-layers handling**
   - When `report.layers` is empty (linting profile), either re-run
     in `detailed` mode or raise `OrchestrationError` with a clear msg
   - Add unit test for the empty case
   - Acceptance: `orchestrate_workload` after a `linting` profile no
     longer crashes

4. **T2.8 — Real `_create_dummy_input_list` based on DLC introspection**
   - Use `snpe-dlc-info` to read input shape/dtype from the converted DLC
   - Fall back to a configurable shape if SDK isn't installed
   - Generate inputs of the right shape (still dummy data — calibration
     dataset is a separate gap, T1.5)
   - Acceptance: `profile_workload` works on a model with non-default input shape

5. **T3.6 — Make `quad detect` actually probe local hardware**
   - Use the existing `examples/sample_app_real_hw.py` PowerShell probe pattern
   - On Linux, parse `/proc/cpuinfo` and `lspci` / `ls /dev/`
   - On Android, use `adb getprop`
   - Acceptance: `quad detect` on the test laptop reports actual Snapdragon X Elite specs

6. **T1.4 — Real `execute_inference` I/O marshalling (partial)**
   - Marshal `input_data` (dict of numpy arrays) into `.raw` files
   - Read output `.raw` files back into numpy arrays
   - Don't truncate stdout in the response
   - Acceptance: `QAIRTAdapter.execute_inference(model, {"input": np.array(...)})` returns matching-shape numpy outputs (in mock mode for tests; real-mode pending hardware)

**Phase A test target:** all existing tests still pass + ~15 new tests for the fixes.

---

## Phase B — Codegen quality

Generated code should compile and run, not be a TODO scaffold.

1. **Fill in QNN C++ templates** (`templates/qnn/inference_so.cpp.j2`,
   `inference_bin.cpp.j2`, `inference_tflite_delegate.cpp.j2`,
   `templates/windows/cpp/inference.cpp.j2`)
   - Real `dlopen` + `QnnInterface_getProviders` calls
   - Real graph compose / finalize / execute
   - Real input/output tensor allocation and binding
   - Real cleanup sequence
   - Reference: `src/quad/qnn/inference_pipeline.py` (already has the
     6-step pipeline documented)

2. **Strengthen C++ validator** (`src/quad/codegen/validators.py`)
   - Beyond bracket balance: require declared identifiers to be used
   - Optionally invoke `gcc -fsyntax-only` if a C++ toolchain is on PATH
   - Add tests with deliberately-broken templates that the old validator
     would have passed but the new one rejects

3. **Heterogeneous codegen** (T2.3)
   - Accept `allocation_map` from `orchestrate_workload`
   - Emit code that loads the model on multiple backends and dispatches
     ops per the allocation
   - For now: a sub-template that gets included when allocation_map is
     non-trivial; full heterogeneous dispatch is its own phase

**Phase B test target:** new compilable-output tests + ~20 new template tests.

---

## Phase C — AIMET + AI Hub adapters

Two missing adapters that the documentation claims work.

1. **`src/quad/adapters/aimet_adapter.py`**
   - Calibration dataset loader (CSV / HDF5 / numpy / image directory)
   - PTQ workflow (`aimet_torch.quantsim.QuantizationSimModel`)
   - INT8 + INT4 paths
   - Wire into `QAIRTAdapter.convert_model` when quantization is requested
   - Tests with mocked `aimet_torch` import (so CI doesn't need AIMET installed)

2. **`src/quad/adapters/aihub_adapter.py`**
   - `qai_hub` SDK integration (also mocked for CI)
   - `profile_on_device(model, device)` → returns same shape as local profile
   - `compile_for_device(model, device)` → cloud-compiled artifact
   - Wire into the MCP `profile_workload` tool with a `cloud=True` option
   - Test with mocked qai_hub client

3. **Configuration**
   - `QAI_HUB_API_KEY` actually consumed
   - `quad doctor` checks for both AIMET and qai_hub installation

**Phase C test target:** ~30 new tests for AIMET, ~20 for AI Hub, all using mocks so they run in CI without those SDKs installed.

---

## Phase D — Real Inference Server + Runtime backends

`quad serve` should actually serve.

1. **`ModelServer.start()` — real FastAPI binding**
   - Add `fastapi` + `uvicorn` to `[real]` extras in pyproject
   - Routes: `POST /infer`, `GET /health`, `GET /models`, `POST /models/{name}/load`, `DELETE /models/{name}`
   - Pydantic request/response models
   - Tests using `httpx.AsyncClient` against the FastAPI app

2. **`ModelServer.infer()` — call real runtime**
   - Replace `np.random.randn(batch, 1000)` with a call to
     `Model.infer()` which delegates to `QAIRTAdapter.execute_inference`
   - For mock mode, retain a deterministic seeded synthetic output
     (current behaviour but with seed)

3. **Real Device.list_devices() (T1.3 partial)**
   - Use the local hardware probe from Phase A.5
   - For now, only list devices on this machine; remote devices are
     a separate phase

4. **Real Device.is_available() / .get_info()**
   - Cross-check the chipset name against the local probe

**Phase D test target:** ~25 new tests for the server (including end-to-end inference round-trip in mock mode), ~15 for the runtime improvements.

---

## Phase E — ONNX frontend in compiler

The compiler returns `b"QUAD_COMPILED_BINARY"` placeholder bytes today.
We can do the **frontend** part this phase (real ONNX parse → real
IRGraph) without doing the **backend** part (which requires deep SDK
integration and is a separate multi-week effort).

1. **Real `compile_onnx()`**
   - Use `onnx.load(path)` to parse the model
   - Walk the graph and convert each ONNX op to an `IRNode`
   - Populate `IRGraph` with real nodes (not the mock 3-node graph)
   - Track input/output shapes through the graph

2. **`compile_pytorch()` via torchscript export**
   - Save to ONNX first via `torch.onnx.export`, then route through ONNX path
   - Document the limitation (PyTorch dynamic models need tracing)

3. **`compile_tensorflow()` via tf2onnx**
   - Use `tf2onnx.convert` then route through ONNX path

4. **Backend stub honesty**
   - The placeholder bytes get a clear comment that they're frontend-only
   - Add a `compile_for_target()` method that raises `NotImplementedError("backend SDK call pending")` rather than silently returning placeholder bytes

5. **Op coverage report**
   - When a frontend can't lower an ONNX op, log it
   - `compile_model()` returns a `coverage` field with the % of ops
     successfully lowered

**Phase E test target:** ~30 new tests using `onnx`'s built-in test models (already shipped with onnxruntime).

---

## Phase F — Claude Code skills + UX

This is the user-visible layer that makes QUAD feel professional in
Claude Code. The user emphasized this — UI aspects, summaries,
suggestions, tips, professional skills look.

1. **`.claude/skills/` directory** with one file per skill
   - `quad-quickstart.md` — interactive walkthrough that uses the MCP tools step-by-step
   - `quad-detect.md` — rich hardware detection summary with NPU/GPU/CPU breakdown
   - `quad-convert.md` — model conversion with conversion notes, image format guidance, calibration tips
   - `quad-profile.md` — profiling with rich latency tables, percentile charts (mermaid), bottleneck highlights
   - `quad-orchestrate.md` — power-mode comparison with recommendations
   - `quad-codegen.md` — code generation with platform/language picker
   - `quad-doctor.md` — diagnostic walkthrough with fix-it suggestions
   - `quad-deploy.md` — guided deployment to a target device
   - `quad-recommend.md` — given a model + target, recommend the best path

2. **Rich response formatters** (`src/quad/ui/`)
   - `format_device.py` — `DeviceProfile` → markdown table with TOPS/TFLOPS/RAM
   - `format_profile.py` — `ProfilingReport` → latency table + utilisation bars + bottleneck callouts
   - `format_conversion.py` — conversion result with size comparison + warnings
   - `format_allocation.py` — allocation map with per-mode comparison
   - All return markdown strings; consumed by MCP tools and the CLI

3. **Suggestions engine** (`src/quad/suggestions.py`)
   - Given a model + target, suggest:
     - Best quantization (INT8 default; INT4 if size-constrained)
     - Best runtime (NPU for >100 ops; GPU for <100; CPU for trivial)
     - Best power mode (performance for camera/realtime; efficiency for batch)
   - Given a profile, suggest optimisations:
     - "Op #4 is a bottleneck (21% overlap) — try replacing Sub with Conv (see HTP linting docs)"
     - "Memory peak is 2× your target's available — reduce batch size"
   - Tests with synthetic profiles

4. **Tips system** (`src/quad/tips.py`)
   - 30+ contextual tips loaded from a YAML file
   - Each tip has: `context`, `level` (info/warning/critical), `text`, `link`
   - Surface 0-3 relevant tips with each MCP tool response
   - Examples:
     - "Tip: Use `quad benchmark` to compare this model against the standard set"
     - "Tip: Set `QUAD_STRICT_REAL=1` to fail-fast in CI when the SDK is missing"

5. **Slash commands** (registered via skill files)
   - `/quad-quickstart` — invokes `quad quickstart`
   - `/quad-detect` — calls `hardware_detect` and renders the rich summary
   - `/quad-doctor` — calls `quad doctor`, reports issues with fixes
   - `/quad-convert <model> <target>` — convert + show conversion report
   - `/quad-profile <model>` — profile + show rich timing breakdown
   - `/quad-bench` — run benchmark suite, show comparison

6. **MCP tool docstrings upgraded**
   - Every MCP tool has examples in its docstring
   - The tool returns: `{"data": ..., "ui": <markdown summary>, "tips": [...]}`
   - Claude Code clients (without QUAD-specific UI) still get useful
     output; clients that render markdown get the rich version

**Phase F test target:** ~40 new tests for formatters, suggestions, tips, skills loading.

---

## Phase G — Documentation + final polish

1. **mkdocs site** (`mkdocs.yml` + `docs/` reorganisation)
   - Auto-API docs via `mkdocstrings`
   - All existing docs (REAL_HARDWARE, SAMPLE_APP_REPORT, GAP_ANALYSIS,
     IMPLEMENTATION_PLAN, USAGE, PREREQUISITES) become nav entries
   - Build on push to `main` and publish to GitHub Pages

2. **README updates**
   - Reflect new capabilities (skills, suggestions, tips, real server)
   - New section: "What's new in this update"
   - Update phase progress table

3. **GAP_ANALYSIS update**
   - Mark each closed gap with ✅ and the commit SHA that closed it
   - Update the scorecard

4. **CLAUDE.md update**
   - New active context for this overnight session

5. **Coverage report on every CI run**
   - `pytest --cov` already runs; add `--cov-fail-under=85` to enforce

---

## Out of scope for this session

These are flagged in `GAP_ANALYSIS.md` but explicitly excluded here
because they're too large for one session:

- T1.1 (full real compiler **backend** — SDK call to produce real binaries)
- T1.3 (full real runtime — ctypes/cffi bindings to QNN's C API)
- T2.5 (Phase 2/3 platform adapters wired into the factory)
- T3.3 (Arduino + Android Studio plugins)
- T3.4 (model registry with real downloads)

Phase E does the **frontend** of the compiler honestly, which closes the
"placeholder bytes" lie even though the backend is still pending.
Phases C and D close the inference-server lie even though the runtime
is still partly numpy.

---

## Commit cadence

After each phase: full `pytest -q`, commit with detailed message,
fast-forward main, push. If any phase produces > ~50 changed files,
break it into multiple commits within the phase.

After Phase G: final commit summarising the session, with a
"## What's new" section in `CLAUDE.md`'s Active Context.
