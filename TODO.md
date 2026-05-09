# QUAD — Pending Tasks

> Last updated: 2026-05-08 (Sprint 1) | Tests: ~2068 passing | Version: 0.4.0+sprint1
>
> **Authoritative refs:**
> - [`docs/PRODUCTION_READINESS_REVIEW_2026-05-08.md`](docs/PRODUCTION_READINESS_REVIEW_2026-05-08.md) — newest review, all current P0/P1/P2 items
> - [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md) — original 4-tier scorecard
> - [`docs/IMPLEMENTATION_PROGRESS.md`](docs/IMPLEMENTATION_PROGRESS.md) — what previously landed
>
> **Sprint 1 closures (this session):** P0-1 (Windows detect_hardware
> no longer crashes), P0-2 (arch-aware bin_dir), P0-3 (qairt flavor
> classification), P0-4 (real qnn-platform-validator parsing), P0-7
> (PyPI dry-run in CI), P0-8 (QUAD-Client stdio-local Python+import
> preflight), P0-9 (QUAD-Client SSE handshake validation), P1-1
> (`_parse_latency` doesn't lie on parse failure), P1-2 (`_SDK_DIR_RE`
> handles `v2.46.0.260424` and bare versions), P1-5 (README/docs use
> `quad.mcp.server`).
>
> **Quick start for real hardware:**
>   1. `quad sdk install <path-to-qairt-archive>`  (download from
>      <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>)
>   2. `export QUAD_ADAPTER_MODE=real`
>   3. `quad mode` to confirm READY, `quad doctor --real-mode` for a full pre-flight.
>   See `docs/REAL_HARDWARE.md` for the complete playbook.
>
> Live sample-app run on Dell Latitude 7455 (Snapdragon X Elite): see
> `docs/SAMPLE_APP_REPORT.md` and `examples/sample_app_real_hw.py`.
> Real CPU inference measured at 388 FPS / 2.56 ms mean / σ=0.95 ms on
> Oryon (MobileNetV2-1.0, 500 iters via ONNX Runtime CPU EP).
> NPU section pending QAIRT SDK install on the test laptop.

---

## Tier 1 — End-to-end blockers

After Sprint 1, the remaining Tier-1 items are all in the Sprint 2-4
scope below. Closed items are kept here struck-through for history.

- [x] ~~Templates not packaged~~ (closed pre-session: hatch `force-include`)
- [x] ~~`QAIRTAdapter.execute_inference` ignores input data~~ (closed: `qairt_adapter.py:563`, fully marshals input/output via `model_inputs`)
- [x] ~~`_create_dummy_input_list` always returns `np.random.randn(1,3,224,224)`~~ (closed: real introspection)
- [x] ~~No AIMET adapter~~ (mock + qairt-quantizer fallback present; real torch/onnx still TODO — see Sprint 4)
- [x] ~~No AI Hub adapter~~ (closed: `aihub_adapter.py` lazily imports `qai_hub`)
- [ ] **Generated C++ scaffolds need a CI compile-test** — templates have real SDK API shape; what's missing is a CI step that actually compiles a generated file (T1.8 follow-up, Sprint 2 P0-6)
- [ ] **Inference server `start()` has no HTTP binding** — Sprint 2 target (T1.2)
- [ ] **Compiler pipeline returns BackendNotImplementedError** (honest, not a placeholder) — Sprint 3 target (T1.1)
- [ ] **Runtime is numpy-backed mocks** — Sprint 2-3 partial; full runtime in Sprint 3+ (T1.3)
- [ ] **AIMET real torch/onnx backends** — Sprint 4 target (T1.5 partial)

---

## Tier 2 — Real-mode produces wrong/incomplete output

- [x] ~~Latency parser silent fallback~~ (Sprint 1 P1-1: returns 0 on no match, callers can detect)
- [x] ~~`detect_hardware` is hardcoded fallback~~ (Sprint 1 P0-1, P0-4: real validator parsing on all platforms)
- [x] ~~No CI on Windows~~ (closed pre-session: matrix includes `windows-latest`)
- [ ] **`orchestrate_workload` output ignored by codegen** — allocation map is dead-end metadata (T2.3)
- [ ] **`orchestrate_workload` crashes on linting profiles** — empty `report.layers` not handled (T2.4)
- [ ] **Phase 2/3 platforms have transport but no adapters** — `LinuxPlatform` / `AndroidPlatform` exist but aren't wired into the factory (T2.5)
- [x] ~~Package name mismatch~~ (Sprint 5: settled on `quad-agent`; CLAUDE.md success-metric updated; release.yml and real-hw.yml workflows added)

---

## Tier 3 — Polish, breadth, error handling

- [ ] Typed exceptions defined but unused — modules throw built-in `RuntimeError`/`ValueError` instead (T3.1)
- [ ] No auto-generated API docs (mkdocs / sphinx) (T3.2)
- [ ] Plugins folder incomplete — VS Code partial; Arduino + Android Studio missing (T3.3)
- [ ] Model registry has no real models — paths point at non-existent files; no `download_model()` (T3.4)
- [ ] `deploy.sh` is SNPE-only and fragile — hardcoded skel paths, no QNN backend (T3.5)
- [ ] `quad detect` is a thin mock returning hardcoded device list (T3.6)
- [ ] `get_supported_ops` is a hardcoded list — doesn't reflect installed SDK version (T3.7)

---

## Priority 1 — Real SDK Integration (Blockers)

These are blocked by missing SDK output format documentation.
Once the CLI `--help` output / stdout format is known, the adapter can be wired.

- [ ] **`QAIRTAdapter.convert_model()` output parser**
  - Need: `qairt-converter` stdout format (what does it print on success/failure?)
  - File: `src/quad/adapters/qairt_adapter.py:convert_model()`
  - Blocked on: qairt-converter stdout/stderr format documentation

- [ ] **`QAIRTAdapter.profile()` output parser**
  - Need: `snpe-diagview` text output format for basic/detailed levels
  - File: `src/quad/adapters/qairt_adapter.py:_profile_standard()`
  - Blocked on: snpe-diagview output format / SNPEDiag .bin schema

- [ ] **`QAIRTAdapter.detect_hardware()` output parser**
  - Need: `qnn-platform-validator` stdout format
  - File: `src/quad/adapters/qairt_adapter.py:detect_hardware()`
  - Blocked on: qnn-platform-validator output format

- [ ] **QNN C++ SDK adapter** (Phase 1 — AI PC)
  - Need: QNN SDK headers + `libQnnHtp.so` dynamic loading
  - Reference: `src/quad/qnn/inference_pipeline.py` (pipeline is documented)
  - Blocked on: physical Snapdragon X Elite device

---

## Priority 2 — Phase 1: AI PC / Windows

- [ ] **AIMET integration** — INT8/INT4 quantization recommendations in `convert_model`
  - Create `src/quad/adapters/aimet_adapter.py`
  - Wire into `QAIRTAdapter.convert_model()` when quantization="int8" or "int4"
  - SDK reference: `src/quad/compiler/model_conversion.py` (QAIRT_QUANTIZER_NOTES)

- [ ] **Qualcomm AI Hub integration** — cloud benchmarking via `qai_hub` Python SDK
  - Create `src/quad/adapters/aihub_adapter.py`
  - Implement `profile_on_device()` API
  - Needs: `QAI_HUB_API_KEY` env var (already in `.env.example`)

- [ ] **QPM3 / Snapdragon Profiler real output parser**
  - Wire `profile_workload` tool to parse actual QPM3 CSV output
  - File: `src/quad/tools/profile_workload.py`

- [ ] **End-to-end validation on Snapdragon X Elite**
  - Run `quad quickstart` on real hardware, verify TTFI < 10 minutes

---

## Priority 3 — Phase 2: Arduino UNO Q / Linux (QCS2210)

- [ ] SNPE DSP adapter — DLC conversion + DSP runtime for QCS2210
- [ ] `hardware_detect` — QCS2210 discovery via SSH/serial
- [ ] `convert_model` — ONNX → SNPE DLC with INT8 for ARM64
- [ ] `profile_workload` — SNPE Profiler + Hexagon Profiler automation
- [ ] `orchestrate_workload` — DSP vs CPU under 3W power budget
- [ ] `generate_code` — Arduino sketch + SNPE runtime integration
- [ ] Cross-compilation pipeline (host → ARM64 target)
- [ ] Deploy and validate on physical QCS2210 hardware

---

## Priority 4 — Phase 3: Mobile / Android (SM8750)

- [ ] ADB-based Snapdragon 8 Elite hardware detection
- [ ] SNPE DLC / QNN binary for Android target
- [ ] Snapdragon Profiler + Perfetto automation
- [ ] Thermal-aware NPU/GPU/CPU workload switching
- [ ] Android AAR generation (Kotlin/Java + SNPE runtime)
- [ ] MediaPipe integration option
- [ ] Power-profile scheduling (performance / balanced / efficiency)

---

## Priority 5 — Phase 4: Cross-Platform & Optimization

- [ ] **AIMET INT4 pipeline** — 4-bit quantization via AIMET
- [ ] **AI Hub CI/CD** — automated benchmarking on every commit
- [ ] **Multi-model pipeline chaining** — detection + classification + segmentation
- [ ] **Cross-platform performance dashboard** — Plotly/Dash results viewer
- [ ] **Perfetto/Snapdragon Profiler trace configs** for CI

---

## Priority 6 — Infrastructure

- [ ] **CI/CD pipeline** — GitHub Actions
  - Run `pytest -q` on every push
  - Lint with ruff
  - Optional: mock-mode E2E test
  - File to create: `.github/workflows/ci.yml`

- [ ] **Development environment validation**
  - Complete the checklist from `docs/PREREQUISITES.md`
  - Run `quad doctor` and resolve all warnings on a clean machine

- [ ] **PyPI publication preparation**
  - Package as `qualcomm-ai-toolkit`
  - Validate `pip install qualcomm-ai-toolkit && quad quickstart` in < 5 minutes
  - Update `pyproject.toml` package name

---

## Priority 7 — SDK Docs Not Yet Integrated

The following SNPE/QNN documentation sections have not yet been shared/integrated:

- [ ] QNN Backend documentation (HTP/CPU/GPU backend APIs — C++ level)
- [ ] QNN HTP Optrace Profiling (referenced in QHAS docs but not shared in full)
- [ ] Running Inception v3 end-to-end tutorial (C/C++ tutorial section)
- [ ] Snapdragon Profiler output format (for real profiling integration)
- [ ] QPM3 CLI output format (for real power measurement integration)
- [ ] Multiple concurrent inference (advanced SNPE patterns)
- [ ] Network resizing API details (Snpe_SNPEBuilder_SetInputDimensions)
- [ ] AI Hub Python SDK reference

---

## Priority 8 — Community / Phase H (Not Started)

- [ ] QUAD Academy — online courses (Beginner, Intermediate, Advanced)
- [ ] Certification program — QUAD Developer certification
- [ ] Community forums / Discord
- [ ] Hackathon integration guide
- [ ] Plugin marketplace for custom SDK adapters

---

## Completed This Session ✅

- [x] Input Image Formatting (NHWC/BGR/batch) → `compiler/model_conversion.py`
- [x] HTP Linting Profile → `profiler/linting.py`
- [x] QHAS Profiling → `profiler/qhas.py`
- [x] All 8 gap fixes (models, tools, adapter dispatch, doctor checks, templates)
- [x] MobilenetSSD Benchmarking → `benchmarks/`
- [x] Inference Accuracy (mAP, Top-1/5) → `accuracy/`
- [x] SDK Tools platform matrix + full CLI builders → `sdk_tools/`
- [x] Architecture Checker → `sdk_tools/architecture_checker.py`
- [x] Accuracy Debugger → `sdk_tools/accuracy_debugger.py`
- [x] QNN SDK API reference + inference pipelines → `qnn/`
- [x] README.md updated to v0.3.0
- [x] CLAUDE.md Active Context updated
