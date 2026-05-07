# QUAD — Pending Tasks

> Last updated: 2026-05-04 | Tests: 1715 passing | Version: 0.3.0

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
