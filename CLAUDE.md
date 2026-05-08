# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the **PRD and implementation artifacts** for a Claude Code-powered AI agent that abstracts Qualcomm SDK toolchains for AI developers. The agent uses MCP (Model Context Protocol) to provide natural-language access to hardware detection, model conversion, profiling, workload orchestration, and code generation across three platforms:

1. **AI PC (Windows)** — Snapdragon X Elite, QNN/QAIRT SDK, QPM3 profiler (Priority 1)
2. **Arduino UNO Q (Linux)** — QCS2210, SNPE SDK, Hexagon DSP (Priority 2)
3. **Mobile (Android)** — Snapdragon 8 Elite, SNPE/QNN SDK, ADB (Priority 3)

## Architecture (4 Layers)

- **Layer 1 — Developer Interface**: VS Code Extension (Windows), Arduino IDE Plugin (Linux), Android Studio Plugin (Android)
- **Layer 2 — Claude Code Agent (MCP Server)**: 5 MCP tools — `hardware_detect`, `profile_workload`, `convert_model`, `orchestrate_workload`, `generate_code`
- **Layer 3 — SDK Abstraction**: Wraps QNN, SNPE, QAIRT, Hexagon SDK, Adreno SDK, AIMET, AI Hub behind unified interfaces
- **Layer 4 — Hardware Execution**: CPU (Oryon/Kryo ARM64), GPU (Adreno), NPU (Hexagon HTP/DSP)

## MCP Tool Summary

| Tool | Purpose | Key I/O |
|------|---------|---------|
| `hardware_detect` | Detect chipset, CPU/GPU/NPU topology, memory | platform enum → JSON device profile |
| `profile_workload` | Run platform profiler, return structured metrics | model_path + runtime → latency/power/memory JSON |
| `convert_model` | Convert ONNX/PyTorch/TF to QNN/SNPE DLC | source_format + quantization → output DLC path |
| `orchestrate_workload` | Allocate inference ops across CPU/GPU/NPU | model + profile + power_mode → allocation map |
| `generate_code` | Emit platform-specific inference code | platform + sdk + language → source + build instructions |

## Key SDKs and Tools

- **QNN SDK / QAIRT 2.x** — Primary inference SDK (all platforms)
- **SNPE SDK 2.x** — Deep learning runtime (Mobile, Arduino UNO Q)
- **Hexagon SDK 5.x** — DSP/HTP programming (all platforms)
- **Adreno SDK** — GPU compute via OpenCL/Vulkan (AI PC, Mobile)
- **AIMET** — Model quantization/compression (AI PC, Mobile)
- **Qualcomm AI Hub** — Cloud profiling and optimization
- **Snapdragon Profiler / QPM3 / SNPE Profiler** — Platform-specific profiling tools

## Development Context

- The PRD targets **TTFI (Time-to-First-Inference) < 10 minutes** from cold start
- Quantization support: FP32, INT8, INT4 (INT4 via AIMET in Phase 4)
- Power targets: AI PC < 15W, Arduino < 3W, Mobile < 5W
- Phased roadmap: Phase 1 (AI PC, June 2026) → Phase 2 (Arduino, June-July) → Phase 3 (Mobile, Aug) → Phase 4 (cross-platform, Sep)

## Technology Stack

- **Language**: Python 3.10+ (3.11 recommended)
- **MCP Framework**: FastMCP (Python)
- **Testing**: pytest + pytest-asyncio
- **Package Management**: pip + venv, pip-tools for locking
- **Full prerequisites**: See `docs/PREREQUISITES.md`

## Related Repository

The sibling directory `/Users/pavanr/work/05/FWProfiling/` contains the Qualcomm WLAN firmware codebase (C/C++, SCons build, QTF test framework) which provides context for the hardware/firmware ecosystem these tools target.

---

## Generating Sample Applications

When asked to generate a sample app or demonstrate QUAD, follow this pattern:

### Standard workflow (reference: `examples/sample_app.py`)

```
1. hardware_detect(platform)                          → device specs
2. convert_model(model, format, quantization, layout) → .dlc path + notes
3. profile_workload(model, profiling_level="detailed")→ latency / power / memory
4. profile_workload(model, profiling_level="linting") → cycle counts + bottlenecks
5. orchestrate_workload(model, power_mode)            → per-layer CPU/GPU/NPU map
6. generate_code(platform, language, model)           → C++/Python source files
```

### Sample prompts

**Full workflow:**
> "Use QUAD to build a sample inference app for `<model>.onnx` on `<platform>`. Run all 5 MCP tools and save to `examples/<name>_app.py`"

**Profile only:**
> "Profile `<model>.dlc` using QUAD linting mode and show the HTP bottleneck ops"

**Compare platforms:**
> "Use QUAD to compare inference of `<model>.onnx` across Windows AI PC, Android, and Linux platforms"

**Generate code only:**
> "Use QUAD generate_code to emit C++ inference code for `<model>.dlc` on Windows"

### Tool parameter reference

| Parameter | Values | Notes |
|---|---|---|
| `platform` | `windows` `linux` `android` | Determines mock device profile |
| `source_format` | `onnx` `pytorch` `tensorflow` `tflite` | |
| `quantization` | `fp32` `int8` `int4` | int8 recommended for NPU |
| `input_layout` | `nchw` `nhwc` `auto` | PyTorch = nchw |
| `channel_order` | `rgb` `bgr` `auto` | Legacy Caffe models = bgr |
| `profiling_level` | `detailed` `linting` `qhas` | linting = HTP cycle counts |
| `power_mode` | `performance` `balanced` `efficiency` | |
| `language` | `cpp` `python` `kotlin` | |

---

## Project Status & Session Resume Protocol

> **When resuming work in this repository, read this section first to understand current state.**

### Vision: Unified AI computing platform for Qualcomm SoCs

QUAD is evolving from an MCP agent into a **full AI computing platform** for Qualcomm SoCs. The platform has 7 layers: Hardware → Compiler → Runtime → Libraries → Optimizer → Serve → DevX/Ecosystem.

### Critical Path

```
Runtime (Phase B) → Libraries (Phase C) → Serve (Phase F) → Ecosystem (Phase G)
```

### Success Metric

```
pip install qualcomm-ai-toolkit && quad quickstart
```
Must work from zero to running NPU inference in **< 5 minutes**.

### Current Phase: REAL SDK INTEGRATION (Blocked on SDK docs)

### Platform Phase Progress

| Phase | Status | Key Deliverables | Completion |
|-------|--------|-----------------|------------|
| Phase A — Foundation | COMPLETE | MCP Agent, 5 tools, mock mode, 90 tests | 100% |
| Phase B — Runtime & Compiler | COMPLETE | `quad.Device`, `quad.Tensor`, `quad.load()`, QUAD IR, QBin | 100% |
| Phase C — Libraries & Optimizer | COMPLETE | QualcommDNN, QualcommBLAS, graph fusion, quantization | 100% |
| Phase D — Deep Profiler | COMPLETE | Roofline, kernel-level, power profiling, memory analysis | 100% |
| Phase E — Kernels & Streams | COMPLETE | Python DSL → Hexagon, QUAD Graphs, HVX primitives | 100% |
| Phase F — Serve & Deploy | COMPLETE | Inference server, model zoo, deployment, deploy.sh | 100% |
| Phase G — Ecosystem | COMPLETE | CLI (quickstart/doctor/benchmark), VS Code setup, install.sh | 100% |
| Phase H — Community | NOT STARTED | Academy, certification, forums, hackathons | 0% |

> **All mock-mode phases complete. 375 tests passing. Real hardware blocked on SDK docs.**

### MCP Agent Real-Mode Status

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 0 — Planning | COMPLETE | 100% |
| Phase 1 — AI PC (Windows) | Mock done; QNN SDK blocked | 70% |
| Phase 2 — Arduino UNO Q | Mock done (via SNPE adapter); hardware blocked | 50% |
| Phase 3 — Mobile (Android) | Mock done; hardware blocked | 50% |
| Phase 4 — Cross-Platform | Not started | 0% |

### Phase 0 Checklist (Planning & Setup)

- [x] PRD reviewed and analyzed
- [x] Prerequisites document created (`docs/PREREQUISITES.md`)
- [x] CLAUDE.md initialized with status tracking
- [x] Design document created (`docs/Design_Document_QUAD_Agent.docx`)
- [x] Phased implementation guide created (`docs/IMPLEMENTATION_GUIDE.md`)
- [x] Cursor rules created (`.cursor/rules/quad.mdc`)
- [x] Copilot instructions created (`.github/copilot-instructions.md`)
- [x] Project scaffolding created (`src/`, `tests/`, `pyproject.toml`)
- [x] FastMCP server skeleton with stub tools
- [x] SDK adapter interface design (implemented in code)
- [x] Code generation template architecture (implemented in code)
- [ ] CI/CD pipeline configuration
- [ ] Development environment validated (verification checklist from prerequisites)

### Phase 1 Checklist (AI PC / Windows)

- [x] `hardware_detect` tool — detect Snapdragon X Elite, enumerate CPU/GPU/NPU
- [ ] QNN SDK adapter — model loading, context creation, graph execution (blocked: need SDK docs)
- [x] `convert_model` tool — ONNX/PyTorch/TF → QNN format with quantization
- [x] `profile_workload` tool — QPM3 + Snapdragon Profiler automation
- [x] `orchestrate_workload` tool — CPU/GPU/NPU allocation based on profiling
- [x] `generate_code` tool — C++/Python inference code for QNN runtime
- [ ] AIMET integration — INT8 quantization recommendations
- [ ] AI Hub integration — cloud benchmarking API
- [x] End-to-end workflow validation (TTFI < 5s in mock mode)
- [x] Unit + integration + E2E tests passing (90 tests)
- [x] VS Code extension scaffold (TypeScript, 5 commands, MCP client)

### Phase 2 Checklist (Arduino UNO Q / Linux)

- [ ] SNPE SDK adapter — DLC conversion, DSP runtime
- [ ] `hardware_detect` — QCS2210 discovery via SSH/serial
- [ ] `convert_model` — ONNX → SNPE DLC with INT8 quantization
- [ ] `profile_workload` — SNPE Profiler + Hexagon Profiler automation
- [ ] `orchestrate_workload` — DSP vs CPU allocation under 3W power budget
- [ ] `generate_code` — Arduino-compatible sketch + SNPE runtime integration
- [ ] Cross-compilation pipeline (host → ARM64 target)
- [ ] Deploy & validate on physical hardware

### Phase 3 Checklist (Mobile / Android)

- [ ] `hardware_detect` — ADB-based Snapdragon 8 Elite detection
- [ ] `convert_model` — SNPE DLC / QNN binary for Android target
- [ ] `profile_workload` — Snapdragon Profiler + Perfetto automation
- [ ] `orchestrate_workload` — Thermal-aware NPU/GPU/CPU switching
- [ ] `generate_code` — Android AAR (Kotlin/Java) with SNPE runtime
- [ ] MediaPipe integration option
- [ ] Power-profile scheduling (performance/balanced/efficiency modes)

### Phase 4 Checklist (Cross-Platform & Optimization)

- [ ] AIMET INT4 quantization pipeline
- [ ] AI Hub CI/CD integration (automated benchmarking on commit)
- [ ] Multi-model pipeline chaining (detection + classification + segmentation)
- [ ] Cross-platform performance dashboard
- [ ] Perfetto/Snapdragon Profiler trace configs for CI

---

### Active Context (Updated Each Session)

> **Last updated**: 2026-05-08
> **Version**: 0.4.0 | **Tests**: 2002 passing / 3 skipped / 0 failed | **Source files**: 120+ Python modules
> **Last action**: Overnight gap-closure session — closed 11 of 17 Tier-1/Tier-2 gaps from `docs/GAP_ANALYSIS.md` plus added a full UX layer (10 Claude Code skills, rich markdown formatters, suggestions engine, contextual tips system, MCP tool response enrichment). 11 phase commits, +191 tests, ~10k lines added. Full progress report at `docs/IMPLEMENTATION_PROGRESS.md`.
> **Next action**: User downloads QAIRT SDK from <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>, runs `./install.sh --qairt-archive ~/Downloads/qairt-X.Y.Z.zip`, then we can land real `QAIRTAdapter` parsers (qairt-converter / snpe-diagview / qnn-platform-validator stdout)
> **Blockers**: SDK CLI output format needed: (1) qairt-converter stdout on success/failure, (2) snpe-diagview text output schema, (3) qnn-platform-validator stdout
> **Critical path**: All mock phases ✅ → **Real SDK wiring** (current) → Physical device testing → PyPI release
> **Success metric**: `pip install qualcomm-ai-toolkit && quad quickstart` in < 5 minutes on real hardware
> **What works now**: Full mock-mode platform (1715 tests), all SDK doc sections integrated as structured Python modules, complete CLI builders for all 18 SDK tools, C++ code gen templates for QNN .so/.bin/.tflite pipelines
> **Decisions made**:
> - QUAD = full computing platform (not just MCP agent wrapper)
> - Name: QUAD (Qualcomm Unified Agent for Developers) — also "quad = 4 compute units"
> - Stack: Python 3.11 (FastMCP) for MCP server
> - Architecture: Mock-first with adapter pattern (mock ↔ real switching via config)
> - Differentiators vs CUDA: power-aware, heterogeneous (CPU+GPU+NPU), edge-native, AI agent
> - Platform layers: 7 (Hardware → Compiler → Runtime → Libraries → Optimizer → Serve → DevX)
> - Timeline: 12-14 months to CUDA platform parity
> - Open source strategy: Runtime + Agent = Apache 2.0; Kernels + Compiler backend = proprietary
> **Modules added since v0.2.0**:
>   - `accuracy/` — mAP, Top-1/5, AccuracyEvaluator
>   - `benchmarks/` — MobilenetSSD, snpe_bench.py, CSV/JSON result parsing
>   - `qnn/` — QNN SDK API table, .so/.bin/.tflite inference pipelines
>   - `sdk_tools/` — platform matrix (18 tools × 6 OS), full CLI builders, Architecture Checker, Accuracy Debugger (6 modes)
>   - `profiler/linting.py` — cycle-count parser, bottleneck analysis
>   - `profiler/qhas.py` — QHAS 3-step workflow, QHASConfig, chrometrace
>   - `profiler/diagview.py` — snpe-diagview wrapper
>   - `profiler/levels.py` — shared ProfilingLevel enum
>   - `models/profiling.py` — LintingLayerProfile, profiling_level field
>   - `models/conversion.py` — input_layout, channel_order, mean_values, conversion_notes
> **Pending (see TODO.md)**:
>   - Real adapter output parsers (blocked on SDK stdout format docs)
>   - AIMET INT8/INT4 integration
>   - AI Hub Python SDK integration
>   - CI/CD pipeline (GitHub Actions)
>   - Physical device testing (all 3 platforms)

---

### Session Resume Instructions

When starting a new session in this repo:

1. **Read this file** — understand current phase and last action taken
2. **Check Active Context** above — see what was last done and what's next
3. **Check phase checklists** — identify which items are complete vs pending
4. **Verify file state** — `ls src/ tests/` to confirm scaffolding matches expected state
5. **Continue from "Next action"** — pick up where the previous session left off

---

### Change Impact Checklist

**Every time you make a change, run through this checklist before committing.**

#### 1. New environment variable or secret?
- [ ] Add it to `.env.example` with a description and blank value
- [ ] Add it to `configs/quad.toml.example` if it's non-sensitive config
- [ ] Add it to `src/quad/cli/configure.py` — ask the user for it in the wizard
- [ ] Update `.claude/skills/quad-configure.md` if the wizard flow needs updating

#### 2. New SDK, tool, or external dependency?
- [ ] Add install step to the relevant `scripts/adapters/setup_<name>.sh`
- [ ] Add to `ADAPTERS_LIST` in `install.sh` if it needs a new adapter script
- [ ] Add verification check in `src/quad/cli/doctor.py`
- [ ] Document in `docs/PREREQUISITES.md`

#### 3. New CLI command or MCP tool?
- [ ] Register in `src/quad/cli/main.py` (CLI) or `src/quad/server/__init__.py` (MCP)
- [ ] Add permissions to `.claude/settings.json` if it's a new MCP tool
- [ ] Add to `docs/USAGE.md` with example input/output
- [ ] Update `README.md` MCP Tools table if applicable

#### 4. New template or code generation output?
- [ ] Add template to `templates/snpe/<platform>/` or appropriate subdirectory
- [ ] Register in `src/quad/codegen/engine.py` if it's a new language target
- [ ] Add test in `tests/unit/test_codegen/`

#### 5. New Python module or package?
- [ ] Add `__init__.py` with exports
- [ ] Add unit tests in `tests/unit/test_<module>/`
- [ ] Export from parent package `__init__.py` if public API

#### 6. Version bump?
- [ ] Update `pyproject.toml` → `version`
- [ ] Update `src/quad/__init__.py` → `__version__`
- [ ] Add changelog entry in `README.md` Changelog section

#### 7. After ANY change — always:
- [ ] Run `pytest tests/ -q` — all tests must pass
- [ ] Run `git add -A && git commit && git push`
- [ ] Update **Active Context** above (last action, next action, test count)

If the Active Context is stale or unclear:
- Run `git log --oneline -10` to see recent commits
- Check for any TODO/FIXME markers in code
- Review `docs/PREREQUISITES.md` for environment setup if starting fresh
