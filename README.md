# QUAD — Qualcomm Unified Agent for Developers

> Abstracting Qualcomm SDKs, Profilers & Hardware for Accelerated AI Development

[![Status](https://img.shields.io/badge/status-v0.3.0%20SDK%20Docs%20Integrated-green)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![Tests](https://img.shields.io/badge/tests-1715%20passing-brightgreen)]()
[![MCP](https://img.shields.io/badge/protocol-MCP-purple)]()
[![License](https://img.shields.io/badge/license-Qualcomm%20Confidential-red)]()

---

## What is QUAD?

**QUAD** (**Q**ualcomm **U**nified **A**gent for **D**evelopers) is a Claude Code-powered MCP (Model Context Protocol) server that gives AI developers a single conversational interface to Qualcomm's SDK ecosystem. Instead of learning QNN, SNPE, Hexagon SDK, Adreno SDK, AIMET, and multiple profilers independently, developers express intent in natural language and receive production-ready inference code, profiling reports, and optimization recommendations automatically.

**Target**: Time-to-First-Inference < 10 minutes from cold start.
**Vision**: The CUDA of Qualcomm — a full 7-layer AI computing platform for Qualcomm SoCs.

---

## Supported Platforms

| Platform | Chipset | NPU | Primary SDK | Mock Mode | Real Hardware |
|----------|---------|-----|-------------|-----------|--------------|
| AI PC (Windows) | Snapdragon X Elite (X1E-80-100) | 45 TOPS | QNN / QAIRT 2.x | ✅ Ready | Pending SDK output format |
| Arduino UNO Q (Linux) | QCS2210 (Robotics RB1) | ~1 TOPS DSP | SNPE 2.x | ✅ Ready | Pending hardware |
| Mobile (Android) | Snapdragon 8 Elite (SM8750) | 48 TOPS | SNPE / QNN 2.x | ✅ Ready | Pending hardware |

> **Mock Mode**: All tools return realistic simulated data — no hardware or SDK required.
> **Real Hardware**: Requires QAIRT SDK v2.45+ and physical device (see `docs/PREREQUISITES.md`).

---

## MCP Tools

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `hardware_detect` | Detect chipset, CPU/GPU/NPU topology, memory | platform | Device profile JSON |
| `convert_model` | Convert ONNX/PyTorch/TF → QNN/SNPE format | model_path, format, quantization, layout | DLC path + conversion notes + image format guidance |
| `profile_workload` | Run profiler (basic/detailed/linting/qhas) | model_path, runtime, profiling_level | Latency/power/memory report + per-op cycle data |
| `orchestrate_workload` | Allocate ops across CPU/GPU/NPU | model, profile, power_mode | Layer→runtime allocation map |
| `generate_code` | Emit platform-specific inference code | platform, language, model | Source files + build instructions |

---

## Quick Start

### Prerequisites

- Python 3.10+ (3.11 recommended)
- Claude Code CLI

### Installation

```bash
git clone <repo-url> && cd QUAD
./setup.sh
source .venv/bin/activate
```

`setup.sh` handles everything: virtual environment, dependencies, `quad.toml`, `.env`, and runs `quad doctor` to confirm the setup.

For real hardware mode (when SDK is available):
```bash
./setup.sh --real
```

### Running the Server

```bash
# Start MCP server (mock mode — no SDK required)
quad-server --config quad.toml
```

### Using with Claude Code

**No manual configuration needed.** The repository includes `.claude/settings.json` which Claude Code reads automatically. The QUAD MCP server is auto-registered with all 5 tools pre-approved.

```bash
./launch.sh    # Start the server, then use Claude Code normally
```

### CLI Commands

```bash
quad quickstart    # Interactive zero-to-inference wizard
quad doctor        # Environment diagnostics (13 checks including SDK env vars)
quad benchmark     # Standard benchmark suite
quad compile       # Compile ONNX/PT to .qbin
quad profile       # Profile a compiled model
quad detect        # List available devices
quad configure     # Interactive SDK configuration wizard
```

### Development Commands

```bash
make serve          # Start MCP server
make test           # Run all tests with coverage
make test-unit      # Unit tests only
make lint           # Ruff + mypy
make format         # Auto-format code
```

---

## Architecture (7 Layers)

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 7: DevX / Ecosystem                                       │
│  CLI (quickstart/doctor/benchmark)  VS Code Extension  Plugins  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Serve & Deploy                                         │
│  ModelServer  ModelRegistry  deploy_model()  Inference Server   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: MCP Agent (5 Tools via FastMCP)                        │
│  hardware_detect  convert_model  profile_workload                │
│  orchestrate_workload  generate_code                             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: SDK Abstraction (Adapter Pattern)                      │
│  MockAdapter ↔ QAIRTAdapter  [config toggle: mock | real]       │
│  SNPE/QNN/Hexagon/Adreno/AIMET/AIHub wrappers                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Libraries & Optimizer                                  │
│  QualcommDNN (Conv, MHA, Flash)  QualcommBLAS (GEMM)            │
│  Fusion / DCE / ConstantFold / MemoryPlanning passes            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Runtime & Compiler                                     │
│  Device  Tensor  Model  Stream  MemoryPool  PowerMonitor        │
│  QUAD IR (.qir)  QBin (.qbin)  ONNX frontend                    │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Hardware                                               │
│  CPU (Oryon/Kryo)   GPU (Adreno)   NPU (Hexagon HTP/DSP)        │
└─────────────────────────────────────────────────────────────────┘
```

**Design Philosophy**: Mock-first — all tools work without hardware or SDKs.
A single config toggle (`adapter_mode = "mock" | "real"`) switches backends.

---

## Project Structure

```
QUAD/
├── src/quad/
│   ├── server/           # FastMCP MCP Agent (5 tools)
│   ├── tools/            # MCP tool handlers
│   ├── adapters/         # Mock + QAIRTAdapter (real mode skeleton)
│   ├── runtime/          # Device, Tensor, Model, Stream, Memory, Power
│   ├── compiler/         # QUAD IR, ONNX frontend, QBin, model_conversion
│   ├── libs/             # QualcommDNN + QualcommBLAS
│   ├── optimizer/        # Graph fusion, DCE, memory planning
│   ├── profiler/         # Roofline, kernel, power, memory, linting, QHAS
│   ├── kernels/          # Python DSL → Hexagon, QUAD Graphs
│   ├── serve/            # Inference server, model zoo, deploy
│   ├── cli/              # quad CLI (quickstart/doctor/benchmark/configure)
│   ├── codegen/          # Jinja2 code generation engine
│   ├── models/           # Pydantic data models (all API boundaries)
│   ├── platforms/        # Windows, Linux, Android platform abstractions
│   ├── psnpe/            # Parallel SNPE (heterogeneous bulk inference)
│   ├── udo/              # User-Defined Operations (custom ops)
│   ├── accuracy/         # Inference accuracy metrics (mAP, Top-1/5)
│   ├── benchmarks/       # snpe_bench.py integration, MobilenetSSD
│   ├── qnn/              # QNN SDK API reference + inference pipelines
│   ├── sdk_tools/        # Full SDK tool reference + CLI builders
│   └── utils/            # SNPE logging, perf profiles, layer support
├── templates/
│   ├── snpe/cpp/         # C++ inference templates (UserBuffer, ITensor, cache)
│   ├── snpe/android/     # Android AAR, Kotlin, build.gradle
│   ├── snpe/profiling/   # QHAS config, linting script, input_list generator
│   ├── snpe/benchmarking/ # benchmark_config.json, imagelist.txt
│   ├── qnn/              # QNN C++ templates (.so, .bin, TFLite delegate)
│   ├── windows/          # Windows C++/Python inference
│   └── linux/            # Linux/Arduino Python/sketch inference
├── tests/                # 1715 passing (unit + integration + e2e)
├── docs/                 # PRD, Design Doc, Prerequisites, Implementation Guide
├── configs/              # quad.toml.example, .env.example, device profiles
├── scripts/adapters/     # setup_qairt.sh, setup_qnn.sh, setup_hexagon.sh
├── plugins/              # VS Code extension, Arduino IDE plugin, Android Studio
└── CLAUDE.md             # Claude Code session context & status tracking
```

---

## Configuration

```toml
# quad.toml
[server]
adapter_mode = "mock"         # "mock" or "real"
log_level = "info"
model_output_dir = "./output"

[adapters.qnn]
sdk_path = "C:/Qualcomm/AIStack/QNN"

[adapters.snpe]
sdk_path = "/opt/snpe-2.x"

[adapters.ai_hub]
api_key_env = "QAI_HUB_API_KEY"

[platforms.linux]
ssh_host = "arduino-uno-q.local"
ssh_user = "root"
```

Environment variables (from `.env.example`):
```bash
QAIRT_SDK_ROOT=         # QNN/QAIRT SDK installation path
QAI_HUB_API_KEY=        # Qualcomm AI Hub API token
ANDROID_SERIAL=         # ADB device serial number
ADSP_LIBRARY_PATH=      # DSP/HTP skel library path
```

---

## SDK Documentation Coverage

All major SNPE/QAIRT SDK documentation sections have been integrated as structured
Python modules with reference notes, CLI builders, and full test coverage:

| SDK Doc Section | Module | Tests |
|---|---|---|
| Model Conversion (ONNX/TF/PyTorch/TFLite/QAIRT) | `compiler/model_conversion.py` | 125 |
| Input Image Formatting (NHWC, BGR, batch concat) | `compiler/model_conversion.py` | +29 |
| HTP Offline Caching + SoC compatibility matrix | `compiler/model_conversion.py` | included |
| Quantization notes + calibration data rules | `compiler/model_conversion.py` | included |
| Profiling levels (basic/detailed/linting/qhas) | `profiler/levels.py` | 71 |
| HTP Linting Profile — per-op cycle analysis | `profiler/linting.py` | 60 |
| QHAS Profiling — full 3-step workflow | `profiler/qhas.py` | 68 |
| snpe-diagview wrapper + parser | `profiler/diagview.py` | included |
| DSP environment (ADSP_LIBRARY_PATH, skel naming) | `adapters/dsp_env.py` | included |
| SNPE Perf Profile API (lifecycle-aware DSP perf) | `utils/perf_profile.py` | included |
| Layer support matrix (134 ops × 5 backends) | `utils/layer_support.py` | included |
| MobilenetSSD Benchmarking (snpe_bench.py) | `benchmarks/` | 66 |
| Inference Accuracy (mAP, Top-1/5 error rate) | `accuracy/` | 58 |
| SDK Tools platform matrix (18 tools × 6 OS) | `sdk_tools/platform_matrix.py` | 96 |
| Full converter/quantizer/runtime CLI flags | `sdk_tools/tool_specs.py` | included |
| Architecture Checker + CSV parser | `sdk_tools/architecture_checker.py` | 53 |
| Accuracy Debugger (6 modes, config models) | `sdk_tools/accuracy_debugger.py` | 88 |
| QNN SDK API (11 components, 3 pipelines) | `qnn/` | 53 |
| PSNPE (parallel heterogeneous inference) | `psnpe/` | included |
| UDO (User-Defined Operations) | `udo/` | included |

---

## Development Roadmap

### Platform Phases (Mock Mode) — ALL COMPLETE

| Phase | Focus | Status |
|-------|-------|--------|
| A — Foundation | MCP Agent, 5 tools, mock adapter | ✅ Complete |
| B — Runtime & Compiler | Device, Tensor, QUAD IR, QBin | ✅ Complete |
| C — Libraries & Optimizer | QualcommDNN, BLAS, fusion passes | ✅ Complete |
| D — Deep Profiler | Roofline, kernel, power, memory | ✅ Complete |
| E — Kernels & Streams | Python DSL → Hexagon, QUAD Graphs | ✅ Complete |
| F — Serve & Deploy | Inference server, model zoo | ✅ Complete |
| G — Ecosystem | CLI: quickstart, doctor, benchmark | ✅ Complete |

### Real SDK Integration — IN PROGRESS

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1 | AI PC / Windows (QNN SDK, QPM3) | 70% — Mock done, real adapter needs SDK output format |
| Phase 2 | Arduino UNO Q / Linux (SNPE, DSP) | 50% — Mock done, hardware blocked |
| Phase 3 | Mobile / Android (SNPE/QNN, ADB) | 50% — Mock done, hardware blocked |
| Phase 4 | Cross-Platform (AIMET INT4, AI Hub) | 0% — Not started |

---

## Current Status

> **Version**: 0.3.0
> **Tests**: 1715 passing (57 skipped)
> **Source files**: 108 Python modules
> **Last Updated**: 2026-05-04

### What Works Now

- Full mock-mode platform — all 5 MCP tools work without hardware or SDKs
- `quad quickstart` → zero-to-inference in < 5 minutes (mock)
- `quad doctor` → 13-check environment diagnostics including SDK env vars
- `quad configure` → interactive wizard for SDK paths and API keys
- All SNPE/QAIRT SDK documentation sections integrated as structured Python modules
- Complete CLI builders for all 18 SDK tools across 6 platforms
- Architecture Checker and Accuracy Debugger integration
- QNN SDK API reference with 3-format inference pipeline documentation
- C++ code generation templates for .so, .bin, and TFLite delegate pipelines

### What Remains

- Real `QAIRTAdapter` wiring (blocked on SDK CLI output format documentation)
- Physical device testing (Snapdragon X Elite, QCS2210, SM8750)
- AIMET INT8/INT4 quantization integration
- Qualcomm AI Hub Python SDK integration
- CI/CD pipeline (GitHub Actions)
- PyPI publication (`pip install qualcomm-ai-toolkit`)

---

## Testing

```bash
# Run full test suite
make test

# Run with coverage report
pytest tests/ -v --cov=src/quad --cov-report=html

# Run specific module tests
pytest tests/unit/test_compiler/    # Compiler + model conversion
pytest tests/unit/test_profiler/    # Linting, QHAS, diagview
pytest tests/unit/test_sdk_tools/   # Architecture Checker, Accuracy Debugger
pytest tests/unit/test_qnn/         # QNN API reference
pytest tests/unit/test_benchmarks/  # MobilenetSSD benchmarking
pytest tests/unit/test_accuracy/    # mAP, Top-1/5 metrics
```

**Coverage Target**: ≥ 85% line coverage for `src/quad/`

---

## SDK Reference

| SDK / Tool | Version | Purpose | Platforms |
|------------|---------|---------|-----------|
| QNN SDK / QAIRT | 2.45+ | Neural network inference & quantization | All |
| SNPE SDK | 2.x | Deep learning inference runtime | Mobile, Linux |
| Hexagon SDK | 5.x | DSP/HTP programming | All |
| Adreno SDK | Latest | GPU compute (OpenCL/Vulkan) | AI PC, Mobile |
| AIMET | Latest | Model quantization & compression | AI PC, Mobile |
| Qualcomm AI Hub | Cloud | Model profiling & optimization | All |
| Snapdragon Profiler | Latest | CPU/GPU/NPU trace capture | AI PC, Mobile |
| QPM3 | Latest | Power measurement & analysis | AI PC |
| snpe-architecture-checker | Bundled | HTP model optimization analysis | All |
| snpe-accuracy-debugger | Bundled | Layer-level accuracy comparison | Linux, Windows (CPU only) |

---

## Contributing

### Setup

```bash
pip install -e ".[dev]"
pre-commit install
make lint && make test
```

### Code Standards

- Python 3.11+ with type hints (mypy --strict)
- Formatting: ruff format
- Linting: ruff check
- All SDK calls through adapter layer
- Pydantic models at all API boundaries
- structlog for logging (no print/logging module)
- Tests required for all new functionality

### Adding a New SDK Documentation Section

When a new SDK doc section is shared:
1. Create `src/quad/<module>/<section>.py` with reference notes + helpers
2. Add `__init__.py` exports
3. Write tests in `tests/unit/test_<module>/`
4. Run `pytest -x -q` to verify
5. Commit with descriptive message and test count

---

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| PRD | `docs/PRD_Qualcomm_DevWorkflows_v3.docx` | Product requirements |
| Design Spec | `docs/Design_Document_QUAD_Agent.docx` | Engineering specification |
| Prerequisites | `docs/PREREQUISITES.md` | SDK, hardware, account requirements |
| Implementation Guide | `docs/IMPLEMENTATION_GUIDE.md` | Sprint-by-sprint build plan |
| Claude Code Context | `CLAUDE.md` | AI assistant context & project status |
| Pending Tasks | `TODO.md` | All pending items by priority |

---

## Changelog

### [0.3.0] — 2026-05-04 — SDK Documentation Integration

#### SDK Reference Modules
- `quad.accuracy` — mAP, Top-1/5 error rate, `AccuracyEvaluator` (exact SNPE AP algorithm)
- `quad.benchmarks` — MobilenetSSD benchmarking: `SNPEBenchmarkConfig`, `snpe_bench.py` integration, CSV/JSON result parsing
- `quad.qnn` — QNN SDK API component table (11 components), 3-format inference pipelines (.so, .bin, .tflite)
- `quad.sdk_tools` — Full SDK tools reference: 18-tool platform matrix, `OnnxConverterArgs`, `QairtConverterArgs`, `QairtQuantizerArgs`, `SnpeNetRunArgs`, `DiagviewArgs`, `ArchCheckerArgs`, all 6 Accuracy Debugger mode builders
- `quad.profiler.linting` — HTP linting profiler: cycle-count parser, bottleneck analysis, Sub→Conv/Div→Mul/PReLU→ReLU substitutions
- `quad.profiler.qhas` — QHAS 3-step profiling workflow: `QHASConfig`, `QHASWorkflow`, chrometrace generation
- `quad.profiler.diagview` — `snpe-diagview` wrapper + `parse_diaglog_as_linting()`
- `quad.profiler.levels` — Shared `ProfilingLevel` enum (off/basic/detailed/linting/qhas)

#### Gap Fixes (All 8 Categories)
- `ProfileRequest` gains `profiling_level`, `htp_soc`, `sdk_root` fields
- `ProfilingReport` gains `LintingLayerProfile` list, cycle counts, QHAS chrometrace path
- `ConversionRequest` gains `input_layout` (nhwc/nchw/auto), `channel_order` (rgb/bgr/auto), `mean_values`
- `ConversionResult` gains `conversion_notes` (MODEL_TIPS) and `image_format_notes`
- `MockAdapter` simulates linting data (sub-op bottleneck) and QHAS chrometrace
- `QAIRTAdapter.profile()` dispatches to `_profile_linting` / `_profile_qhas` / `_profile_standard`
- `doctor.py` gains 5 new checks: SDK env vars, tools in PATH, ADSP_LIBRARY_PATH, Android tools, QHAS prerequisites
- New templates: `qhas_config.json.j2`, `run_linting.sh.j2`, `generate_input_list.py.j2`

#### Input Image Formatting
- `convert_nchw_to_nhwc()`, `convert_channel_order()`, `prepare_batch_input()` helpers
- `IMAGE_FORMAT_NOTES` reference dict covering NHWC/NCHW, BGR/RGB, MNIST 4D tensor, AlexNet example

### [0.2.0] — 2026-04-30 — Platform Complete (Mock Mode)
- All 7 platform phases complete (Phases A–G)
- 530 tests passing at release

### [0.1.0] — 2026-04-15 — Foundation
- MCP Agent with 5 tools, mock adapter, project scaffold
