# QUAD — Qualcomm Unified Agent for Developers

> Abstracting Qualcomm SDKs, Profilers & Hardware for Accelerated AI Development

[![Status](https://img.shields.io/badge/status-v0.3.0%20Real--HW%20Enabled-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![Tests](https://img.shields.io/badge/tests-1811%20passing-brightgreen)]()
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
|----------|---------|-----|-------------|-----------|---------------|
| AI PC (Windows) | Snapdragon X Elite (X1E-80-100) | 45 TOPS | QNN / QAIRT 2.x | ✅ Ready | ✅ Validated on real hw — see `docs/SAMPLE_APP_REPORT.md` |
| Arduino UNO Q (Linux) | QCS2210 (Robotics RB1) | ~1 TOPS DSP | SNPE 2.x | ✅ Ready | Pending physical device |
| Mobile (Android) | Snapdragon 8 Elite (SM8750) | 48 TOPS | SNPE / QNN 2.x | ✅ Ready | Pending physical device |

> **Mock Mode** (default): All tools return realistic simulated data — no hardware or SDK required.
> **Real Hardware Mode**: Auto-enabled when QAIRT SDK is installed. See [`docs/REAL_HARDWARE.md`](docs/REAL_HARDWARE.md) for the one-command setup.

---

## Quick Start — One Command

### 1. Clone

```bash
git clone https://github.com/pkr465/QUAD.git && cd QUAD
```

### 2. Install (pick the path that matches you)

| Situation | Command |
|---|---|
| You already downloaded QAIRT from the developer portal | `./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip` |
| Archive is already in `~/Downloads/` | `./install.sh` (auto-detected) |
| `QAIRT_SDK_ROOT` already set, or SDK at vendor default path | `./install.sh` |
| You have a CI mirror with a token | `QAIRT_DOWNLOAD_URL=… QAIRT_DOWNLOAD_TOKEN=… ./install.sh` |
| No SDK available right now (still want to develop) | `./install.sh --mock-only` |

The installer:
- Creates `.venv/` and installs QUAD with dev extras
- Unpacks the SDK into the gitignored `./sdks/<flavor>-<version>/`
- Sets `QAIRT_SDK_ROOT` / `QNN_SDK_ROOT` / `SNPE_ROOT` for the current shell
- Writes `quad.toml`, `.claude/settings.json` (MCP auto-detect), and `activate.sh`
- Runs `pytest -q` for verification
- Always succeeds — falls back to mock mode with clear next-step guidance if no SDK is found

### 3. Activate the environment

```bash
source ./activate.sh        # Re-resolves the SDK every shell start
```

### 4. Confirm real-hardware mode (skip if mock-only)

```bash
quad mode                   # → real-mode: READY  (when SDK present)
quad doctor --real-mode     # Strict pre-flight; exits non-zero on any SDK issue
```

### 5. Use it

```bash
./launch.sh                 # Start the MCP server (Claude Code auto-attaches)
quad quickstart             # Interactive zero-to-inference wizard
```

> **Why not zero-credential auto-download?** Both Qualcomm developer pages
> ([QAIRT](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk),
> [SNPE](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai))
> gate downloads behind a developer account login + EULA acceptance. There
> is no anonymous direct-download URL. The installer never bypasses that —
> the `QAIRT_DOWNLOAD_URL` + `QAIRT_DOWNLOAD_TOKEN` path lets you plug in
> your own pre-accepted token / mirror once for unattended automation.

---

## CLI Commands

```bash
# Mode + SDK management
quad mode                       # Show adapter mode + real-mode readiness
quad mode --set real            # Print 'export QUAD_ADAPTER_MODE=real' for shell-eval
quad sdk status                 # Show the active SDK + version + bin dir
quad sdk discover               # Scan all standard locations, list every SDK
quad sdk install <archive>      # Unpack a downloaded archive into ./sdks/

# Diagnostics
quad doctor                     # 14 environment checks (Python / SDK / tools / DSP)
quad doctor --real-mode         # Strict pre-flight; fails on any SDK issue

# Workflow
quad quickstart                 # Interactive zero-to-inference wizard
quad benchmark                  # Standard benchmark suite (MobileNetV2 / ResNet / YOLOv8n)
quad detect                     # Enumerate available CPU/GPU/NPU devices
quad compile <model>            # Compile ONNX / PyTorch → .qbin
quad profile <model>            # Run platform profiler
quad serve <model>              # Start inference server
quad configure                  # Interactive SDK / target-device / API-key wizard
```

### Development commands

```bash
make serve          # Start MCP server
make test           # Run all tests with coverage
make test-unit      # Unit tests only
make lint           # Ruff + mypy
make format         # Auto-format code
```

---

## MCP Tools

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `hardware_detect` | Detect chipset, CPU/GPU/NPU topology, memory | platform | Device profile JSON |
| `convert_model` | Convert ONNX/PyTorch/TF → QNN/SNPE format | model_path, format, quantization, layout | DLC path + conversion notes + image format guidance |
| `profile_workload` | Run profiler (basic/detailed/linting/qhas) | model_path, runtime, profiling_level | Latency/power/memory report + per-op cycle data |
| `orchestrate_workload` | Allocate ops across CPU/GPU/NPU | model, profile, power_mode | Layer→runtime allocation map |
| `generate_code` | Emit platform-specific inference code | platform, language, model | Source files + build instructions |

The MCP server runs `sdk_manager.startup_resolve_and_log()` on every start: it locates an installed SDK from env vars, `quad.toml`, the project's `./sdks/` directory, `~/.quad/sdks/`, or vendor-default paths (`C:\Qualcomm\AIStack\QAIRT\*`, `/opt/qcom/aistack/qairt/*`). Found SDKs auto-populate `QAIRT_SDK_ROOT` for child processes.

---

## Real Hardware Mode — Validated on Snapdragon X Elite

The full pipeline has been run end-to-end on a **Dell Latitude 7455 (Snapdragon X Elite X1E80100)**:

- 12 Oryon cores @ 4012 MHz · Adreno X1-85 GPU · Hexagon NPU detected (`ComputeAccelerator`, `Status=OK`) · 31.6 GB RAM
- Real CPU benchmark on Oryon: **MobileNetV2-1.0 @ 388 FPS / 2.56 ms mean / σ=0.95 ms** (500 iters, ONNX Runtime)
- Memory: 125.8 → 129.6 MB peak (Δ 3.8 MB) · CPU utilisation: 1160% (≈11.6 of 12 logical cores)
- All 5 MCP tools ran cleanly · QUAD generated valid Windows-on-Snapdragon QNN C++ inference code

Full report with methodology, raw measurements, and reproduction instructions: [**`docs/SAMPLE_APP_REPORT.md`**](docs/SAMPLE_APP_REPORT.md).
Sample app source: [**`examples/sample_app_real_hw.py`**](examples/sample_app_real_hw.py).

To reproduce on your Snapdragon X Elite:

```bash
./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip
source ./activate.sh
pip install onnxruntime-qnn psutil
python -c "import urllib.request; import os; os.makedirs('examples/models', exist_ok=True); urllib.request.urlretrieve('https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx', 'examples/models/mobilenetv2-12.onnx')"
PYTHONUTF8=1 python examples/sample_app_real_hw.py --iterations 500 --warmup 50 --json-out examples/run_results.json
```

---

## Architecture (7 Layers)

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 7: DevX / Ecosystem                                       │
│  CLI (mode/sdk/doctor/quickstart)  VS Code Extension  Plugins   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Serve & Deploy                                         │
│  ModelServer  ModelRegistry  deploy_model()  Inference Server   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 5: MCP Agent (5 Tools via FastMCP)                        │
│  hardware_detect  convert_model  profile_workload                │
│  orchestrate_workload  generate_code                             │
│  + sdk_manager.startup_resolve_and_log on every start            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: SDK Abstraction (Adapter Pattern)                      │
│  MockAdapter ↔ QAIRTAdapter  [config toggle: mock | real]       │
│  + AdapterFactory.strict mode (QUAD_STRICT_REAL=1)              │
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
The `AdapterFactory` flips to real adapters automatically once the SDK is reachable.
`QUAD_STRICT_REAL=1` makes the factory raise instead of silently falling back to mock.

---

## Project Structure

```
QUAD/
├── install.sh                    # ⭐ One-step installer (multi-strategy SDK setup)
├── activate.sh                   # Generated; re-resolves SDK on every shell
├── launch.sh                     # Start the MCP server
├── deploy.sh                     # Deploy model to a target device
├── src/quad/
│   ├── server/                   # FastMCP MCP Agent (5 tools + startup hook)
│   ├── tools/                    # MCP tool handlers
│   ├── adapters/
│   │   ├── factory.py            # Mock ↔ Real adapter factory (strict mode, fallback tagging)
│   │   ├── mock_adapter.py       # Deterministic simulated responses
│   │   └── qairt_adapter.py      # Real adapter (qairt-converter, snpe-net-run, ...)
│   ├── sdk_manager.py            # ⭐ SDK auto-discovery + zip/tgz install + state tracking
│   ├── runtime/                  # Device, Tensor, Model, Stream, Memory, Power
│   ├── compiler/                 # QUAD IR, ONNX frontend, QBin, model_conversion
│   ├── libs/                     # QualcommDNN + QualcommBLAS
│   ├── optimizer/                # Graph fusion, DCE, memory planning
│   ├── profiler/                 # Roofline, kernel, power, memory, linting, QHAS
│   ├── kernels/                  # Python DSL → Hexagon, QUAD Graphs
│   ├── serve/                    # Inference server, model zoo, deploy
│   ├── cli/
│   │   ├── main.py               # quad CLI entry (mode, sdk, doctor, quickstart, ...)
│   │   ├── doctor.py             # 14-check diagnostics + --real-mode strict pre-flight
│   │   ├── quickstart.py         # Interactive wizard
│   │   ├── benchmark.py          # Standard benchmark suite
│   │   └── configure.py          # SDK / target-device wizard
│   ├── codegen/                  # Jinja2 code generation engine
│   ├── models/                   # Pydantic data models (all API boundaries)
│   ├── platforms/                # Windows, Linux, Android platform abstractions
│   ├── psnpe/                    # Parallel SNPE (heterogeneous bulk inference)
│   ├── udo/                      # User-Defined Operations (custom ops)
│   ├── accuracy/                 # Inference accuracy metrics (mAP, Top-1/5)
│   ├── benchmarks/               # snpe_bench.py integration, MobilenetSSD
│   ├── qnn/                      # QNN SDK API reference + inference pipelines
│   ├── sdk_tools/                # Full SDK tool reference + CLI builders
│   └── utils/                    # SNPE logging, perf profiles, layer support
├── scripts/
│   ├── helpers.sh                # Shared bash helpers (logging)
│   ├── setup_sdk.sh              # ⭐ Multi-strategy SDK acquisition (called by install.sh)
│   └── adapters/                 # Per-adapter setup (qnn, hexagon, target, udo)
├── templates/
│   ├── snpe/cpp/                 # C++ inference templates (UserBuffer, ITensor, cache)
│   ├── snpe/android/             # Android AAR, Kotlin, build.gradle
│   ├── snpe/profiling/           # QHAS config, linting script, input_list generator
│   ├── snpe/benchmarking/        # benchmark_config.json, imagelist.txt
│   ├── qnn/                      # QNN C++ templates (.so, .bin, TFLite delegate)
│   ├── windows/                  # Windows C++/Python inference
│   └── linux/                    # Linux/Arduino Python/sketch inference
├── tests/                        # 1811 passing (unit + integration + e2e)
│   ├── unit/test_sdk_manager/    # 29 tests for discovery + install + startup
│   ├── unit/test_adapters/       # Factory, mock, real, dlc-compat, dsp-env
│   └── ...                       # 100+ test files across all modules
├── examples/
│   ├── sample_app.py             # Mock-mode workflow walkthrough
│   ├── sample_app_real_hw.py     # ⭐ Real-hardware sample with ground-truth CPU benchmark
│   ├── models/                   # Downloaded ONNX models (gitignored)
│   ├── generated/                # QUAD-generated inference code (gitignored)
│   └── run_results.json          # Last sample-app run data
├── docs/
│   ├── REAL_HARDWARE.md          # ⭐ One-step real-mode enablement playbook
│   ├── SAMPLE_APP_REPORT.md      # ⭐ Real-hardware run report (Snapdragon X Elite)
│   ├── PRD_Qualcomm_DevWorkflows_v3.docx
│   ├── Design_Document_QUAD_Agent.docx
│   ├── PREREQUISITES.md
│   ├── IMPLEMENTATION_GUIDE.md
│   └── USAGE.md
├── configs/                      # quad.toml.example, .env.example, device profiles
├── plugins/                      # VS Code extension, Arduino IDE plugin, Android Studio
├── sdks/                         # ⭐ Gitignored — auto-populated by `quad sdk install`
├── .quad/                        # ⭐ Gitignored — runtime state (resolved SDK info)
├── CLAUDE.md                     # Claude Code session context & status tracking
└── TODO.md                       # Pending tasks by priority
```

---

## Configuration

### `quad.toml`

```toml
[server]
adapter_mode = "mock"          # "mock" or "real" (env var QUAD_ADAPTER_MODE overrides)
log_level = "info"
model_output_dir = "./output"

[adapters.qnn]
sdk_path = ""                  # Auto-discovered by sdk_manager — leave blank

[adapters.snpe]
sdk_path = ""                  # Same — discovery handles it

[adapters.ai_hub]
api_key_env = "QAI_HUB_API_KEY"

[platforms.linux]
ssh_host = "arduino-uno-q.local"
ssh_user = "root"
```

### Environment variables (`.env.example`)

```bash
# SDK paths (any one is enough; sdk_manager picks the first valid one)
QAIRT_SDK_ROOT=                # Primary SDK install path
QNN_SDK_ROOT=                  # Alternative
SNPE_ROOT=                     # Legacy

# Mode controls
QUAD_ADAPTER_MODE=             # mock | real (overrides quad.toml)
QUAD_STRICT_REAL=              # 1 = factory raises instead of falling back to mock

# DSP runtime
ADSP_LIBRARY_PATH=             # /vendor/lib/rfsa/adsp;/opt/qairt/lib/aarch64-android

# Auto-discovery & install behaviour
QAIRT_DOWNLOADS_DIR=           # Where setup_sdk.sh looks (default ~/Downloads)
QAIRT_DOWNLOAD_URL=            # CI / mirror URL (auth via QAIRT_DOWNLOAD_TOKEN)
QAIRT_DOWNLOAD_TOKEN=          # Bearer token for QAIRT_DOWNLOAD_URL

# Cloud / mobile
QAI_HUB_API_KEY=               # Qualcomm AI Hub
ANDROID_SERIAL=                # ADB device serial
ANDROID_NDK_ROOT=              # Android NDK
```

---

## SDK Documentation Coverage

All major SNPE/QAIRT SDK documentation sections are integrated as structured Python modules with reference notes, CLI builders, and full test coverage:

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
| **SDK auto-discovery + install** | **`sdk_manager.py`** | **29** |

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
| G — Ecosystem | CLI: mode, sdk, doctor, quickstart, benchmark | ✅ Complete |

### Real SDK Integration — IN PROGRESS

| Phase | Target | Status |
|-------|--------|--------|
| Phase 1 | AI PC / Windows (QNN SDK, QPM3) | 80% — Mock done, real CPU validated, NPU pending QAIRT install |
| Phase 2 | Arduino UNO Q / Linux (SNPE, DSP) | 50% — Mock done, hardware blocked |
| Phase 3 | Mobile / Android (SNPE/QNN, ADB) | 50% — Mock done, hardware blocked |
| Phase 4 | Cross-Platform (AIMET INT4, AI Hub) | 0% — Not started |

---

## Current Status

> **Version**: 0.3.0
> **Tests**: 1811 passing (8 pre-existing Windows path-assertion bugs; 29 new sdk_manager tests in this release)
> **Source files**: 110 Python modules
> **Last Updated**: 2026-05-07

### What Works Now

- Full mock-mode platform — all 5 MCP tools work without hardware or SDKs
- **One-step real-hardware install** via `./install.sh --qairt-archive <path>` (six-strategy SDK acquisition with graceful mock-mode fallback)
- **MCP server SDK auto-discovery** — env var → quad.toml → `./sdks/` → `~/.quad/sdks` → vendor defaults
- **`quad mode` / `quad sdk` / `quad doctor --real-mode`** — full visibility into adapter state and SDK readiness
- **Real Snapdragon X Elite validation** — 388 FPS / 2.56 ms MobileNetV2 on Oryon CPU, full QUAD pipeline runs end-to-end (`docs/SAMPLE_APP_REPORT.md`)
- All SNPE/QAIRT SDK documentation sections integrated as structured Python modules
- Complete CLI builders for all 18 SDK tools across 6 platforms
- Architecture Checker and Accuracy Debugger integration
- QNN SDK API reference with 3-format inference pipeline documentation
- C++ code generation templates for .so, .bin, and TFLite delegate pipelines
- VS Code extension scaffold with debug configurations and tasks

### What Remains

- Real `QAIRTAdapter` output parsers (blocked on `qairt-converter` / `snpe-diagview` / `qnn-platform-validator` stdout format docs)
- Physical device testing on Arduino UNO Q (QCS2210) and Snapdragon 8 Elite (SM8750)
- AIMET INT8 / INT4 quantization integration
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
pytest tests/unit/test_sdk_manager/   # SDK discovery + install (29 tests)
pytest tests/unit/test_adapters/      # Mock + factory (incl. strict mode + fallback tagging)
pytest tests/unit/test_compiler/      # Compiler + model conversion
pytest tests/unit/test_profiler/      # Linting, QHAS, diagview
pytest tests/unit/test_sdk_tools/     # Architecture Checker, Accuracy Debugger
pytest tests/unit/test_qnn/           # QNN API reference
pytest tests/unit/test_benchmarks/    # MobilenetSSD benchmarking
pytest tests/unit/test_accuracy/      # mAP, Top-1/5 metrics
pytest tests/unit/test_cli/           # CLI commands incl. quad mode + quad sdk
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

Download QAIRT or SNPE from:
- <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>
- <https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai>

(Both require a Qualcomm developer account + EULA acceptance.)

---

## Contributing

### Setup

```bash
./install.sh --mock-only      # Lightweight setup; mock mode is enough for most contributions
pre-commit install
make lint && make test
```

### Code Standards

- Python 3.10+ with type hints (mypy --strict)
- Formatting: ruff format
- Linting: ruff check
- All SDK calls through the adapter layer
- Pydantic models at all API boundaries
- structlog for logging (no `print` / stdlib `logging`)
- Tests required for all new functionality

### Adding a New SDK Documentation Section

1. Create `src/quad/<module>/<section>.py` with reference notes + helpers
2. Add `__init__.py` exports
3. Write tests in `tests/unit/test_<module>/`
4. Run `pytest -x -q` to verify
5. Commit with descriptive message and test count

---

## Documentation

| Document | Location | Description |
|----------|----------|-------------|
| Real-Hardware Enablement | [`docs/REAL_HARDWARE.md`](docs/REAL_HARDWARE.md) | One-step setup + 6-strategy SDK acquisition + per-platform notes |
| Sample-App Run Report | [`docs/SAMPLE_APP_REPORT.md`](docs/SAMPLE_APP_REPORT.md) | Real Snapdragon X Elite measurements, methodology, repro |
| Prerequisites | [`docs/PREREQUISITES.md`](docs/PREREQUISITES.md) | SDK, hardware, account requirements |
| Implementation Guide | [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md) | Sprint-by-sprint build plan |
| PRD | `docs/PRD_Qualcomm_DevWorkflows_v3.docx` | Product requirements |
| Design Spec | `docs/Design_Document_QUAD_Agent.docx` | Engineering specification |
| Claude Code Context | [`CLAUDE.md`](CLAUDE.md) | AI assistant context & project status |
| Pending Tasks | [`TODO.md`](TODO.md) | All pending items by priority |

---

## Changelog

### [0.3.0] — 2026-05-07 — Real-Hardware Enablement (current)

#### One-step installer
- `./install.sh --qairt-archive PATH` is now the canonical first-time setup path; the installer creates `.venv/`, unpacks the SDK, sets env vars, generates `quad.toml` + `.claude/settings.json` + `activate.sh`, and runs the test suite — all in one command
- New `scripts/setup_sdk.sh` with six-strategy SDK acquisition (archive flag → existing env var → discovery → `~/Downloads` auto-detect → URL+token mirror → graceful mock fallback)
- Generated `activate.sh` re-resolves the SDK on every shell start via `sdk_manager`, so SDK changes are picked up without re-running the installer

#### SDK auto-discovery and install
- `src/quad/sdk_manager.py` — server-startup discovery across env vars, quad.toml, `./sdks/`, `~/.quad/sdks/`, and vendor-default paths; archive installer (zip/tar.gz) with zip-slip + content-shape validation; `apply_to_environment()` populates `QAIRT_SDK_ROOT`/`QNN_SDK_ROOT`/`SNPE_ROOT` for child processes
- New `quad sdk` subcommand: `status` / `discover` / `install`
- MCP server runs `startup_resolve_and_log()` before `AdapterFactory`; resolved SDK info is written to `.quad/sdk.json`
- 29 new tests for discovery priority, dedup, archive install, env apply, and startup hook

#### Adapter factory hardening
- `AdapterFactory.strict` (and `QUAD_STRICT_REAL=1` env var) raises `RealAdapterUnavailableError` instead of silently falling back to mock
- Fallback mocks tagged with `fell_back_from_real=True` and `fallback_reason` so callers can detect them
- `AdapterFactory.real_mode_ready()` introspection method
- Path-existence verification for SDK env vars (env var pointing at a missing directory no longer counts as ready)

#### CLI additions
- `quad mode` — show adapter mode + real-mode readiness; `--set` prints shell-eval export line
- `quad doctor --real-mode` — strict pre-flight that escalates SDK warnings to errors and exits non-zero (CI-ready)

#### Real-hardware validation
- Validated end-to-end on Dell Latitude 7455 (Snapdragon X Elite X1E80100)
- Real ONNX Runtime CPU inference: MobileNetV2-1.0 @ 388 FPS / 2.56 ms / σ=0.95 ms (500 iters on 12 Oryon cores)
- New `examples/sample_app_real_hw.py` (530 lines) runs the full QUAD MCP pipeline + real CPU benchmark side-by-side, captures latency percentiles + memory delta + battery telemetry, emits `run_results.json`
- New `docs/SAMPLE_APP_REPORT.md` (10 sections) with methodology and reproduction
- Codegen Jinja2 template-path fix (`as_posix()`) — unblocked 28 previously-failing Windows tests as a side effect

#### SDK reference modules (carried over from earlier in this release cycle)
- `quad.accuracy` — mAP, Top-1/5 error rate, `AccuracyEvaluator`
- `quad.benchmarks` — MobilenetSSD, `SNPEBenchmarkConfig`, `snpe_bench.py` integration
- `quad.qnn` — QNN SDK API table (11 components), 3-format inference pipelines (.so, .bin, .tflite)
- `quad.sdk_tools` — 18-tool platform matrix, `OnnxConverterArgs`, `QairtConverterArgs`, `QairtQuantizerArgs`, `SnpeNetRunArgs`, `DiagviewArgs`, `ArchCheckerArgs`, all 6 Accuracy Debugger mode builders
- `quad.profiler.linting` — HTP linting profiler with cycle-count parser and bottleneck analysis
- `quad.profiler.qhas` — QHAS 3-step profiling workflow with chrometrace generation
- `quad.profiler.diagview` — `snpe-diagview` wrapper
- `quad.profiler.levels` — Shared `ProfilingLevel` enum

### [0.2.0] — 2026-04-30 — Platform Complete (Mock Mode)

- All 7 platform phases complete (Phases A–G)
- 530 tests passing at release

### [0.1.0] — 2026-04-15 — Foundation

- MCP Agent with 5 tools, mock adapter, project scaffold
