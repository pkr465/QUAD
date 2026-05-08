# QUAD — Qualcomm Unified Agent for Developers

> Abstracting Qualcomm SDKs, Profilers & Hardware for Accelerated AI Development

[![Status](https://img.shields.io/badge/status-v0.4.0%20Gap--Closure-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![Tests](https://img.shields.io/badge/tests-2002%20passing-brightgreen)]()
[![MCP](https://img.shields.io/badge/protocol-MCP-purple)]()
[![Skills](https://img.shields.io/badge/Claude%20Code%20skills-10-blueviolet)]()
[![License](https://img.shields.io/badge/license-Qualcomm%20Confidential-red)]()

---

## What is QUAD?

**QUAD** (**Q**ualcomm **U**nified **A**gent for **D**evelopers) is a Claude Code-powered MCP (Model Context Protocol) server that gives AI developers a single conversational interface to Qualcomm's SDK ecosystem. Instead of learning QNN, SNPE, Hexagon SDK, Adreno SDK, AIMET, and multiple profilers independently, developers express intent in natural language and receive production-ready inference code, profiling reports, and optimization recommendations automatically.

**Target**: Time-to-First-Inference < 10 minutes from cold start.
**Vision**: A full 7-layer unified AI computing platform for Qualcomm SoCs.

---

## Supported Platforms

| Platform | Chipset | NPU | Primary SDK | Mock Mode | Real Hardware |
|----------|---------|-----|-------------|-----------|---------------|
| AI PC (Windows) | Snapdragon X Elite (X1E-80-100) | 45 TOPS | QNN / QAIRT 2.x | ✅ Ready | ✅ CPU validated on Dell Latitude 7455 — see [`docs/SAMPLE_APP_REPORT.md`](docs/SAMPLE_APP_REPORT.md). NPU pending QAIRT SDK install. |
| Arduino UNO Q (Linux) | QCS2210 (Robotics RB1) | ~1 TOPS DSP | SNPE 2.x | ✅ Ready | Pending physical device |
| Mobile (Android) | Snapdragon 8 Elite (SM8750) | 48 TOPS | SNPE / QNN 2.x | ✅ Ready | Pending physical device |

> **Mock Mode** (default): All tools return realistic simulated data — no hardware or SDK required.
> **Real Hardware Mode**: Auto-enabled when QAIRT SDK is installed. See [`docs/REAL_HARDWARE.md`](docs/REAL_HARDWARE.md) for the one-command setup.

---

## Quick Start — One Command

### Pick your deployment topology first

QUAD has a separate **server** (heavy: SDK + adapters + compiler) and
**client** (light: just Claude Code provisioning + 5 MB of deps). Pick
the topology that matches you:

| Topology | Server install | Client install |
|---|---|---|
| **Local dev** (one machine, default) | `./install.sh` (does both) | _(included in install.sh)_ |
| **Lightweight laptop + lab machine over SSH** | `./install.sh --server-only` on lab | `./install-client.sh --transport=stdio-ssh ...` on laptop |
| **IDE laptop + hosted MCP service** | _(operator-managed)_ | `./install-client.sh --transport=sse-http ...` |
| **CI runner** | `./install.sh --server-only --mock-only` | _(not needed)_ |

Detailed walkthroughs:
- [`docs/CLIENT_INSTALL.md`](docs/CLIENT_INSTALL.md) — IDE-machine setup (any transport)
- [`docs/SERVER_INSTALL.md`](docs/SERVER_INSTALL.md) — server-machine setup
- [`docs/CLIENT_SERVER_ARCHITECTURE.md`](docs/CLIENT_SERVER_ARCHITECTURE.md) — internal architecture

### Prerequisites (one-time, per machine)

**Recommended hardware: a Snapdragon X-series Copilot+ PC** —
Snapdragon X / X Elite / X2 Elite. These ship with the Hexagon NPU
that real-mode targets (45+ TOPS). Tested examples: Dell Latitude 7455,
Lenovo ThinkPad T14s Gen 6, Microsoft Surface Pro 11 / Laptop 7,
HP OmniBook X, Samsung Galaxy Book4 Edge. Other x86_64 Windows
machines, macOS, and Linux all work for mock-mode development.

| OS | What you need before `git clone` |
|---|---|
| **Windows (default)** | Nothing — `bootstrap.ps1` installs Git for Windows. Python 3.10+ from the Microsoft Store (or `winget install Python.Python.3.12`). PowerShell 7 recommended (`winget install Microsoft.PowerShell`); 5.1 also works. |
| **macOS** | Python 3.10+ (`brew install python@3.12`). bash 4+ (`brew install bash` if on default 3.2). |
| **Linux** | Python 3.10+ (`apt install python3.10` / `dnf install python3.12`). bash already present. |

For **real-hardware mode**, also download the QAIRT SDK once:
1. Visit <https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk>
2. Sign in with your Qualcomm developer account, accept the EULA
3. Download the latest `.zip` (~1 GB) and save it anywhere — you'll point QUAD at it during install

> **Why doesn't the installer download QAIRT for me?** The Qualcomm developer
> portal gates downloads behind a developer-account login + per-version EULA
> acceptance. There is no anonymous direct-download URL, and the installer
> never bypasses that. For unattended automation (CI), set
> `QAIRT_DOWNLOAD_URL` and `QAIRT_DOWNLOAD_TOKEN` to a pre-authorised mirror
> + bearer token; the installer will use those.

### Fastest path — fresh machine to running QUAD

#### Windows (default — Snapdragon X / X Elite / X2 Elite Copilot+ PC)

In **PowerShell**, from anywhere you want to clone the repo:

```powershell
git clone https://github.com/pkr465/QUAD.git
cd QUAD
.\bootstrap.ps1 -QairtArchive C:\Users\<you>\Downloads\qairt-2.45.0.260326.zip
```

`bootstrap.ps1` installs Git Bash if needed, then runs the bash installer.
After it finishes, **open a new Git Bash window** (recommended — `source ./activate.sh`
is a bash command) and:

```bash
cd /c/path/to/QUAD          # use the path you cloned to
source ./activate.sh
quad mode                    # → 'real-mode: READY'
```

If you prefer to stay in PowerShell, you can activate the Python venv directly
(`.venv\Scripts\Activate.ps1`) and call `quad mode` — the SDK was already
discovered and recorded in `.quad/sdk.json` by the bootstrap, so it'll be
picked up automatically.

#### macOS / Linux / WSL

```bash
git clone https://github.com/pkr465/QUAD.git && cd QUAD
./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip
source ./activate.sh && quad mode    # → 'real-mode: READY'
```

That's it. Either way, then start the MCP server (see [Run](#run-the-mcp-server-or-a-sample-app)) or open Claude Code (it auto-detects).

### Other ways to install

Pick the row that matches your situation. All commands assume you've already
run `git clone` and `cd QUAD`.

| Situation | Command |
|---|---|
| **Windows (default)** — you downloaded QAIRT from the developer portal | `.\bootstrap.ps1 -QairtArchive C:\Users\<you>\Downloads\qairt-2.45.0.260326.zip` |
| Windows — archive already in your Downloads folder | `.\bootstrap.ps1` (auto-detected) |
| Windows — no SDK available right now (mock-mode dev) | `.\bootstrap.ps1 -MockOnly` |
| Windows — prefer cmd.exe over PowerShell | `bootstrap.bat --qairt-archive "C:\Users\<you>\Downloads\qairt.zip"` |
| Windows — full reinstall | `.\bootstrap.ps1 -Clean` (or `bootstrap.bat --clean`) |
| macOS / Linux / WSL — you downloaded QAIRT | `./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip` |
| macOS / Linux / WSL — archive already in `~/Downloads/` | `./install.sh` (auto-detected) |
| macOS / Linux / WSL — `QAIRT_SDK_ROOT` already set | `./install.sh` |
| macOS / Linux / WSL — no SDK available right now | `./install.sh --mock-only` |
| Any OS — CI mirror with a token | `QAIRT_DOWNLOAD_URL=… QAIRT_DOWNLOAD_TOKEN=… ./install.sh` |

What every install path does, in order:

1. Creates `.venv/` and installs QUAD with dev extras (`pip install -e .[dev]`)
2. Acquires the SDK using a six-strategy chain (archive flag → existing
   env var → `quad sdk discover` of vendor defaults / `./sdks/` →
   `~/Downloads` auto-detect → `QAIRT_DOWNLOAD_URL` + `QAIRT_DOWNLOAD_TOKEN`
   → graceful mock-mode fallback)
3. Unpacks the SDK into the gitignored `./sdks/<flavor>-<version>/`
4. Sets `QAIRT_SDK_ROOT` / `QNN_SDK_ROOT` / `SNPE_ROOT` for the current shell
5. Generates `quad.toml`, `.claude/settings.json` (MCP auto-detect), and `activate.sh`
6. Runs `pytest -q` for verification
7. Always succeeds — falls back to mock mode with clear next-step guidance if no SDK is found

On Windows, `bootstrap.ps1` first ensures bash is available (installing
Git for Windows via `winget` if needed), then hands off to `install.sh`.
It's idempotent — re-runs are safe and will skip the Git install when bash
is already present. `bootstrap.bat` is a cmd.exe shim that prefers `pwsh.exe`
(PowerShell 7+) over `powershell.exe` 5.1 and sets the bypass execution policy.

### Verify

```bash
quad mode                   # → adapter mode + real-mode readiness
quad sdk status             # → which SDK was discovered, where, version
quad doctor                 # → 14 environment checks
quad doctor --real-mode     # → strict pre-flight; exits non-zero on any SDK issue
```

Expected output for a fully-real-mode install:

```
adapter_mode:    real
sdk:             qairt 2.45.0.260326  (project:./sdks)
real-mode:       READY
  reason:        Real mode active. SDK root: ./sdks/qairt-2.45.0.260326
```

### Add the SDK later (optional)

If you installed in mock mode first and want to enable real hardware afterwards:

```bash
quad sdk install ~/Downloads/qairt-2.45.0.260326.zip
export QUAD_ADAPTER_MODE=real           # bash / Git Bash
# or in PowerShell: $env:QUAD_ADAPTER_MODE = "real"
quad mode                                # should now report 'real-mode: READY'
```

### Run the MCP server or a sample app

#### A. MCP server for Claude Code (the typical path)

```bash
./launch.sh                              # macOS / Linux / Git Bash on Windows
./launch.sh --real --verbose             # explicit real mode + debug logs
./launch.sh --sse                        # SSE transport instead of stdio (for IDE plugins)
```

In **PowerShell** (no bash), use the Python module entry point directly:

```powershell
python -m quad.server                    # equivalent to ./launch.sh
$env:QUAD_ADAPTER_MODE = "real"; python -m quad.server   # real mode
```

`.claude/settings.json` is generated at install time so Claude Code
auto-detects the MCP server with all 5 tools pre-approved. Open Claude Code
and ask things like:

> "Detect the hardware on this machine."
> "Convert mobilenetv2.onnx to QNN INT8 and profile it on the NPU."
> "Generate Windows C++ inference code for this model."

#### B. Interactive zero-to-inference wizard

```bash
quad quickstart
```

#### C. Run a benchmark

```bash
quad benchmark                           # default model set (MobileNetV2 / ResNet50 / YOLOv8n)
quad benchmark --models mobilenetv2      # specific model
quad benchmark --device npu              # specific runtime
```

#### D. End-to-end sample app with real CPU/NPU measurements

```bash
# Install the small extra deps for the sample
pip install onnxruntime-qnn psutil

# Download a standard model (one-time)
python -c "import urllib.request, os; os.makedirs('examples/models', exist_ok=True); \
  urllib.request.urlretrieve('https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx', \
  'examples/models/mobilenetv2-12.onnx')"

# Run the full QUAD pipeline + real CPU benchmark on Oryon
PYTHONUTF8=1 python examples/sample_app_real_hw.py --iterations 500 --warmup 50 \
  --json-out examples/run_results.json
```

Produces a structured report with real measurements. See
[`docs/SAMPLE_APP_REPORT.md`](docs/SAMPLE_APP_REPORT.md) for the format.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `quad mode` says `NOT READY` | Run `quad doctor --real-mode` — it tells you exactly what's missing. |
| `bootstrap.ps1` won't run (execution policy) | Use `bootstrap.bat` instead, or `powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1`. |
| `bash: command not found` on Windows | Run `bootstrap.ps1` first — it installs Git Bash via winget. |
| `./launch.sh` not found in PowerShell | Use `python -m quad.server` instead, or run launch.sh from Git Bash. |
| 8 tests fail on Windows | Pre-existing path-assertion bugs in modules unrelated to the install. No functional impact. |
| SDK installed but not detected | `quad sdk discover` shows what was scanned. The SDK directory must contain `bin/<arch>/qairt-converter` (or `snpe-net-run`). |
| Want CI / unattended install | Set `QAIRT_DOWNLOAD_URL` + `QAIRT_DOWNLOAD_TOKEN` to a pre-authorised mirror; the installer fetches and unpacks automatically. |

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
quad doctor                     # 16 environment checks (Python / SDK / tools / DSP / AIMET / AI Hub)
quad doctor --real-mode         # Strict pre-flight; fails on any SDK issue

# Workflow
quad quickstart                 # Interactive zero-to-inference wizard
quad benchmark                  # Standard benchmark suite (MobileNetV2 / ResNet / YOLOv8n)
quad detect                     # Real OS-level probe of CPU/GPU/NPU + RAM + OS
quad detect --refresh           # Bypass the discovery cache and re-probe
quad compile <model>            # Compile ONNX / PyTorch → .qbin (frontend real, backend honest stub)
quad profile <model>            # Run platform profiler
quad serve <model>              # Start FastAPI inference server (POST /infer, GET /health, /metrics)
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

## Claude Code Skills

When you use the QUAD MCP server through Claude Code, ten skill files
in [`.claude/skills/`](.claude/skills/) route phrases like "what
hardware do I have?" or "find bottlenecks" to the right MCP tool flow,
with full handling instructions, edge cases, and follow-up suggestions.

| Skill | Triggered by | What it does |
|---|---|---|
| `quad-quickstart` | "get started", "set up QUAD" | End-to-end walkthrough: detect → convert → profile → orchestrate → codegen (8 steps) |
| `quad-detect` | "what hardware", "is this Snapdragon", "do I have NPU" | Real probe + Qualcomm-vs-other tip routing |
| `quad-convert` | "convert this model", "quantize", "compile ONNX" | Conversion with calibration data + image format guidance |
| `quad-profile` | "profile", "measure latency", "find bottlenecks", "QHAS" | Picks the right level (basic/detailed/linting/qhas) + bottleneck callouts |
| `quad-orchestrate` | "allocate across CPU/GPU/NPU", "compare power modes" | 3-mode comparison + fallback analysis |
| `quad-codegen` | "generate inference code", "C++ for this model", "Android JNI" | Platform/language/sdk picker + build commands |
| `quad-doctor` | "is QUAD set up", "diagnose", "why isn't real mode working" | Diagnostic translation table — every check has an exact fix command |
| `quad-deploy` | "deploy to my phone", "push to Arduino" | `deploy.sh` + remote profiling walkthrough |
| `quad-recommend` | "what's the best way", "should I use INT4 or INT8", "NPU or GPU" | Synthesises model + target + use case into a categorised plan |
| `quad-serve` | "start an inference server", "expose as HTTP API" | FastAPI server setup + curl + Python client snippets |

Every MCP tool response now also includes:
- **`payload["ui"]`** — markdown summary via the matching formatter
- **`payload["tips"]`** — 2 contextual tips from a 25-entry catalogue
- **`payload["suggestions"]`** — for `profile_workload`, per-bottleneck Suggestion list

So even without the skill files, Claude Code chats with QUAD render
rich tables, utilisation bars, and contextual recommendations
inline rather than raw JSON dumps.

---

## MCP Tools

| Tool | Description | Input | Output |
|------|-------------|-------|--------|
| `hardware_detect` | Detect chipset, CPU/GPU/NPU topology, memory | platform | Device profile JSON |
| `convert_model` | Convert ONNX/PyTorch/TF/TFLite → QNN/SNPE format | model_path, source_format, target_sdk, quantization, layout | Output path (`.bin` for QNN target, `.dlc` for SNPE) + conversion notes + image-format guidance |
| `profile_workload` | Run profiler (basic/detailed/linting/qhas) | model_path, runtime, profiling_level | Latency/power/memory report + per-op cycle data |
| `orchestrate_workload` | Allocate ops across CPU/GPU/NPU | model, profile, power_mode | Layer→runtime allocation map |
| `generate_code` | Emit platform-specific inference code | platform, language, model | Source files + build instructions |

The MCP server runs `sdk_manager.startup_resolve_and_log()` on every start: it locates an installed SDK from env vars, `quad.toml`, the project's `./sdks/` directory, `~/.quad/sdks/`, or vendor-default paths (`C:\Qualcomm\AIStack\QAIRT\*`, `/opt/qcom/aistack/qairt/*`). Found SDKs auto-populate `QAIRT_SDK_ROOT` for child processes.

---

## Real Hardware Mode — Validated on Snapdragon X Elite

The full pipeline has been run end-to-end on a **Dell Latitude 7455 (Snapdragon X Elite X1E80100, Windows 11 Pro)**:

| Metric | Result | Source |
|---|---|---|
| CPU mean latency (MobileNetV2-1.0, 500 iters) | **2.56 ms** | ONNX Runtime CPU EP on 12 Oryon cores |
| Throughput | **388 FPS** | Same |
| Latency stddev | **0.95 ms** | Same |
| Memory delta during run | 3.8 MB (125.8 → 129.6 MB peak) | psutil RSS |
| CPU utilisation | 1160% (≈11.6 of 12 logical cores) | psutil |
| Hardware detected | Adreno X1-85 GPU · Hexagon NPU (`ComputeAccelerator`, `Status=OK`) · 31.6 GB RAM | Win32 PnP / Get-CimInstance |
| QUAD pipeline | All 5 MCP tools ran cleanly; valid Windows QNN C++ code generated | `examples/sample_app_real_hw.py` |

NPU benchmarks are pending — the Hexagon NPU is detected and ready, but
direct measurement requires the QAIRT SDK install. Full methodology, raw
data, and reproduction steps: [**`docs/SAMPLE_APP_REPORT.md`**](docs/SAMPLE_APP_REPORT.md).
Sample app source: [**`examples/sample_app_real_hw.py`**](examples/sample_app_real_hw.py).
The reproduction commands are in [Run → D](#d-end-to-end-sample-app-with-real-cpunpu-measurements) above.

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
│   ├── tools/                    # MCP tool handlers (each enriches with ui+tips+suggestions)
│   ├── adapters/
│   │   ├── factory.py            # Mock ↔ Real adapter factory (strict mode, fallback tagging)
│   │   ├── mock_adapter.py       # Deterministic simulated responses
│   │   ├── qairt_adapter.py      # Real adapter (qairt-converter, snpe-net-run, ...)
│   │   ├── aimet_adapter.py      # ⭐ AIMET PTQ for INT8/INT4 quantization (mock + real-stub)
│   │   ├── aihub_adapter.py      # ⭐ Qualcomm AI Hub cloud profiling + remote compilation
│   │   └── model_inputs.py       # ⭐ Model introspection + shape-aware input generation
│   ├── sdk_manager.py            # ⭐ SDK auto-discovery + zip/tgz install + state tracking
│   ├── ui/                       # ⭐ Markdown formatters (rich MCP tool responses)
│   ├── suggestions.py            # ⭐ Recommendation engine (quantization/runtime/power/optim)
│   ├── tips.py                   # ⭐ 25-entry contextual tips catalogue
│   ├── runtime/
│   │   ├── device.py             # Device list with real local-host probe
│   │   ├── host_probe.py         # ⭐ Per-OS hardware probe (Win32/procfs/sysctl/adb)
│   │   └── ...                   # Tensor, Model, Stream, Memory, Power
│   ├── compiler/
│   │   ├── pipeline.py           # Honest backend stub + op-coverage in metadata
│   │   ├── frontend_onnx.py      # Real ONNX → IR (uses onnx Python module)
│   │   ├── op_coverage.py        # ⭐ Op-coverage report per target backend
│   │   └── ...                   # QUAD IR, QBin, model_conversion
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
│   ├── serve/
│   │   ├── server.py             # ModelServer (in-process inference engine)
│   │   └── http.py               # ⭐ FastAPI binding (POST /infer, /health, /metrics)
│   └── utils/                    # SNPE logging, perf profiles, layer support
├── .claude/skills/               # ⭐ 10 Claude Code skill files (quad-quickstart, etc.)
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
├── tests/                        # 2002 passing / 3 skipped / 0 failed
│   ├── unit/test_sdk_manager/    # 29 tests for discovery + install + startup
│   ├── unit/test_adapters/       # Factory, mock, real, dlc-compat, dsp-env, AIMET, AI Hub, model_inputs
│   ├── unit/test_ui/             # ⭐ Formatters (23) + suggestions (20) + tips (13) — 56 total
│   ├── unit/test_compiler/       # IR + frontend + op-coverage (168 incl. 22 new for op_coverage)
│   ├── unit/test_serve/          # FastAPI HTTP + ModelServer
│   └── ...                       # 100+ test files across all modules
├── examples/
│   ├── sample_app.py             # Mock-mode workflow walkthrough
│   ├── sample_app_real_hw.py     # ⭐ Real-hardware sample with ground-truth CPU benchmark
│   ├── models/                   # Downloaded ONNX models (gitignored)
│   ├── generated/                # QUAD-generated inference code (gitignored)
│   └── run_results.json          # Last sample-app run data
├── docs/
│   ├── REAL_HARDWARE.md                     # ⭐ One-step real-mode enablement playbook
│   ├── SAMPLE_APP_REPORT.md                 # ⭐ Real-hardware run report (Snapdragon X Elite)
│   ├── GAP_ANALYSIS.md                      # ⭐ End-to-end gap inventory (4 tiers, 17 items)
│   ├── IMPLEMENTATION_PLAN.md               # ⭐ Phased gap-closure execution plan
│   ├── IMPLEMENTATION_PROGRESS.md           # ⭐ What landed in the gap-closure session
│   ├── BACKEND_COMPLETION_DEPENDENCIES.md   # ⭐ T1.1 dependency map (3 implementation paths)
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

> **Version**: 0.4.0
> **Tests**: 2002 passing / 3 skipped / 0 failed (was 1811 / 8 failed before the 2026-05-08 gap-closure session)
> **Source files**: 120+ Python modules
> **Last Updated**: 2026-05-08

### What Works Now

- Full mock-mode platform — all 5 MCP tools work without hardware or SDKs
- **One-step real-hardware install** via `./install.sh --qairt-archive <path>` (six-strategy SDK acquisition with graceful mock-mode fallback)
- **MCP server SDK auto-discovery** — env var → quad.toml → `./sdks/` → `~/.quad/sdks` → vendor defaults
- **`quad mode` / `quad sdk` / `quad doctor --real-mode`** — full visibility into adapter state and SDK readiness
- **Real Snapdragon X Elite validation** — 388 FPS / 2.56 ms MobileNetV2 on Oryon CPU, full QUAD pipeline runs end-to-end (`docs/SAMPLE_APP_REPORT.md`)
- **`quad detect`** does real local hardware probing (PowerShell on Windows, `/proc` on Linux, sysctl on macOS, ADB on Android)
- **Model input introspection** — `convert_model` reads real input shape/dtype from DLC/ONNX (was hardcoded `(1,3,224,224) float32`)
- **`quad serve`** spins up a real FastAPI server with `/infer`, `/health`, `/metrics`, `/models` endpoints
- **AIMET adapter** (mock + real-stub) for INT8 / INT4 quantization with calibration data sources
- **AI Hub adapter** (mock + real `qai_hub` integration) for cloud profiling and remote compilation
- **Generated Windows C++** has real QNN init / load / execute / cleanup (no more TODO scaffolds)
- **Honest compiler stub** — no more `b"QUAD_COMPILED_BINARY"` placeholder; emits `BackendNotImplementedError` plus a per-target op-coverage report
- **Strengthened codegen validator** — TODO/FIXME detection, optional `gcc -fsyntax-only` invocation, string-literal-aware brace counting
- **10 Claude Code skill files** route user phrases to the right MCP tool flow with rich markdown formatters, contextual tips, and concrete recommendations
- **Windows CI matrix** — all 8 pre-existing path-assertion failures fixed
- All SNPE/QAIRT SDK documentation sections integrated as structured Python modules
- Complete CLI builders for all 18 SDK tools across 6 platforms (except `qnn-context-binary-generator` — see `docs/BACKEND_COMPLETION_DEPENDENCIES.md`)
- C++ code generation templates for .so, .bin, and TFLite delegate pipelines
- VS Code extension scaffold with debug configurations and tasks

### What Remains

See [`docs/IMPLEMENTATION_PROGRESS.md`](docs/IMPLEMENTATION_PROGRESS.md)
for the per-gap scorecard. Headline pending items:

- **T1.1 — Real compiler backend** — full IR → SDK call → real binary. Three implementation paths catalogued in [`docs/BACKEND_COMPLETION_DEPENDENCIES.md`](docs/BACKEND_COMPLETION_DEPENDENCIES.md); recommend Path A + C in 1 week, Path B as follow-up sprint.
- **T1.3 — Runtime ctypes/cffi to QNN C API** — `Device.infer` etc. still numpy mocks; full real binding is 1-2 weeks
- **AIMET real backend** — `_quantize_aimet_torch` / `_quantize_aimet_onnx` raise `NotImplementedError` (full PyTorch model integration is ~2 weeks)
- **T2.3 — Orchestration → codegen wiring** — allocation map ignored by templates today
- **T2.5 — Phase 2/3 platform adapters wired** — `LinuxPlatform`/`AndroidPlatform` exist but aren't routed through the factory
- **T2.7 — Package name decision** — `quad-agent` vs `qualcomm-ai-toolkit` before PyPI publication
- Physical device testing on Arduino UNO Q (QCS2210) and Snapdragon 8 Elite (SM8750)

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
| **Server Install** | [`docs/SERVER_INSTALL.md`](docs/SERVER_INSTALL.md) | Full QUAD MCP server + SDK + adapters install |
| **Client Install** | [`docs/CLIENT_INSTALL.md`](docs/CLIENT_INSTALL.md) | Lightweight Claude Code provisioning (3 transport options) |
| Client/Server Architecture | [`docs/CLIENT_SERVER_ARCHITECTURE.md`](docs/CLIENT_SERVER_ARCHITECTURE.md) | Layering, separation of concerns, residual gaps |
| Real-Hardware Enablement | [`docs/REAL_HARDWARE.md`](docs/REAL_HARDWARE.md) | One-step setup + 6-strategy SDK acquisition + per-platform notes |
| Sample-App Run Report | [`docs/SAMPLE_APP_REPORT.md`](docs/SAMPLE_APP_REPORT.md) | Real Snapdragon X Elite measurements, methodology, repro |
| **Gap Analysis** | [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md) | End-to-end gap inventory with 4-tier severity scorecard |
| **Implementation Plan** | [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) | Phased gap-closure roadmap (A through G) |
| **Implementation Progress** | [`docs/IMPLEMENTATION_PROGRESS.md`](docs/IMPLEMENTATION_PROGRESS.md) | What landed in the gap-closure session + closure scorecard |
| **Backend Dependencies** | [`docs/BACKEND_COMPLETION_DEPENDENCIES.md`](docs/BACKEND_COMPLETION_DEPENDENCIES.md) | Full T1.1 dependency map across 5 categories + 3 implementation paths |
| Prerequisites | [`docs/PREREQUISITES.md`](docs/PREREQUISITES.md) | SDK, hardware, account requirements |
| Implementation Guide | [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md) | Sprint-by-sprint build plan |
| PRD | `docs/PRD_Qualcomm_DevWorkflows_v3.docx` | Product requirements |
| Design Spec | `docs/Design_Document_QUAD_Agent.docx` | Engineering specification |
| Claude Code Context | [`CLAUDE.md`](CLAUDE.md) | AI assistant context & project status |
| Pending Tasks | [`TODO.md`](TODO.md) | All pending items by priority |

---

## Changelog

### [0.4.0] — 2026-05-08 — Gap Closure (current)

The overnight gap-closure session executed Phases A–G from
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md): closed
11 of 17 Tier-1/Tier-2 gaps catalogued in
[`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md), grew the test suite
from 1811 to 2002 passing, and added a full UX layer.

#### Tier-1 / Tier-2 closures
- **T1.7** — Templates bundled in wheel via hatch `force-include`; smart resolver chain
- **T1.8** — Real Windows QNN C++ template (init / load / execute / cleanup); strengthened validator with TODO/FIXME detection + optional `gcc -fsyntax-only`
- **T1.4** — Real `execute_inference` I/O marshalling (no more 500-char stdout truncation)
- **T1.5** — AIMET adapter (mock + real-stub) with INT4 per-channel scheme + calibration data sources
- **T1.6** — AI Hub adapter (mock + real `qai_hub`) for cloud profiling and remote compile
- **T1.2** — FastAPI inference server: `quad serve` is now a real HTTP service
- **T1.1** (partial) — Honest backend stub: no more `b"QUAD_COMPILED_BINARY"` placeholder; emits `BackendNotImplementedError` plus per-target op-coverage report
- **T2.4** — `orchestrate_workload` empty-layers handling (auto-reprofiles in detailed mode)
- **T2.6** — Windows added to CI matrix; all 8 pre-existing path-assertion failures fixed
- **T2.8** — Real model-input introspection (`model_inputs.py`) replaces hardcoded `np.random.randn(1,3,224,224)`
- **T3.6** — `quad detect` does real PowerShell / procfs / sysctl / ADB probes

#### New UX layer (Phase F)
- **`src/quad/ui/`** — 8 markdown formatters (table, utilization bar, device, profile, conversion, allocation, doctor, coverage, sdk_status)
- **`src/quad/suggestions.py`** — 5 recommendation generators (quantization, runtime, power_mode, optimisations, suggest_for_workflow)
- **`src/quad/tips.py`** — 25-entry contextual tips catalogue across 7 contexts
- **`.claude/skills/`** — 10 Claude Code skill files routing user phrases to MCP tool flows
- All 5 MCP tools now enrich responses with `payload["ui"]`, `payload["tips"]`, `payload["suggestions"]`

#### New documentation
- [`docs/GAP_ANALYSIS.md`](docs/GAP_ANALYSIS.md) — 4-tier gap inventory with scorecard
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — phased execution plan
- [`docs/IMPLEMENTATION_PROGRESS.md`](docs/IMPLEMENTATION_PROGRESS.md) — session deliverable + closure scorecard
- [`docs/BACKEND_COMPLETION_DEPENDENCIES.md`](docs/BACKEND_COMPLETION_DEPENDENCIES.md) — T1.1 dependency map across 5 categories

#### Test growth
- 191 new tests across 11 phase commits
- Final: 2002 passing / 3 skipped / 0 failed (was 1811 / 8 failed)

### [0.3.0] — 2026-05-07 — Real-Hardware Enablement

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
