# Prerequisites — Claude Code AI Agent for Qualcomm Developer Workflows

This document defines all prerequisites required to build and deploy the MCP-based Claude Code agent described in `PRD_Qualcomm_DevWorkflows_v3.docx`. Prerequisites are organized by phase and category. Items marked with their phase indicate when they become required; items under "All Phases" are needed from day one.

---

## Table of Contents

1. [Development Environment (All Phases)](#1-development-environment-all-phases)
2. [Phase 1 — AI PC / Windows (Snapdragon X Elite)](#2-phase-1--ai-pc--windows-snapdragon-x-elite)
3. [Phase 2 — Arduino UNO Q / Linux (QCS2210)](#3-phase-2--arduino-uno-q--linux-qcs2210)
4. [Phase 3 — Mobile / Android (Snapdragon 8 Elite)](#4-phase-3--mobile--android-snapdragon-8-elite)
5. [Phase 4 — Cross-Platform & Optimization](#5-phase-4--cross-platform--optimization)
6. [Testing & Validation (All Phases)](#6-testing--validation-all-phases)
7. [Accounts & Access](#7-accounts--access)
8. [Sample Models & Data](#8-sample-models--data)
9. [Prerequisite Verification Checklist](#9-prerequisite-verification-checklist)

---

## 1. Development Environment (All Phases)

### 1.1 Python Runtime & Package Management

| Requirement | Details |
|-------------|---------|
| Python | 3.10+ (3.11 recommended for performance) |
| Package Manager | `pip` + `venv` or `conda` |
| Virtual Environment | Isolated venv per project (`python -m venv .venv`) |
| pip-tools | For deterministic dependency locking (`pip-compile`) |

### 1.2 MCP Server Framework

| Requirement | Details |
|-------------|---------|
| FastMCP | Python MCP server framework (latest stable) |
| MCP SDK | `mcp` Python package for protocol compliance |
| JSON Schema | For MCP tool input/output validation |
| asyncio | Python async runtime (stdlib, but design around it) |

```bash
pip install fastmcp mcp pydantic jsonschema
```

### 1.3 Development Tools

| Tool | Purpose | Install |
|------|---------|---------|
| Claude Code CLI | Agent runtime & testing | `npm install -g @anthropic-ai/claude-code` |
| VS Code | Primary IDE | With Claude Code extension |
| Git | Version control | 2.40+ |
| Docker | Containerized SDK environments | Docker Desktop or Engine 24+ |
| Make | Build automation | GNU Make 4+ |

### 1.4 Python Development Dependencies

```bash
pip install \
    pydantic>=2.0 \
    httpx \
    aiofiles \
    structlog \
    rich \
    typer \
    python-dotenv
```

### 1.5 Project Structure (Expected)

```
QUAD/
├── docs/                    # PRD, prerequisites, architecture docs
├── src/
│   ├── server/              # FastMCP server entry point
│   ├── tools/               # MCP tool implementations
│   │   ├── hardware_detect.py
│   │   ├── profile_workload.py
│   │   ├── convert_model.py
│   │   ├── orchestrate_workload.py
│   │   └── generate_code.py
│   ├── adapters/            # Qualcomm SDK abstraction adapters
│   │   ├── qnn_adapter.py
│   │   ├── snpe_adapter.py
│   │   ├── hexagon_adapter.py
│   │   ├── adreno_adapter.py
│   │   └── aimet_adapter.py
│   ├── platforms/           # Platform-specific logic
│   │   ├── windows.py
│   │   ├── linux.py
│   │   └── android.py
│   └── utils/               # Shared utilities
├── tests/                   # pytest test suites
├── templates/               # Code generation templates
├── configs/                 # SDK configuration profiles
├── pyproject.toml           # Project metadata & dependencies
└── CLAUDE.md
```

---

## 2. Phase 1 — AI PC / Windows (Snapdragon X Elite)

### 2.1 Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Device | Snapdragon X Elite (X1E-80-100) laptop/desktop |
| CPU | 12-core Oryon ARM64 @ 3.8 GHz |
| GPU | Adreno X1-85 (4.6 TFLOPS) |
| NPU | Hexagon NPU — 45 TOPS |
| RAM | 16 GB minimum (32 GB recommended) |
| Storage | 50 GB free (SDKs + models) |
| OS | Windows 11 on Snapdragon (WoS) |

**Alternative**: Windows on ARM VM with Qualcomm SDK emulation (limited NPU testing).

### 2.2 Qualcomm SDKs

#### QNN SDK 2.x (Primary)

| Item | Details |
|------|---------|
| Package | Qualcomm AI Engine Direct SDK (QNN) |
| Version | 2.x GA (latest stable) |
| Download | Qualcomm Developer Network (QDN) |
| Install Path | `C:\Qualcomm\AIStack\QNN\` |
| Key Env Vars | `QNN_SDK_ROOT`, `QNN_TARGET_ARCH=aarch64-windows` |
| Python Bindings | `qnn-python` wheel included in SDK |
| Key Binaries | `qnn-model-lib-generator`, `qnn-net-run`, `qnn-context-binary-generator` |

#### QAIRT (Qualcomm AI Runtime)

| Item | Details |
|------|---------|
| Package | Included with QNN SDK 2.x |
| Purpose | Unified inference runtime across QNN backends |
| Key API | `QnnInterface_t`, `QnnContext`, `QnnGraph` |

#### Hexagon SDK 5.x

| Item | Details |
|------|---------|
| Package | Hexagon SDK |
| Version | 5.x GA |
| Purpose | DSP/HTP programming, custom op development |
| Install Path | `C:\Qualcomm\Hexagon_SDK\5.x\` |
| Key Env Vars | `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT` |
| Key Tools | `hexagon-clang`, `hexagon-sim`, `hexagon-lldb` |

#### Adreno SDK

| Item | Details |
|------|---------|
| Package | Adreno GPU SDK |
| Purpose | GPU compute — OpenCL, Vulkan ML workloads |
| Install Path | `C:\Qualcomm\AdrenoSDK\` |
| Key Env Vars | `ADRENO_SDK_ROOT` |
| Dependencies | Vulkan SDK 1.3+, OpenCL headers |

#### AIMET (AI Model Efficiency Toolkit)

| Item | Details |
|------|---------|
| Package | `aimet-torch` or `aimet-tensorflow` |
| Version | Latest stable |
| Purpose | Quantization (PTQ, QAT), compression, pruning |
| Install | `pip install aimet-torch` (requires CUDA for some features) |
| Key APIs | `QuantizationSimModel`, `CrossLayerEqualization`, `Adaround` |

### 2.3 Profiling Tools

#### QPM3 (Qualcomm Power Measurement 3)

| Item | Details |
|------|---------|
| Purpose | Real-time power measurement and analysis |
| Install | QDN download → Windows installer |
| Interface | CLI + GUI; Python API for automation |
| Output | JSON/CSV power traces (mW granularity) |
| Hardware Req | QPM3-compatible measurement jig (or software estimation mode) |

#### Snapdragon Profiler

| Item | Details |
|------|---------|
| Purpose | CPU/GPU/NPU trace capture and visualization |
| Install | QDN download → Windows installer |
| Interface | GUI + CLI (`sdp` command) |
| Output | `.sdp` trace files, exportable to JSON/CSV |
| Connection | USB or Wi-Fi to target device |

#### Qualcomm AI Hub

| Item | Details |
|------|---------|
| Purpose | Cloud-based model profiling and optimization |
| Access | Web portal + REST API + Python SDK |
| Install | `pip install qai-hub` |
| Auth | API key from AI Hub portal |
| Output | Benchmark JSON (latency, throughput, compatibility) |

### 2.4 Model Framework Dependencies

```bash
pip install \
    torch>=2.0 \
    torchvision \
    tensorflow>=2.13 \
    onnx>=1.14 \
    onnxruntime \
    onnxruntime-qnn \
    numpy \
    pillow \
    opencv-python
```

### 2.5 ONNX Runtime with QNN Execution Provider

| Item | Details |
|------|---------|
| Package | `onnxruntime-qnn` |
| Purpose | Run ONNX models directly on QNN backend |
| Config | `providers=['QNNExecutionProvider']` with device options |

---

## 3. Phase 2 — Arduino UNO Q / Linux (QCS2210)

### 3.1 Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Board | Arduino UNO Q (QCS2210 / Qualcomm Robotics RB1) |
| CPU | 4-core Kryo ARM64 |
| GPU | Adreno 504 |
| NPU/DSP | Hexagon DSP V66 (HTP) |
| RAM | 1-2 GB |
| Storage | 8 GB eMMC + microSD expansion |
| Power Budget | < 3W total system |
| Connectivity | USB-C, GPIO, Wi-Fi, Ethernet |

### 3.2 Host Development Machine

| Requirement | Details |
|-------------|---------|
| OS | Ubuntu 22.04 LTS (host for cross-compilation) |
| Cross-compiler | `aarch64-linux-gnu-gcc` toolchain |
| SSH | For remote deployment to Arduino UNO Q |
| Serial | UART/USB-serial for debug console |
| Docker | For reproducible build environments |

### 3.3 Board Software

| Item | Details |
|------|---------|
| Board OS | Ubuntu 22.04 (ARM64) or Yocto Linux BSP |
| BSP Image | Qualcomm-provided Robotics RB1 image |
| SNPE Runtime | Pre-installed or deployed via SSH |
| Python | 3.10+ on-device (for agent scripts) |
| systemd | For inference service management |

### 3.4 Qualcomm SDKs

#### SNPE SDK 2.x

| Item | Details |
|------|---------|
| Package | Snapdragon Neural Processing Engine |
| Version | 2.x GA |
| Download | QDN portal |
| Host Install | Linux x86_64 (conversion tools) |
| Target Install | ARM64 runtime libraries on QCS2210 |
| Key Env Vars | `SNPE_ROOT`, `SNPE_TARGET_ARCH=aarch64-linux` |
| Key Binaries | `snpe-dlc-quantize`, `snpe-net-run`, `snpe-dlc-info`, `snpe-throughput-net-run` |
| Python Tools | `snpe-onnx-to-dlc`, `snpe-tensorflow-to-dlc`, `snpe-dlc-graph-prepare` |

#### Hexagon SDK 5.x (DSP Development)

| Item | Details |
|------|---------|
| Purpose | Custom DSP op development, HTP optimization |
| Host Install | Linux x86_64 |
| Key Tools | `hexagon-clang` cross-compiler, `hexagon-sim` |
| DSP Libraries | `libcdsprpc.so` (deployed to target) |

### 3.5 Profiling Tools

#### SNPE Profiler

| Item | Details |
|------|---------|
| Purpose | Layer-level inference profiling on DSP/CPU/GPU |
| Usage | `snpe-net-run --perf_profile burst` + timing output |
| Output | Per-layer execution time, memory, DSP utilization |

#### Hexagon Profiler

| Item | Details |
|------|---------|
| Purpose | DSP cycle-accurate profiling |
| Tool | `hexagon-prof` (part of Hexagon SDK) |
| Output | Cycle counts, cache behavior, VTCM utilization |

#### Linux perf

| Item | Details |
|------|---------|
| Purpose | System-level CPU profiling on target |
| Install | `apt install linux-tools-generic` on target |

### 3.6 Deployment Tools

```bash
# Host machine packages
sudo apt install \
    aarch64-linux-gnu-gcc \
    aarch64-linux-gnu-g++ \
    cmake \
    ninja-build \
    sshpass \
    picocom    # serial console
```

---

## 4. Phase 3 — Mobile / Android (Snapdragon 8 Elite)

### 4.1 Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Device | Snapdragon 8 Elite (SM8750) phone/dev kit |
| CPU | 8-core Oryon (2x Prime @ 4.32 GHz + 6x Performance) |
| GPU | Adreno 830 (5.0 TFLOPS) |
| NPU | Hexagon NPU (HTP) — 48 TOPS |
| RAM | 12+ GB |
| OS | Android 15+ |
| USB Debugging | Enabled (Developer Options) |
| Bootloader | Unlocked (for custom runtime deployment) |

### 4.2 Android Development Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Android SDK | API 35+ | Platform tools, build tools |
| Android NDK | r26+ | Native C/C++ compilation |
| ADB | Latest | Device communication |
| Android Studio | Latest | IDE + plugin development |
| Gradle | 8.x | Build system for AAR/APK |
| Kotlin | 1.9+ | Android app/plugin code |
| CMake | 3.22+ | NDK build system |

### 4.3 Qualcomm SDKs (Android Target)

#### SNPE SDK 2.x (Android)

| Item | Details |
|------|---------|
| Target | `aarch64-android` |
| Libraries | `libSNPE.so`, pushed via ADB |
| Java/Kotlin API | SNPE Android AAR |
| Key Operations | DLC execution, runtime selection (DSP/GPU/CPU) |

#### QNN SDK 2.x (Android)

| Item | Details |
|------|---------|
| Target | `aarch64-android` |
| Libraries | `libQnn*.so` backend libraries |
| Integration | Via JNI wrapper or SNPE QNN backend |

### 4.4 Profiling Tools (Android)

#### Snapdragon Profiler (Android Mode)

| Item | Details |
|------|---------|
| Connection | USB (ADB) to device |
| Captures | CPU scheduling, GPU rendering, NPU inference traces |
| Export | JSON, CSV, Perfetto format |

#### Perfetto

| Item | Details |
|------|---------|
| Purpose | System-wide tracing (GPU, CPU, memory, power) |
| Install | Built into Android 15+; host tools via `pip install perfetto` |
| Config | Custom trace configs for Adreno GPU + Hexagon NPU |

#### Android GPU Inspector (AGI)

| Item | Details |
|------|---------|
| Purpose | Adreno GPU workload analysis |
| Install | Standalone tool from Android developer site |

### 4.5 AAR/APK Build Dependencies

```groovy
// build.gradle.kts dependencies
implementation("com.qualcomm.qti:snpe-release:2.x.x")
implementation("org.pytorch:pytorch_android:2.x")
implementation("com.google.mediapipe:mediapipe:latest")
```

### 4.6 On-Device Requirements

```bash
# Push SNPE runtime to device
adb push $SNPE_ROOT/lib/aarch64-android/ /data/local/tmp/snpe/
adb shell "export LD_LIBRARY_PATH=/data/local/tmp/snpe/:$LD_LIBRARY_PATH"

# Verify NPU availability
adb shell "cat /sys/class/npu/npu0/status"
```

---

## 5. Phase 4 — Cross-Platform & Optimization

### 5.1 Advanced Quantization

| Requirement | Details |
|-------------|---------|
| AIMET INT4 | `aimet-torch` with INT4 quantization support |
| Mixed Precision | Per-layer INT4/INT8/FP16 selection |
| Calibration Data | Representative dataset (100-1000 samples per model) |
| QNN INT4 Backend | QNN SDK 2.x with INT4 op support (check release notes) |

### 5.2 CI/CD Integration

| Tool | Purpose |
|------|---------|
| GitHub Actions | Automated testing pipeline |
| Qualcomm AI Hub API | CI model benchmarking |
| Docker | Reproducible SDK environments for CI |
| DVC / MLflow | Model version tracking |

### 5.3 Multi-Model Pipeline

| Requirement | Details |
|-------------|---------|
| Pipeline Framework | Custom DAG executor or integration with existing (e.g., MediaPipe) |
| Model Registry | Local or AI Hub-based model catalog |
| Memory Budgeting | Combined memory footprint analysis for chained models |

### 5.4 Dashboard & Reporting

```bash
pip install \
    plotly \
    dash \
    pandas \
    jinja2    # report template rendering
```

---

## 6. Testing & Validation (All Phases)

### 6.1 Test Framework

```bash
pip install \
    pytest>=7.0 \
    pytest-asyncio \
    pytest-cov \
    pytest-mock \
    pytest-timeout \
    hypothesis    # property-based testing
```

### 6.2 MCP Testing

| Approach | Details |
|----------|---------|
| Unit Tests | Mock SDK calls, test tool logic in isolation |
| Integration Tests | Real SDK calls against local/device targets |
| MCP Protocol Tests | Validate JSON-RPC request/response conformance |
| End-to-End Tests | Full workflow from natural language → inference result |

### 6.3 Test Fixtures & Sample Models

| Model | Format | Size | Use Case |
|-------|--------|------|----------|
| MobileNetV2 | ONNX | ~14 MB | Quick validation, classification |
| ResNet-50 | ONNX / PyTorch | ~98 MB | Standard benchmark |
| YOLOv8-nano | ONNX | ~6 MB | Object detection pipeline |
| Whisper-tiny | ONNX | ~150 MB | Audio inference test |
| Stable Diffusion (text-encoder only) | ONNX | ~500 MB | Large model handling |

### 6.4 Validation Criteria per Tool

| Tool | Validation |
|------|-----------|
| `hardware_detect` | Returns valid JSON with all fields populated; chipset matches known device |
| `profile_workload` | Latency > 0, power > 0, memory within device limits |
| `convert_model` | Output DLC/QNN binary exists, file size > 0, `supported_ops_pct` > 0 |
| `orchestrate_workload` | All layers assigned to valid runtime, projected metrics reasonable |
| `generate_code` | Generated code compiles/runs without errors on target platform |

---

## 7. Accounts & Access

| Account | Purpose | Required By |
|---------|---------|-------------|
| Qualcomm Developer Network (QDN) | SDK downloads | Phase 1 |
| Qualcomm AI Hub | Cloud profiling API, model benchmarks | Phase 1 |
| Qualcomm Package Manager (QPM3) | SDK license activation | Phase 1 |
| GitHub (or internal VCS) | Source code, CI/CD | All Phases |
| PyPI (publish) | Package distribution (if open-sourcing) | Phase 4 |
| Docker Hub / Registry | Container image hosting | Phase 2+ |

### 7.1 API Keys & Credentials

| Credential | Storage | Notes |
|------------|---------|-------|
| AI Hub API Key | `.env` file (never committed) | Required for `qai-hub` Python SDK |
| QPM3 License | System keystore | Activated during SDK install |
| ADB Device Key | `~/.android/adbkey` | Auto-generated on first ADB connection |

---

## 8. Sample Models & Data

### 8.1 Model Acquisition

```bash
# Download standard ONNX models for testing
pip install onnx-models  # or download from ONNX Model Zoo

# Specific models needed:
# - mobilenetv2-12.onnx (classification)
# - yolov8n.onnx (detection)
# - resnet50-v2-7.onnx (benchmark)
```

### 8.2 Calibration Data

| Dataset | Purpose | Size |
|---------|---------|------|
| ImageNet subset (100 images) | INT8/INT4 calibration for vision models | ~50 MB |
| LibriSpeech subset (10 clips) | Audio model calibration | ~20 MB |
| Random noise tensors | Smoke-test fallback | Generated at runtime |

---

## 9. Prerequisite Verification Checklist

Run these checks to confirm environment readiness before development begins.

### Phase 1 Readiness

```bash
# Python environment
python --version                     # >= 3.10
pip show fastmcp                     # installed
pip show qai-hub                     # installed

# QNN SDK
echo $QNN_SDK_ROOT                   # path set
ls $QNN_SDK_ROOT/bin/qnn-net-run     # binary exists
qnn-net-run --version                # runs without error

# Hexagon SDK
echo $HEXAGON_SDK_ROOT               # path set
hexagon-clang --version              # compiler accessible

# AIMET
python -c "import aimet_torch"       # importable

# Snapdragon Profiler
which sdp                            # CLI available

# AI Hub connectivity
python -c "import qai_hub; qai_hub.get_devices()"  # API reachable

# Device (if hardware present)
# Check NPU availability on WoS device
```

### Phase 2 Readiness

```bash
# Cross-compilation toolchain
aarch64-linux-gnu-gcc --version      # installed

# SNPE SDK
echo $SNPE_ROOT                      # path set
snpe-dlc-info --version              # runs

# Device connectivity
ssh user@arduino-uno-q "uname -a"    # SSH works
# or
picocom /dev/ttyUSB0                 # serial works

# SNPE on target
ssh user@arduino-uno-q "snpe-net-run --version"
```

### Phase 3 Readiness

```bash
# Android tools
adb version                          # installed
adb devices                          # device listed
android list targets                 # API 35+ available

# NDK
echo $ANDROID_NDK_HOME               # path set
$ANDROID_NDK_HOME/ndk-build --version

# SNPE Android libraries
ls $SNPE_ROOT/lib/aarch64-android/libSNPE.so

# Device NPU
adb shell "cat /sys/class/npu/npu0/status"  # available
```

### Phase 4 Readiness

```bash
# AIMET INT4
python -c "from aimet_torch.v2.quantization import Int4"  # importable

# AI Hub CI integration
qai-hub --version
qai-hub benchmark --help             # CLI works

# Dashboard
python -c "import plotly; import dash"  # importable
```

---

## Dependency Version Matrix

| Dependency | Min Version | Recommended | Notes |
|------------|-------------|-------------|-------|
| Python | 3.10 | 3.11 | ARM64 native on WoS |
| FastMCP | latest | latest | Rapid development |
| QNN SDK | 2.22 | 2.x latest | Check QDN for releases |
| SNPE SDK | 2.18 | 2.x latest | Must match QNN version |
| Hexagon SDK | 5.3 | 5.x latest | Required for custom ops |
| PyTorch | 2.0 | 2.2+ | ONNX export stability |
| ONNX | 1.14 | 1.16+ | Opset 18+ for modern models |
| ONNX Runtime | 1.16 | 1.18+ | QNN EP support |
| AIMET | 1.32 | latest | INT4 requires recent version |
| Android NDK | r26 | r27+ | ARM64 optimizations |
| ADB | 34.0 | latest | Android 15 compatibility |

---

## Notes

- All Qualcomm SDKs require acceptance of their respective license agreements via QDN
- QPM3 hardware measurement jig is optional; software estimation mode available for initial development
- Docker images with pre-configured SDKs are recommended for CI/CD to avoid host machine SDK sprawl
- Keep SDKs version-aligned (QNN + SNPE + Hexagon from same quarterly release) to avoid compatibility issues
- WoS (Windows on Snapdragon) native Python is required for NPU inference; x86 emulated Python may not access NPU
