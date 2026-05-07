# Phased Implementation Guide — QUAD Agent

## Overview

| Attribute | Value |
|-----------|-------|
| Stack | Python 3.11, FastMCP, Pydantic v2, Jinja2, pytest, structlog |
| Architecture | Mock-first, adapter pattern, 5 MCP tools |
| Platforms | Windows (Snapdragon X Elite), Linux (QCS2210), Android (Snapdragon 8 Elite) |
| IDE Plugins | VS Code (TypeScript), Arduino IDE (Java), Android Studio (Kotlin) |
| Target | TTFI < 10 minutes for all platforms |

---

## Phase 0: Foundation (Week 1–2)

### Sprint 0.1: Project Scaffolding

**Goal**: Running MCP server with 5 stub tools.

**Tasks**:
1. Initialize project structure:
   ```
   src/quad/{server,tools,adapters,platforms,codegen,models}/
   tests/{unit,integration,e2e,fixtures}/
   templates/{windows,linux,android}/
   configs/device_profiles/
   plugins/{vscode,arduino,android-studio}/
   ```
2. Create `pyproject.toml`:
   ```toml
   [project]
   name = "quad-agent"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = [
       "fastmcp>=0.1",
       "pydantic>=2.0",
       "pydantic-settings>=2.0",
       "jinja2>=3.1",
       "structlog>=24.0",
       "httpx>=0.27",
       "aiofiles>=24.0",
       "tomli>=2.0; python_version<'3.11'",
       "typer>=0.12",
       "python-dotenv>=1.0",
   ]

   [project.optional-dependencies]
   dev = ["pytest>=7.0", "pytest-asyncio", "pytest-cov", "pytest-mock", "ruff", "mypy", "pre-commit"]
   real = ["asyncssh", "paramiko", "pyserial"]

   [project.scripts]
   quad-server = "quad.server.main:cli"
   ```
3. Create FastMCP server entry point (`src/quad/server/main.py`) that registers 5 tools returning `{"status": "stub", "tool": "<name>"}`.
4. Create `quad.toml.example` with all configuration sections.
5. Create `Makefile`:
   ```makefile
   .PHONY: serve test lint format

   serve:
       python -m quad.server.main

   test:
       pytest tests/ -v --cov=src/quad --cov-report=term-missing

   test-unit:
       pytest tests/unit/ -v

   lint:
       ruff check src/ tests/
       mypy src/

   format:
       ruff format src/ tests/
   ```
6. Create `.pre-commit-config.yaml` (ruff + mypy).

**Verification**:
```bash
pip install -e ".[dev]"
make serve          # Server starts on stdio
make test           # 5 stub tests pass
make lint           # No errors
```

**Exit Criteria**: `fastmcp dev src/quad/server/main.py` starts; `quad-server` CLI works.

---

### Sprint 0.2: Core Infrastructure

**Goal**: Complete data model layer, exception hierarchy, mock adapter returning realistic data.

**Tasks**:
1. Implement Pydantic models (`src/quad/models/`):
   - `device.py` → `DeviceProfile`
   - `conversion.py` → `ConversionRequest`, `ConversionResult`
   - `profiling.py` → `ProfileRequest`, `ProfilingReport`, `LatencyStats`, `LayerProfile`
   - `orchestration.py` → `OrchestrationRequest`, `AllocationMap`
   - `codegen.py` → `CodegenRequest`, `GeneratedCode`
   - `config.py` → `ServerConfig` (Pydantic Settings)
   - `errors.py` → `ToolError`
2. Implement exception hierarchy (`src/quad/exceptions.py`):
   - `QUADError` → `PlatformError`, `SDKError`, `ConversionError`, `ProfilingError`, `OrchestrationError`, `CodegenError`
   - Each with `code`, `message`, `recoverable` fields
3. Implement config loader (`src/quad/config.py`):
   - Load `quad.toml`, merge `QUAD_*` env vars
   - Return validated `ServerConfig` instance
4. Implement structlog logging (`src/quad/logging.py`):
   - JSON processor for production, console for dev
   - Context binding (tool_name, request_id)
5. Implement `SDKAdapter` ABC (`src/quad/adapters/base.py`):
   - Abstract methods: `detect_hardware`, `convert_model`, `profile`, `get_supported_ops`
6. Implement `MockAdapter` (`src/quad/adapters/mock_adapter.py`):
   - Pre-defined device profiles (from `configs/device_profiles/*.json`)
   - Realistic conversion metrics (size reduction based on quantization)
   - Simulated profiling data (latency proportional to model complexity)
   - Heuristic allocation maps
7. Implement `AdapterFactory` (`src/quad/adapters/factory.py`):
   - Config-driven selection (mock vs real)
   - Adapter registry with lazy initialization

**Verification**:
```bash
make test-unit      # All model validation tests pass
python -c "from quad.adapters.factory import AdapterFactory; print('OK')"
```

**Exit Criteria**: MockAdapter returns valid, schema-conforming responses for all 5 tool operations.

---

### Sprint 0.3: Code Generation Engine

**Goal**: Template engine produces syntactically valid code for all 5 language targets.

**Tasks**:
1. Implement Jinja2 engine (`src/quad/codegen/engine.py`):
   - Template discovery from `templates/` directory
   - Variable injection (model_path, sdk_imports, runtime_config)
   - Multi-file output support (source + build file + config)
2. Create base templates:
   - `templates/windows/cpp/inference.cpp.j2` — QNN C++ inference
   - `templates/windows/cpp/CMakeLists.txt.j2` — CMake build
   - `templates/windows/python/inference.py.j2` — QNN Python inference
   - `templates/linux/python/inference.py.j2` — SNPE Python inference
   - `templates/linux/arduino_sketch/inference.ino.j2` — Arduino sketch
   - `templates/linux/arduino_sketch/CMakeLists.txt.j2`
   - `templates/android/kotlin/InferenceEngine.kt.j2` — Android Kotlin
   - `templates/android/kotlin/build.gradle.kts.j2` — Gradle build
   - `templates/android/jni/inference_jni.cpp.j2` — JNI bridge
3. Implement output validators (`src/quad/codegen/validators.py`):
   - Python: `ast.parse()` — catches syntax errors
   - C++: bracket/brace balance + basic structure check
   - Kotlin/Java: bracket balance + package declaration check
   - All: non-empty output, no unrendered `{{ }}` placeholders
4. Write unit tests for each template with sample variables

**Verification**:
```bash
make test-unit      # Template rendering tests pass
python -c "
from quad.codegen.engine import CodegenEngine
engine = CodegenEngine('templates/')
result = engine.render('windows', 'python', {'model_path': 'model.onnx', 'sdk': 'qnn'})
print(result.source_files.keys())
"
```

**Exit Criteria**: All templates render without errors; Python output passes `ast.parse()`.

---

## Phase 1: AI PC / Windows (Week 3–6)

### Sprint 1.1: hardware_detect Tool

**Goal**: Fully functional hardware detection for Windows platform.

**Tasks**:
1. Implement `WindowsPlatform` (`src/quad/platforms/windows.py`):
   - WMI queries for CPU (Win32_Processor), GPU (Win32_VideoController)
   - QNN device enumeration (`qnn-device-info` CLI or Python bindings)
   - SDK path detection (registry + common paths + env vars)
   - NPU detection via Hexagon runtime check
2. Implement `hardware_detect` tool handler (`src/quad/tools/hardware_detect.py`):
   - Input validation (platform enum)
   - Platform selection → adapter dispatch
   - Response caching (hardware doesn't change per session)
   - Error handling: `PlatformNotDetectedError`, `SDKNotFoundError`
3. Mock mode: Return pre-defined Snapdragon X Elite profile from `configs/device_profiles/snapdragon_x_elite.json`
4. Create device profile JSON files:
   ```json
   {
     "chipset": "Snapdragon X Elite X1E-80-100",
     "platform": "windows",
     "cpu_cores": 12,
     "cpu_arch": "Oryon ARM64",
     "cpu_freq_ghz": 3.8,
     "gpu_model": "Adreno X1-85",
     "gpu_tflops": 4.6,
     "npu_model": "Hexagon NPU",
     "npu_tops": 45,
     "ram_gb": 32,
     "available_runtimes": ["cpu", "gpu", "npu"]
   }
   ```
5. Unit tests + integration test (mock MCP call → JSON response)

**Verification**:
```bash
# MCP tool call via fastmcp dev
echo '{"method":"tools/call","params":{"name":"hardware_detect","arguments":{"platform":"windows"}}}' | quad-server
# Returns valid DeviceProfile JSON
```

---

### Sprint 1.2: convert_model Tool

**Goal**: Model conversion pipeline for QNN target (ONNX → QNN format).

**Tasks**:
1. Implement `QNNAdapter.convert_model()`:
   - Pipeline: validate input → `qnn-onnx-converter` → `qnn-model-lib-generator` → `qnn-context-binary-generator`
   - Parse converter output for unsupported ops
   - Quantization: pass `--input_list` calibration data for INT8
   - Output validation: check file exists, size > 0
2. Implement `convert_model` tool handler (`src/quad/tools/convert_model.py`):
   - Input: source_format, model_path, target_sdk, quantization
   - Format validation (file extension matches declared format)
   - Adapter dispatch (QNN or SNPE based on target_sdk)
   - Timeout handling (300s for large models)
3. Mock mode:
   - Parse ONNX model metadata (opset, graph structure) via `onnx` library
   - Simulate conversion: calculate size reduction (INT8 = ~4x, INT4 = ~8x)
   - Generate realistic unsupported ops list based on known QNN limitations
4. Create `configs/supported_ops/qnn_ops.json`:
   - List of QNN-supported ONNX operators
   - Used by mock to determine `supported_ops_pct`
5. Tests: happy path, unsupported format, missing file, large model timeout

**Verification**:
```bash
# With sample ONNX model
python -c "
from quad.tools.convert_model import convert_model
result = await convert_model(source_format='onnx', model_path='tests/fixtures/models/mobilenetv2.onnx', target_sdk='qnn', quantization='int8')
assert result.supported_ops_pct > 90
"
```

---

### Sprint 1.3: profile_workload Tool

**Goal**: Automated profiling with structured output.

**Tasks**:
1. Implement profiler adapters:
   - `QNNAdapter.profile()`: invoke `qnn-net-run` with timing flags, parse output
   - QPM3 integration: launch power capture, correlate with inference timing
   - `AIHubAdapter.profile()`: REST API call to AI Hub benchmark endpoint
2. Implement `profile_workload` tool handler (`src/quad/tools/profile_workload.py`):
   - Input: model_path, platform, runtime, duration_s, iterations
   - Select profiler based on platform
   - Concurrent: run inference + capture profiler output
   - Parse output into `ProfilingReport` with per-layer breakdown
3. Mock mode:
   - Generate metrics based on model metadata:
     - `latency_ms ≈ model_size_mb × 0.5` (NPU), `× 2.0` (CPU)
     - `power_mw ≈ runtime_factor × latency`
     - `throughput_fps = 1000 / latency_ms`
   - Generate per-layer timing from ONNX graph weights
4. Tests: each runtime mode, timeout handling, device not connected

**Verification**:
```bash
make test-unit  # Profile tool tests pass
# Mock returns plausible: latency > 0, power > 0, throughput > 0
```

---

### Sprint 1.4: orchestrate_workload Tool

**Goal**: Intelligent layer-to-runtime allocation.

**Tasks**:
1. Implement allocation algorithm (`src/quad/tools/orchestrate_workload.py`):
   - Parse model layer graph (from ONNX or profile_report.layers)
   - Check each op against supported_ops list for NPU/GPU
   - Apply power_mode strategy:
     - **performance**: all compatible → NPU, rest → GPU, remainder → CPU
     - **balanced**: top 70% compute → NPU, next 20% → GPU, 10% → CPU
     - **efficiency**: only highest-compute layers → NPU, everything else → CPU
   - Memory constraint: `sum(layer_memory) < device_ram × 0.8`
   - If exceeded: move layers from NPU/GPU → CPU until within budget
2. Implement projected metrics calculation:
   - `projected_latency = sum(layer.latency × runtime_speedup_factor)`
   - `projected_power = sum(layer.power × runtime_power_factor)`
3. Mock mode: same algorithm works (uses mock profile data)
4. Tests: each power mode, memory overflow case, all ops unsupported (CPU fallback)

**Verification**:
```bash
# All layers get assigned; no layer left unallocated
# Power mode changes allocation ratios (performance → more NPU)
```

---

### Sprint 1.5: generate_code Tool

**Goal**: Produce compilable inference code for Windows (C++, Python).

**Tasks**:
1. Implement `generate_code` tool handler (`src/quad/tools/generate_code.py`):
   - Input: platform, sdk, language, model_path, allocation_map
   - Select template based on (platform, language) combination
   - Inject variables: model_path, SDK paths from config, allocation hints
   - Validate output with codegen validators
   - Return `GeneratedCode` with source_files dict, build instructions, dependencies
2. Enhance C++ template (`templates/windows/cpp/inference.cpp.j2`):
   - QNN headers and initialization
   - Model loading from context binary
   - Input tensor preparation (from image/numpy)
   - Graph execution
   - Output extraction and formatting
   - Cleanup/teardown
3. Enhance Python template (`templates/windows/python/inference.py.j2`):
   - QNN Python SDK imports
   - Session creation with device selection
   - NumPy input preparation
   - Inference execution
   - Result postprocessing
4. Tests: each language target produces valid output; build instructions are correct

**Verification**:
```bash
python -c "
from quad.tools.generate_code import generate_code
result = await generate_code(platform='windows', sdk='qnn', language='python', model_path='model.qnn')
import ast; ast.parse(result.source_files['inference.py'])  # Valid Python
"
```

---

### Sprint 1.6: VS Code Extension

**Goal**: VS Code extension connecting to QUAD MCP server.

**Tasks**:
1. Scaffold extension (`plugins/vscode/`):
   ```bash
   npx yo code --type=typescript --name=quad-vscode
   ```
2. Implement MCP client (stdio transport):
   - Spawn `quad-server` as child process
   - Send/receive JSON-RPC messages
   - Connection lifecycle management (start/stop/restart)
3. Register commands:
   - `quad.detectHardware` → calls `hardware_detect`
   - `quad.convertModel` → file picker + calls `convert_model`
   - `quad.profileWorkload` → calls `profile_workload`
   - `quad.orchestrateWorkload` → calls `orchestrate_workload`
   - `quad.generateCode` → language picker + calls `generate_code`
4. UI components:
   - Output channel: "QUAD Agent" (JSON tool results)
   - Status bar item: server connection state
   - Webview panel: profiling results visualization (optional, stretch)
5. Extension settings:
   - `quad.serverCommand`: path to quad-server
   - `quad.configPath`: path to quad.toml
   - `quad.adapterMode`: mock/real toggle

**Verification**:
```bash
cd plugins/vscode && npm run compile  # Builds without errors
# Manual: install VSIX, run "QUAD: Detect Hardware" command
```

---

### Sprint 1.7: Integration & E2E Testing

**Goal**: Full pipeline validated end-to-end.

**Tasks**:
1. E2E test (`tests/e2e/test_windows_e2e.py`):
   - Start MCP server in mock mode
   - Call hardware_detect → get device profile
   - Call convert_model with sample ONNX → get conversion result
   - Call profile_workload with converted model → get profiling report
   - Call orchestrate_workload with profile → get allocation map
   - Call generate_code with allocation → get source code
   - Validate: all outputs conform to schemas; no errors
2. TTFI measurement:
   - Time from `quad-server` start to complete code generation
   - Target: < 30 seconds in mock mode (real SDK adds conversion time)
3. Performance benchmarks:
   - Each tool call latency (target < 10s for generate_code)
   - Memory usage during conversion (target < 500MB in mock)
4. Documentation:
   - Update CLAUDE.md phase checklist
   - Write user-facing quickstart guide

**Verification**:
```bash
make test               # All tests pass (unit + integration + e2e)
pytest tests/e2e/ -v    # E2E specifically passes
```

**Exit Criteria**: Phase 1 complete. All 5 tools work on Windows (mock mode). VS Code extension functional.

---

## Phase 2: Arduino UNO Q / Linux (Week 7–9)

### Sprint 2.1: SNPE Adapter

**Goal**: SNPE SDK operations behind adapter interface.

**Tasks**:
1. Implement `SNPEAdapter` (`src/quad/adapters/snpe_adapter.py`):
   - `convert_model()`: `snpe-onnx-to-dlc` → `snpe-dlc-quantize` pipeline
   - `profile()`: `snpe-throughput-net-run` + parse CSV timing output
   - `get_supported_ops()`: parse SNPE op list for DSP backend
   - `detect_hardware()`: `snpe-platform-validator` or device query
2. Create `configs/supported_ops/snpe_ops.json`:
   - DSP-compatible ops (Conv2D, DepthwiseConv, Relu, Pool, etc.)
   - CPU-fallback ops (custom ops, unsupported activations)
3. Mock mode enhancement:
   - SNPE-specific response formatting (DLC file extension, DSP timing)
   - Different performance characteristics than QNN (lower TOPS, higher efficiency)
4. Tests: conversion pipeline, profiler output parsing, op compatibility check

**Verification**: SNPE adapter passes same test interface as QNN adapter.

---

### Sprint 2.2: Linux Platform & Device Access

**Goal**: Remote device communication for Arduino UNO Q.

**Tasks**:
1. Implement `LinuxPlatform` (`src/quad/platforms/linux.py`):
   - SSH connection management (asyncssh):
     - Connect, authenticate, maintain session
     - Execute remote commands, capture output
   - SCP file transfer (model upload, result download)
   - Serial console fallback (pyserial for UART debug)
   - Device health check: memory, storage, running processes
2. Hardware detection for QCS2210:
   - Remote `/proc/cpuinfo` parsing
   - DSP availability check (`/dev/cdsp` exists)
   - SNPE runtime version on device
3. File deployment:
   - Push model DLC to device `/opt/models/`
   - Push SNPE runtime libraries if missing
   - Verify file integrity post-transfer (checksum)
4. Tests: mock SSH session, file transfer simulation, health check parsing

**Verification**: Can connect (mock), deploy files, execute remote inference command.

---

### Sprint 2.3: DSP Profiling & Power-Constrained Orchestration

**Goal**: Profiling on Hexagon DSP with 3W power budget enforcement.

**Tasks**:
1. SNPE profiler integration:
   - Remote `snpe-throughput-net-run --perf_profile burst --duration 10`
   - Parse output: per-layer timing, runtime assignment, memory
2. Hexagon profiler integration:
   - Remote `hexagon-prof` execution (if available)
   - Cycle count and VTCM utilization metrics
3. Linux perf integration:
   - Remote `perf stat` for CPU-level metrics
   - Power estimation from CPU frequency/utilization
4. Power-constrained orchestration:
   - QCS2210 power model: DSP ~1W active, CPU ~2W active
   - Hard cap: total inference power < 3W
   - Algorithm: assign to DSP first; if power > 3W, move layers to CPU (lower performance but same power budget — fewer active units)
5. Tests: power budget exceeded → layers moved to CPU

**Verification**: Profile reports include DSP metrics; allocation stays within 3W.

---

### Sprint 2.4: Arduino Sketch Generation

**Goal**: Generate complete Arduino project from single tool call.

**Tasks**:
1. Enhance Arduino sketch template (`templates/linux/arduino_sketch/inference.ino.j2`):
   - SNPE runtime initialization
   - Model loading from filesystem
   - Input tensor preparation (camera/sensor data)
   - DSP inference execution
   - Output parsing and serial output
   - Loop with timing measurement
2. Generate supporting files:
   - `CMakeLists.txt` for cross-compilation
   - `deploy.sh` — SCP + SSH deployment script
   - `README.md` — build and deploy instructions
   - `.vscode/tasks.json` — build/deploy tasks
3. Cross-compilation support:
   - CMake toolchain file for aarch64-linux-gnu
   - SNPE library linking paths
4. Tests: generated sketch has valid C++ structure; CMake file references correct paths

**Verification**: Generated project passes C++ syntax validation; deploy script is executable.

---

### Sprint 2.5: Arduino IDE Plugin

**Goal**: Arduino IDE integration for QUAD tools.

**Tasks**:
1. Plugin scaffold (`plugins/arduino/`):
   - Arduino IDE 2.x extension point (based on Arduino CLI)
   - Board definition for "Arduino UNO Q (QCS2210)"
2. Tool menu integration:
   - "QUAD: Deploy AI Model" → convert + deploy to device
   - "QUAD: Profile Inference" → remote profiling
   - "QUAD: Generate Sketch" → template rendering
3. MCP connection via HTTP/SSE transport (Arduino IDE doesn't support stdio plugins)
4. Serial monitor integration: parse inference output, display metrics
5. Tests: plugin loads without error; commands registered

**Verification**: Plugin compiles; board definition appears in board manager.

---

## Phase 3: Mobile / Android (Week 10–12)

### Sprint 3.1: Android Platform & ADB Integration

**Goal**: Full ADB device lifecycle management.

**Tasks**:
1. Implement `AndroidPlatform` (`src/quad/platforms/android.py`):
   - Device detection: `adb devices` parsing
   - Device properties: `adb shell getprop` (chipset, Android version, NPU status)
   - Library deployment: `adb push` SNPE/QNN .so files
   - Command execution: `adb shell` with output capture
   - Logcat monitoring: `adb logcat -s QUAD:*` for inference output
   - APK management: install, launch, uninstall
2. Device selection:
   - Multiple device handling (serial-based selection)
   - Auto-select if single device connected
3. Security:
   - Validate device serial against `adb devices` output
   - Sanitize all shell commands (no string interpolation)
4. Tests: mock ADB subprocess calls, multi-device scenarios

**Verification**: Device detected via mock ADB; properties extracted correctly.

---

### Sprint 3.2: Android Model Conversion & Profiling

**Goal**: Models convert and profile on Android targets.

**Tasks**:
1. SNPE DLC for Android:
   - Target architecture: `aarch64-android`
   - Library path: `/data/local/tmp/snpe/`
   - Model path: `/data/local/tmp/models/`
2. QNN binary for Android:
   - Target: `aarch64-android` with Hexagon backend
3. Profiling pipeline:
   - Push model + runtime to device via ADB
   - Execute `snpe-net-run` on device, capture timing
   - Snapdragon Profiler: connect via USB, capture trace
   - Perfetto: push trace config, capture, pull trace file, parse
4. Parse Perfetto traces:
   - Extract GPU scheduling events
   - Extract NPU inference markers
   - Calculate utilization percentages
5. Tests: conversion produces Android-compatible output; profiling parses correctly

**Verification**: DLC generated for `aarch64-android`; profile JSON includes GPU/NPU metrics.

---

### Sprint 3.3: Thermal-Aware Orchestration

**Goal**: Dynamic workload adaptation based on device thermal state.

**Tasks**:
1. Thermal monitoring:
   - `adb shell cat /sys/class/thermal/thermal_zone*/temp` — all zones
   - Map zones to components (CPU, GPU, NPU, battery)
   - Define thresholds: warning (70°C), throttle (80°C), critical (90°C)
2. Battery state:
   - `adb shell dumpsys battery` — charging status, level, temperature
   - Adjust strategy: charging = more aggressive; discharging = conservative
3. Dynamic orchestration:
   - Input: current thermal state + profile report + power mode
   - If NPU thermal > warning: shift 30% of NPU layers to GPU
   - If GPU thermal > warning: shift to CPU
   - If battery < 20% and discharging: force efficiency mode
4. Orchestration response includes:
   - `thermal_state: {cpu_temp, gpu_temp, npu_temp}`
   - `thermal_action: "none" | "shift_to_gpu" | "shift_to_cpu" | "throttle"`
5. Tests: thermal threshold triggers correct allocation changes

**Verification**: Allocation changes when mock thermal data crosses thresholds.

---

### Sprint 3.4: Android AAR Generation

**Goal**: Complete Android library project from single tool call.

**Tasks**:
1. AAR template structure:
   - `InferenceEngine.kt` — public API (init, run inference, cleanup)
   - `build.gradle.kts` — dependencies, NDK config, SNPE AAR inclusion
   - `jni/inference_jni.cpp` — JNI bridge to native SNPE/QNN
   - `CMakeLists.txt` — NDK native build
   - `proguard-rules.pro` — keep SNPE classes
   - `AndroidManifest.xml` — library manifest
2. Sample app generation (optional):
   - `MainActivity.kt` — camera input + inference demo
   - `activity_main.xml` — preview + result overlay
3. Build instructions:
   - Gradle commands for AAR assembly
   - ADB commands for sample app deployment
4. Tests: generated Gradle structure is valid; Kotlin compiles (basic syntax check)

**Verification**: Generated project has all required files; `build.gradle.kts` parses without error.

---

### Sprint 3.5: Android Studio Plugin

**Goal**: IntelliJ-platform plugin for Android Studio.

**Tasks**:
1. Plugin scaffold (`plugins/android-studio/`):
   - IntelliJ Platform SDK (Gradle-based build)
   - Plugin descriptor: `plugin.xml`
   - Compatibility: Android Studio Ladybug+
2. Tool window:
   - Device selector panel (ADB devices list)
   - Model converter form (file picker, quantization dropdown)
   - Profiler launcher (runtime selection, duration)
   - Results display (JSON tree view)
3. Run configuration:
   - "QUAD AI Inference" run config type
   - Deploys model + runs inference on selected device
4. MCP connection: stdio transport to local quad-server
5. Tests: plugin builds; tool window renders with mock data

**Verification**: Plugin JAR builds; installs in Android Studio without errors.

---

## Phase 4: Cross-Platform & Optimization (Week 13–16)

### Sprint 4.1: AIMET INT4 Quantization

**Tasks**:
1. Enhance `AIMETAdapter`:
   - INT4 mixed-precision quantization pipeline
   - Per-layer sensitivity analysis (which layers tolerate INT4)
   - Calibration data ingestion (representative dataset, 100-1000 samples)
   - Accuracy impact estimation (simulated quantization noise)
2. Mixed-precision strategy:
   - Sensitive layers (first/last conv, attention) → INT8 or FP16
   - Insensitive layers (middle conv, linear) → INT4
   - Decision based on sensitivity analysis output
3. Integration with `convert_model`:
   - `quantization="int4"` triggers AIMET pipeline before SDK conversion
   - Output includes per-layer precision map
4. Tests: INT4 produces smaller model; accuracy estimation within bounds

---

### Sprint 4.2: AI Hub CI/CD Integration

**Tasks**:
1. GitHub Actions workflow template:
   - Trigger: push to main, PR to main
   - Steps: install QUAD → convert model → benchmark on AI Hub → compare with baseline
   - Fail if: latency regression > 10%, accuracy drop > 1%
2. AI Hub API integration:
   - Submit model for profiling across target devices
   - Poll for results (async with timeout)
   - Parse benchmark JSON (latency, accuracy, power per device)
3. Badge generation:
   - Performance badge: "Latency: 5.2ms on X Elite NPU"
   - Compatibility badge: "95% ops supported"
4. Workflow template stored in `templates/ci/github_actions.yml.j2`
5. Tests: generated YAML is valid; API client handles errors

---

### Sprint 4.3: Multi-Model Pipeline

**Tasks**:
1. Pipeline definition format (YAML):
   ```yaml
   pipeline:
     name: "detection_classification"
     stages:
       - model: yolov8n.onnx
         task: detection
         output: bounding_boxes
       - model: mobilenetv2.onnx
         task: classification
         input_from: stage_0.crops
         output: class_labels
   ```
2. Pipeline executor:
   - Parse pipeline YAML
   - Execute stages sequentially (future: parallel branches)
   - Pass outputs between stages (tensor routing)
   - Aggregate profiling across all stages
3. Memory budget analysis:
   - Calculate peak memory (all models loaded simultaneously vs sequential)
   - Warn if combined footprint exceeds device RAM × 0.8
4. Pipeline profiling:
   - Per-stage latency + total pipeline latency
   - Bottleneck identification (which stage dominates)
5. Tests: 3-stage pipeline executes; memory calculated correctly

---

### Sprint 4.4: Cross-Platform Dashboard

**Tasks**:
1. Dashboard generation (`src/quad/dashboard/`):
   - Input: profiling reports from multiple platforms
   - Output: interactive HTML file (Plotly + Chart.js)
2. Visualizations:
   - Latency comparison bar chart (same model across platforms)
   - Power/performance Pareto scatter plot
   - Layer-level runtime allocation heatmap
   - NPU utilization gauge per platform
   - Quantization impact chart (FP32 vs INT8 vs INT4)
3. Export options:
   - Interactive HTML (self-contained, no server needed)
   - Static PNG/PDF for reports
   - JSON data export
4. Template: `templates/dashboard/report.html.j2`
5. Tests: dashboard renders with sample data; all charts have data

---

### Sprint 4.5: Final Integration & Polish

**Tasks**:
1. Cross-platform E2E test suite:
   - Same model through all 3 platforms (mock mode)
   - Compare outputs are consistent (same schema, reasonable metrics)
2. Performance optimization:
   - Cache hardware detection per session
   - Async adapter calls where possible
   - Template pre-compilation for code generation
3. Security audit:
   - Verify no credentials in any output or log
   - Path traversal testing
   - ADB command injection testing
4. Documentation:
   - API reference (auto-generated from Pydantic models)
   - User guide (quickstart, configuration, troubleshooting)
   - Contributor guide (adding new adapters, templates, platforms)
5. Release packaging:
   - PyPI: `pip install quad-agent`
   - VS Code Marketplace: `.vsix` package
   - GitHub Releases: versioned artifacts

**Exit Criteria**: All tests green, docs complete, packages built, ready for deployment.

---

## Dependencies Between Sprints

```
Phase 0:  0.1 ──→ 0.2 ──→ 0.3
                     │
Phase 1:  ←─────────┘
          1.1 ──→ 1.2
          1.1 ──→ 1.3 ──→ 1.4
          1.2 + 1.4 ──→ 1.5
          (any tool) ──→ 1.6
          (all) ──→ 1.7

Phase 2:  ←── Phase 0 complete
          2.1 ──→ 2.2 ──→ 2.3
          2.1 + 2.3 ──→ 2.4
          (any tool) ──→ 2.5

Phase 3:  ←── Phase 0 + SNPE adapter (2.1)
          3.1 ──→ 3.2 ──→ 3.3
          3.2 ──→ 3.4
          (any tool) ──→ 3.5

Phase 4:  ←── All phases
          4.1 (independent)
          4.2 (independent)
          4.1 ──→ 4.3
          (all platforms) ──→ 4.4
          (all) ──→ 4.5
```

---

## Information Required Before Development Starts

### Critical (Blocks Sprint 0.1)

| # | Information Needed | Why | Source |
|---|---|---|---|
| 1 | FastMCP latest API reference | Tool registration, server lifecycle, transport options | FastMCP docs/source |
| 2 | Pydantic v2 Settings integration | Config management pattern | Pydantic docs |
| 3 | MCP protocol specification | JSON-RPC schema, error codes, tool definitions | MCP spec |

### Required for Phase 1 (Blocks Sprint 1.1–1.5)

| # | Information Needed | Why | Source |
|---|---|---|---|
| 4 | QNN SDK 2.x Python API reference | Exact method signatures for adapter | QDN portal / SDK docs |
| 5 | QNN CLI tool flags and output formats | `qnn-onnx-converter`, `qnn-net-run`, `qnn-model-lib-generator` | QDN portal |
| 6 | QNN supported ONNX operators list | Op compatibility checking for mock | QNN SDK release notes |
| 7 | Qualcomm AI Hub REST API documentation | Endpoints, auth, request/response schemas | AI Hub developer docs |
| 8 | QPM3 output format specification | Power trace JSON/CSV structure | QPM3 documentation |
| 9 | Snapdragon Profiler CLI (`sdp`) documentation | Command flags, trace export formats | QDN portal |
| 10 | ONNX opset 18 operator list | Template code generation for different ops | ONNX spec |
| 11 | Sample QNN SDK CLI output | Realistic mock responses | Run on actual device or from docs |
| 12 | WoS (Windows on Snapdragon) WMI classes | Qualcomm-specific WMI entries for detection | WoS developer docs |

### Required for Phase 2 (Blocks Sprint 2.1–2.4)

| # | Information Needed | Why | Source |
|---|---|---|---|
| 13 | SNPE SDK 2.x CLI reference | `snpe-onnx-to-dlc`, `snpe-net-run` flags and output | QDN portal |
| 14 | SNPE DSP-supported operations | Which ops run on Hexagon V66 DSP | SNPE SDK docs |
| 15 | SNPE profiler output format | Per-layer timing CSV/JSON structure | SNPE SDK docs |
| 16 | QCS2210 device specs (exact) | Memory limits, filesystem layout, runtime paths | Arduino UNO Q docs |
| 17 | Arduino UNO Q BSP/SDK documentation | Board support package, Linux image details | Qualcomm Robotics RB1 docs |
| 18 | Hexagon DSP V66 capabilities | VTCM size, supported operations | Hexagon SDK docs |

### Required for Phase 3 (Blocks Sprint 3.1–3.4)

| # | Information Needed | Why | Source |
|---|---|---|---|
| 19 | SNPE Android AAR API reference | Java/Kotlin integration classes | SNPE SDK docs |
| 20 | Android 15 NNAPI delegation for QNN/SNPE | How SDK integrates with NNAPI | Android developer docs |
| 21 | Snapdragon 8 Elite thermal zone mappings | Which `/sys/class/thermal/thermal_zone*` maps to which component | Device-specific docs |
| 22 | Perfetto trace config for Adreno/Hexagon | Custom trace categories for GPU/NPU events | Perfetto docs + Qualcomm |
| 23 | ADB property keys for Snapdragon devices | `ro.board.platform`, NPU-related properties | Device testing |

### Required for Phase 4 (Blocks Sprint 4.1–4.3)

| # | Information Needed | Why | Source |
|---|---|---|---|
| 24 | AIMET INT4 quantization API | Mixed-precision API, sensitivity analysis methods | AIMET docs |
| 25 | AI Hub CI/CD integration docs | Webhook/API for automated benchmarking | AI Hub developer portal |
| 26 | QNN INT4 backend support status | Which QNN version supports INT4 execution | QNN release notes |

### Nice-to-Have (Improves Quality)

| # | Information Needed | Why | Source |
|---|---|---|---|
| 27 | Qualcomm reference inference apps | Code patterns to replicate in templates | QDN sample apps |
| 28 | Model compatibility matrix | Which models verified on which chipset | AI Hub benchmarks |
| 29 | Power budget per compute unit | Detailed TDP breakdown for allocation algorithm | Chipset datasheets |
| 30 | VS Code Extension API (latest) | Extension development best practices | VS Code docs |
| 31 | IntelliJ Platform SDK for Android Studio | Plugin development API | JetBrains docs |
| 32 | Arduino IDE 2.x extension API | Plugin extension points | Arduino docs |

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| SDK APIs change between versions | Medium | High | Version-pinned adapters with compatibility matrix; abstract interface isolates changes |
| No hardware access during dev | High | Medium | Mock-first architecture; all development works without hardware |
| QNN/SNPE output format undocumented | Medium | Medium | Reverse-engineer from sample outputs; contact Qualcomm DevRel |
| FastMCP API instability (early-stage library) | Medium | Medium | Pin version; wrap in thin adapter; contribute fixes upstream |
| IDE plugin APIs are complex | Medium | Low | Start with minimal viable plugins; VS Code first (best documented) |
| TTFI target < 10 min ambitious with real SDK | Low | Medium | Optimize critical path; pre-cache detection; parallel conversion+profiling |
| Cross-platform test matrix explosion | Medium | Medium | Focus on mock-mode CI; real-device testing in nightly/manual |

---

## Definition of Done (per Sprint)

- [ ] All unit tests passing (≥85% coverage for new code)
- [ ] Integration tests passing (mock mode minimum)
- [ ] Pydantic models validate all edge cases (valid + invalid inputs)
- [ ] Error handling covers all identified failure modes
- [ ] Structured logging in place for all tool operations
- [ ] No security issues (no secrets, validated inputs, no shell injection)
- [ ] Code passes `ruff check` + `mypy --strict` + `ruff format --check`
- [ ] CLAUDE.md Active Context updated with completion status
- [ ] Phase checklist item(s) marked complete

---

## Quick Reference: Key Commands

```bash
# Development
make serve                    # Start MCP server (mock mode)
make test                     # Run all tests with coverage
make test-unit                # Unit tests only
make lint                     # Ruff + mypy
make format                   # Auto-format code

# MCP Testing
fastmcp dev src/quad/server/main.py    # Interactive MCP development
fastmcp test src/quad/server/main.py   # Automated MCP tool testing

# IDE Plugin Development
cd plugins/vscode && npm run compile   # Build VS Code extension
cd plugins/vscode && npm run package   # Create .vsix

# Docker
docker build -t quad-agent .           # Build CI image
docker run quad-agent make test        # Run tests in container
```
