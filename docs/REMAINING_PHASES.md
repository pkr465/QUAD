# Remaining Phases — What's Left to Build

> **Current State**: Phase 0 complete, Phase 1 at ~70% (mock mode working, real QNN adapter pending)
> **Last Updated**: 2026-04-29 | **90 tests passing**

---

## Phase 1 — AI PC / Windows (Remaining Items)

### Blocked on QNN SDK Documentation

| Item | What's Needed | Status |
|------|---------------|--------|
| `QNNAdapter` (real mode) | CLI flags for `qnn-onnx-converter`, `qnn-net-run`, `qnn-model-lib-generator`, output format parsing | BLOCKED |
| QPM3 power profiling | QPM3 CLI/API, output JSON/CSV schema | BLOCKED |
| Snapdragon Profiler integration | `sdp` CLI flags, trace export format | BLOCKED |
| AI Hub REST API | Endpoints, auth, benchmark response schema | BLOCKED |
| AIMET INT8 quantization | Python API for `QuantizationSimModel`, calibration flow | BLOCKED |
| WoS device detection (real) | WMI class names for Qualcomm NPU on Windows | BLOCKED |

### Not Blocked (Can Build Now)

| Item | Description | Effort |
|------|-------------|--------|
| VS Code extension polish | Install deps (`npm install`), add webview for profiling charts | 2-3 hours |
| Hardware detection caching | Cache `hardware_detect` results per session in server | 30 min |
| Error recovery in tools | Graceful degradation when mock → real switch fails | 1 hour |

---

## Phase 2 — Arduino UNO Q / Linux (Full Phase)

### Sprint 2.1: SNPE Adapter
- [ ] `SNPEAdapter` class implementing `SDKAdapter` ABC
- [ ] `snpe-onnx-to-dlc` conversion pipeline (mock + real)
- [ ] `snpe-dlc-quantize` INT8 quantization
- [ ] `snpe-net-run` inference execution
- [ ] `snpe-throughput-net-run` profiling output parsing
- [ ] SNPE supported ops list (`configs/supported_ops/snpe_ops.json`)
- [ ] Mock mode: SNPE-specific response formatting (DLC extension, DSP timing characteristics)

### Sprint 2.2: Linux Platform & Device Access
- [ ] `LinuxPlatform` class implementing `Platform` ABC
- [ ] SSH connection management (asyncssh)
- [ ] SCP file transfer (model upload to device)
- [ ] Remote command execution with output capture
- [ ] Serial console fallback (pyserial for UART debug)
- [ ] Device health check (memory, storage, processes)
- [ ] QCS2210 hardware detection via remote `/proc/cpuinfo`

### Sprint 2.3: DSP Profiling & Power-Constrained Orchestration
- [ ] SNPE profiler integration (remote `snpe-throughput-net-run --perf_profile burst`)
- [ ] Hexagon profiler integration (`hexagon-prof` cycle counts, VTCM usage)
- [ ] Linux `perf stat` integration for CPU metrics
- [ ] Power-constrained orchestration (hard cap: total < 3W)
- [ ] QCS2210 power model: DSP ~1W, CPU ~2W active

### Sprint 2.4: Arduino Sketch Generation
- [ ] Enhanced Arduino sketch template (SNPE runtime init, DSP inference, serial output)
- [ ] CMakeLists.txt generation for cross-compilation (aarch64-linux-gnu)
- [ ] `deploy.sh` script generation (SCP + SSH to device)
- [ ] README generation with build/deploy instructions

### Sprint 2.5: Arduino IDE Plugin
- [ ] Java plugin scaffold for Arduino IDE 2.x
- [ ] Board definition: "Arduino UNO Q (QCS2210)"
- [ ] Tools menu: Deploy AI Model, Profile Inference, Generate Sketch
- [ ] MCP connection via HTTP/SSE transport
- [ ] Serial monitor integration (parse inference output)

**Phase 2 Dependencies**: Phase 0 complete (done), SNPE SDK docs (needed for real adapter)

---

## Phase 3 — Mobile / Android (Full Phase)

### Sprint 3.1: Android Platform & ADB Integration
- [ ] `AndroidPlatform` class implementing `Platform` ABC
- [ ] Device detection: `adb devices` parsing
- [ ] Device properties: `adb shell getprop` (chipset, Android version, NPU status)
- [ ] Library deployment: `adb push` SNPE/QNN .so files
- [ ] Remote command execution: `adb shell` with output capture
- [ ] Logcat monitoring: `adb logcat -s QUAD:*`
- [ ] APK management: install, launch, uninstall
- [ ] Multi-device handling (serial-based selection)
- [ ] Security: sanitize all ADB shell commands

### Sprint 3.2: Android Model Conversion & Profiling
- [ ] SNPE DLC for `aarch64-android` target
- [ ] QNN binary for Android Hexagon backend
- [ ] Snapdragon Profiler Android mode (USB/ADB connection)
- [ ] Perfetto trace capture: push config, capture, pull, parse
- [ ] Parse Perfetto traces for GPU scheduling + NPU inference markers

### Sprint 3.3: Thermal-Aware Orchestration
- [ ] Thermal monitoring via `adb shell cat /sys/class/thermal/thermal_zone*/temp`
- [ ] Zone-to-component mapping (CPU, GPU, NPU, battery)
- [ ] Thresholds: warning (70°C), throttle (80°C), critical (90°C)
- [ ] Battery state: `adb shell dumpsys battery` (charging status, level)
- [ ] Dynamic orchestration: shift layers NPU→GPU→CPU on thermal events
- [ ] Battery-aware strategy: charging = aggressive, discharging = conservative

### Sprint 3.4: AAR Generation
- [ ] Android AAR template: `InferenceEngine.kt` + `build.gradle.kts` + JNI bridge
- [ ] CMakeLists.txt for NDK native build
- [ ] ProGuard rules for SNPE classes
- [ ] AndroidManifest.xml for library
- [ ] Sample app generation (`MainActivity.kt` + camera demo)
- [ ] Build instructions: Gradle commands + ADB deployment

### Sprint 3.5: Android Studio Plugin
- [ ] IntelliJ Platform plugin scaffold (Kotlin, Gradle-based)
- [ ] plugin.xml descriptor (compatibility: Android Studio Ladybug+)
- [ ] Tool window: device selector, model converter, profiler, code generator
- [ ] Run configuration: "QUAD AI Inference" profile
- [ ] MCP connection: stdio to local quad-server

**Phase 3 Dependencies**: Phase 0 complete (done), SNPE adapter from Phase 2 (Sprint 2.1), Android device for real testing

---

## Phase 4 — Cross-Platform & Optimization (Full Phase)

### Sprint 4.1: AIMET INT4 Quantization
- [ ] Enhanced `AIMETAdapter`: INT4 mixed-precision pipeline
- [ ] Per-layer sensitivity analysis (which layers tolerate INT4)
- [ ] Calibration data ingestion (representative dataset, 100-1000 samples)
- [ ] Mixed-precision strategy: sensitive layers → INT8/FP16, insensitive → INT4
- [ ] Accuracy impact estimation (simulated quantization noise)
- [ ] Integration with `convert_model` tool (quantization="int4" triggers AIMET)

### Sprint 4.2: AI Hub CI/CD Integration
- [ ] GitHub Actions workflow template (`templates/ci/github_actions.yml.j2`)
- [ ] AI Hub API client: submit model → poll results → parse benchmarks
- [ ] Performance regression detection (latency/accuracy delta thresholds)
- [ ] Badge generation for README (latency, compatibility scores)
- [ ] `generate_ci_pipeline` tool or subcommand

### Sprint 4.3: Multi-Model Pipeline
- [ ] Pipeline definition format (YAML):
  ```yaml
  pipeline:
    stages:
      - model: yolov8n.onnx
        task: detection
      - model: mobilenetv2.onnx
        task: classification
        input_from: stage_0.crops
  ```
- [ ] Pipeline executor: parse YAML → sequential execution → tensor routing
- [ ] Memory budget analysis (combined footprint vs device RAM)
- [ ] Pipeline profiling: per-stage + total latency, bottleneck identification

### Sprint 4.4: Cross-Platform Dashboard
- [ ] Dashboard generation (`src/quad/dashboard/`)
- [ ] Plotly/Chart.js visualizations:
  - Latency comparison bar chart (same model across platforms)
  - Power/performance Pareto scatter plot
  - Layer-level runtime allocation heatmap
  - NPU utilization gauge per platform
  - Quantization impact chart (FP32 vs INT8 vs INT4)
- [ ] Self-contained interactive HTML output
- [ ] Export to PNG/PDF for reports

### Sprint 4.5: Final Integration & Polish
- [ ] Cross-platform E2E test suite (same model through all 3 platforms)
- [ ] Performance optimization: cache hardware detection, async ops, template pre-compilation
- [ ] Security audit: no credentials in output/logs, path traversal testing, ADB injection testing
- [ ] Documentation: API reference (auto-gen from Pydantic), user guide, contributor guide
- [ ] Release packaging: PyPI (`pip install quad-agent`), VS Code Marketplace, GitHub Releases
- [ ] Version 1.0 release criteria validation

**Phase 4 Dependencies**: All previous phases complete, AIMET docs (Sprint 4.1), AI Hub API docs (Sprint 4.2)

---

## Summary: What's Buildable Now vs. Blocked

### Can Build Without SDK Docs

| Phase | Sprint | Item |
|-------|--------|------|
| 2 | 2.1 | SNPE adapter (mock mode) |
| 2 | 2.2 | Linux platform SSH/SCP infrastructure |
| 2 | 2.4 | Arduino sketch template enhancement |
| 3 | 3.1 | Android platform ADB integration |
| 3 | 3.3 | Thermal monitoring logic |
| 3 | 3.4 | AAR template structure |
| 4 | 4.3 | Multi-model pipeline YAML parser/executor |
| 4 | 4.4 | Dashboard generation framework |

### Blocked on SDK/API Documentation

| Phase | Sprint | Item | Blocker |
|-------|--------|------|---------|
| 1 | — | QNN real adapter | QNN SDK CLI docs |
| 1 | — | QPM3 integration | QPM3 output format |
| 1 | — | AI Hub integration | REST API reference |
| 2 | 2.1 | SNPE real adapter | SNPE CLI docs |
| 2 | 2.3 | Hexagon profiler | Hexagon SDK docs |
| 3 | 3.2 | Perfetto trace parsing | Trace schema for Adreno/Hexagon |
| 4 | 4.1 | AIMET INT4 | AIMET Python API reference |
| 4 | 4.2 | AI Hub CI/CD | AI Hub webhook/REST API |

### Blocked on Hardware Access

| Phase | Sprint | Item | Hardware Needed |
|-------|--------|------|-----------------|
| 1 | — | Real device validation | Snapdragon X Elite laptop |
| 2 | 2.2-2.4 | Arduino deployment | Arduino UNO Q board |
| 3 | 3.1-3.4 | Android testing | Snapdragon 8 Elite phone |

---

## Estimated Remaining Effort

| Phase | Sprints | Mock Mode | Real Mode (after SDK docs) |
|-------|---------|-----------|---------------------------|
| Phase 1 remainder | 2 items | 2-3 hours | 2-3 days |
| Phase 2 | 5 sprints | 2-3 days | 1 week |
| Phase 3 | 5 sprints | 2-3 days | 1 week |
| Phase 4 | 5 sprints | 3-4 days | 1-2 weeks |
| **Total** | **17 sprints** | **~2 weeks (mock)** | **~4 weeks (mock + real)** |

---

## Decision: What to Build Next

**Recommended order** (maximize value with current constraints):

1. **Phase 2 mock** (SNPE adapter + Linux platform) — extends platform coverage
2. **Phase 3 mock** (Android platform + ADB) — completes all 3 platforms in mock
3. **Phase 4.3** (multi-model pipeline) — high-value feature, no SDK dependency
4. **Phase 4.4** (dashboard) — visualization, no SDK dependency
5. **Real adapters** (when SDK docs arrive) — swap mock → real per platform
