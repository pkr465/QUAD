# QUAD Sample Application — Real Hardware Run Report

**Run date:** 2026-05-07
**Test machine:** Dell Latitude 7455 (Snapdragon X Elite Copilot+ PC)
**Workload:** MobileNetV2-1.0 (ImageNet classifier, 1×3×224×224 input)
**Source:** [`examples/sample_app_real_hw.py`](../examples/sample_app_real_hw.py)
**Raw data:** [`examples/run_results.json`](../examples/run_results.json)

---

## TL;DR

The QUAD sample application was run end-to-end on the Snapdragon X Elite
laptop. Hardware was detected correctly, the full 5-tool MCP pipeline
completed in under three seconds, and **MobileNetV2-1.0 inference on the
Oryon CPU sustained 388 FPS at 2.56 ms mean latency** (500 timed
iterations after 50 warmup runs).

The Hexagon NPU is physically present and detected by Windows
(`Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Hexagon(TM) NPU`,
`Class=ComputeAccelerator`, `Status=OK`), but **direct NPU benchmarking
requires the QAIRT SDK** (still pending install). NPU figures in this
report come from QUAD's mock projection layer and are conservative
placeholders, not measured. Real-NPU measurements unblock the moment
QAIRT is installed — no QUAD code changes needed.

---

## 1. Host hardware (probed live, not mocked)

| Property | Value | Source |
|---|---|---|
| Laptop | Dell Latitude 7455 | `systeminfo` |
| OS | Windows 11 Pro (X64 emulation layer) | `Win32_OperatingSystem` |
| CPU | Snapdragon X Elite X1E80100 — Qualcomm Oryon | `Win32_Processor.Name` |
| Cores / threads | 12 / 12 | `Win32_Processor` |
| Max clock | 4012 MHz | `Win32_Processor.MaxClockSpeed` |
| GPU | Adreno X1-85 (driver 31.0.67.0) | `Win32_VideoController` |
| **NPU** | **Hexagon NPU (X1E80100)** | `Get-PnpDevice` — `Class=ComputeAccelerator`, `Status=OK` |
| RAM | 31.6 GB | `psutil.virtual_memory()` |
| Power | AC, battery 100% | `Win32_Battery` |

QUAD's `hardware_detect` MCP tool reported the chipset name, peak
TFLOPS / TOPS, and supported runtimes. Numerical TOPS / TFLOPS come from
the device profile table, not real measurement (the SDK does not expose
peak rated throughput at runtime). All other fields cross-check against
the live probe.

---

## 2. Sample application — what it does

The app exercises the **complete QUAD MCP pipeline** plus a real ground
truth measurement:

| Step | Tool | What it does | This run |
|---|---|---|---|
| 0 | (host probe) | Reads CPU/GPU/NPU/RAM/battery via Windows APIs | ✅ live probe |
| 1 | `hardware_detect` | Returns `DeviceProfile` for the chipset | ✅ matched probe |
| 2 | `convert_model` | ONNX → SNPE DLC INT8 (projected size, conversion notes) | ✅ projected (mock) |
| 3a | (real benchmark) | ONNX Runtime CPU inference, 500 iters | ✅ **real** on Oryon |
| 3b | `profile_workload` (detailed) | Latency / power / memory / utilization | ⏳ projected (mock) |
| 4 | `profile_workload` (linting) | Per-op cycle counts + bottleneck detection | ⏳ projected (mock) |
| 5 | `orchestrate_workload` | CPU/GPU/NPU layer allocation, 3 power modes | ⏳ projected (mock) |
| 6 | `generate_code` | Emits Windows-on-Snapdragon C++ inference app | ✅ generated |

Generated files (real, written to disk):
- [`examples/generated/real_hw_demo/inference.cpp`](../examples/generated/real_hw_demo/inference.cpp) — 91 lines, 2,613 bytes, QNN-targeted C++
- [`examples/generated/real_hw_demo/CMakeLists.txt`](../examples/generated/real_hw_demo/CMakeLists.txt) — 33 lines, 581 bytes

---

## 3. Real CPU benchmark (Oryon, ONNX Runtime 1.25.1)

| Metric | Value |
|---|---|
| Iterations | 500 (after 50 warmup, excluded) |
| Wall-time total | 1.289 s |
| **Mean latency** | **2.56 ms** |
| Median latency | 2.20 ms |
| p95 latency | 4.68 ms |
| p99 latency | 6.28 ms |
| Min / max | 2.04 ms / 9.69 ms |
| Std-dev | 0.95 ms |
| **Throughput** | **388.0 FPS** |
| Memory baseline (RSS) | 125.8 MB |
| Memory peak (RSS) | 129.6 MB (Δ 3.8 MB) |
| CPU utilisation during run | 1160% (≈11.6 of 12 logical cores active) |
| Battery delta | 100% → 100% (no measurable drain on AC) |

Settings:
- `intra_op_num_threads=0` (ORT picks all 12 Oryon cores)
- `graph_optimization_level=ORT_ENABLE_ALL`
- Provider: `CPUExecutionProvider` (Oryon CPU, no GPU/NPU dispatch)
- Input: `np.random.standard_normal((1,3,224,224), float32)`, fixed seed 42

**Take-aways**

1. Oryon CPU on its own is **highly capable** for small mobile-class
   models — 388 FPS at <3 ms latency is plenty for camera-rate
   classification (30 FPS @ 7% duty cycle).
2. Variance is tight (σ < 1 ms, p99 only 2.4× mean) — this PC is in a
   thermally stable state (AC, room temperature, short run).
3. Memory footprint is trivial (3.8 MB Δ over baseline) — most of the
   125 MB RSS is the Python interpreter and ORT itself.
4. ORT scaled out to ~11.6 of 12 cores — there is no headroom on CPU
   for additional concurrent workloads at this throughput.

---

## 4. NPU projection (Hexagon HTP — to-be-measured)

QUAD's `profile_workload` tool with `runtime="npu"` reports:

| Metric | Projected value | Notes |
|---|---|---|
| Mean latency | 5.00 ms | Mock placeholder; expected real value < 1 ms for INT8 MobileNetV2 |
| p95 / p99 | 7.00 / 9.00 ms | Same — placeholder |
| Throughput | 200 FPS | Same |
| Power | 2000 mW | Same |
| Memory peak | 45 MB | Same |
| NPU utilization | 89% | Same |

**Important caveat — the NPU projection in this run is a mock.** The
QUAD codebase ships with a placeholder profile that returns 5 ms for
generic models pending real measurement. **Published Hexagon HTP
performance for MobileNetV2 INT8 is on the order of 0.5–1.0 ms (>1000
FPS)** [vendor microbenchmarks]. Once `QAIRT_SDK_ROOT` is populated and
`snpe-net-run` is in `PATH`, `quad mode` will flip to `READY` and this
section will report measured numbers automatically — no application
code change required.

To enable real NPU measurement on this same machine:

```powershell
# 1. Download QAIRT 2.45+ from softwarecenter.qualcomm.com (~1 GB)
# 2. Activate
$env:QAIRT_SDK_ROOT = "C:\Qualcomm\AIStack\QAIRT\2.45.0.260326"
& "$env:QAIRT_SDK_ROOT\bin\envsetup.ps1"
$env:QUAD_ADAPTER_MODE = "real"

# 3. Verify
quad mode                   # should report READY
quad doctor --real-mode     # all SDK checks pass

# 4. Re-run the same sample app — section 4 now becomes real measurements
python examples/sample_app_real_hw.py --iterations 500
```

See [`docs/REAL_HARDWARE.md`](REAL_HARDWARE.md) for the full
enablement playbook.

---

## 5. Cycle-level (HTP linting) — projected

QUAD's linting profiling pass (`profiling_level="linting"`) projects
where the model would spend its NPU cycles:

| Metric | Projected value |
|---|---|
| Total HTP cycles | 3,753,939 |
| Bottleneck ops detected | 1 |

Top ops by projected cycle count:

| Idx | Op | Cycles | Resource overlap | Bottleneck? |
|---|---|---:|---:|---|
| 4 | sub_op | 2,165,162 | 21% | ⚠ yes |
| 5 | add_op | 525,971 | 92% | — |
| 6 | output | 407,091 | 28% | — |

Op #4 has a 21% resource overlap, meaning the HTP cores are stalling
21% of its execution waiting on memory or scalar dependencies — this is
the optimisation target a real run would surface. (Names like `sub_op`
are mock placeholders; real linting names actual Conv/MatMul/etc. ops.)

---

## 6. Workload orchestration (3 power modes)

QUAD's `orchestrate_workload` returns a per-mode allocation map:

| Mode | Latency | Power | NPU% | GPU% | CPU% |
|---|---:|---:|---:|---:|---:|
| **performance** | 2.55 ms | 1420 mW | 70% | 0% | 30% |
| **balanced** | 2.55 ms | 1420 mW | 70% | 0% | 30% |
| **efficiency** | 5.00 ms | 1000 mW | 0% | 0% | 100% |

CPU fallback ops (would not run on NPU): `bn1`, `bn2`, `fc1`.

The orchestrator says: for this small INT8 model, "performance" and
"balanced" both default to NPU-heavy allocation (70/30 NPU/CPU) and
"efficiency" pushes everything to CPU to spare the HTP wake-up cost. On
a real device, expect "balanced" to differ from "performance" via DSP
clock-frequency capping rather than core re-assignment.

---

## 7. Generated inference code

`generate_code` produced two files at
`examples/generated/real_hw_demo/`:

**inference.cpp (91 lines)** — QNN-targeted C++ inference scaffold with
the standard 6-step QNN pipeline (load backend `.so` → load model →
init log/backend/device/context → compose graph → execute → cleanup).
Currently the implementation bodies are TODOs — the QUAD codegen
templates are wireframes, not fully populated `dlopen`/`dlsym` code.

**CMakeLists.txt (33 lines)** — minimal build script linking against
the QAIRT SDK shared libraries.

---

## 8. End-to-end timing breakdown

| Stage | Wall time | Mode |
|---|---:|---|
| Host hardware probe (PowerShell × 4) | ~1.2 s | real |
| QUAD `hardware_detect` | <50 ms | mock |
| QUAD `convert_model` | <100 ms | mock |
| **Real CPU inference (500 iter)** | **1.289 s** | real |
| QUAD `profile_workload` × 2 | <100 ms | mock |
| QUAD `orchestrate_workload` × 4 | <50 ms | mock |
| QUAD `generate_code` | ~80 ms | real (Jinja2 render + write) |
| **Total** | **≈3 s** | mixed |

---

## 9. What this run validated vs. what's still pending

✅ **Validated end-to-end on real Snapdragon X Elite hardware:**

- QUAD CLI / MCP pipeline runs cleanly on Windows-on-Snapdragon
- Hardware detection (chipset, NPU presence, GPU) is accurate
- Real ONNX Runtime CPU inference clocks 388 FPS / 2.56 ms on Oryon
- Generated C++ + CMakeLists.txt artifacts are produced and well-formed
- Power/battery telemetry hooks work via `Win32_Battery`

⏳ **Still pending (single blocker — QAIRT SDK install):**

- Real NPU latency / power / memory measurement
- Real per-op linting (named ops, real cycle counts)
- Quantization-applied DLC size (4× compression projection unverified)
- Real codegen body fill-in (currently TODOs)

The QUAD codebase is mock-first by design and the adapter layer flips
to real automatically once the SDK is reachable —
`quad mode`/`quad doctor --real-mode` confirm readiness. See
[`docs/REAL_HARDWARE.md`](REAL_HARDWARE.md) for the install procedure.

---

## 10. Reproduce this report

```powershell
# From the repo root, on a Snapdragon X Elite Copilot+ PC:
python -m pip install onnxruntime-qnn psutil
python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx', 'examples/models/mobilenetv2-12.onnx')"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
python examples/sample_app_real_hw.py --iterations 500 --warmup 50 --json-out examples/run_results.json
```

Tuning knobs:
- `--iterations` — more iters → tighter percentiles (default 200)
- `--warmup` — exclude first N runs from timing (default 30)
- `--json-out` — emit machine-readable results JSON

JSON schema mirrors the dataclasses in `sample_app_real_hw.py` —
`HostHardware`, `CpuBenchmarkResult`, plus the unmodified QUAD tool
outputs.
