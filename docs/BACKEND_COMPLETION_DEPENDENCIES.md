# Real Compiler Backend — Dependencies to Close T1.1

> **Goal:** replace `b"QUAD_COMPILED_BINARY"` placeholder in
> `src/quad/compiler/pipeline.py:71` with real, executable target
> binaries (.bin / .dlc / .tflite). Today the placeholder branch is
> opt-in (`QUAD_PLACEHOLDER_BACKEND=1`), and the default raises
> `BackendNotImplementedError`. This doc enumerates every dependency
> that has to land before that error can become a real artifact.

The dependency list is grouped by category. For each item: **what
it is**, **why it's needed**, **where it has to land**, **effort
estimate**, and **what it depends on transitively**. Three viable
implementation paths are compared at the end — each has a different
dependency profile.

---

## Quick reference — what each path needs

| Path | Local SDK? | Network? | Effort | Notes |
|---|:-:|:-:|---:|---|
| **A. Shell out to `qairt-converter`** (reuse existing adapter) | yes | no | ~3-5 days | Forfeits IR optimization layer |
| **B. Real IR → ONNX → SDK** (preserves IR) | yes | no | ~1.5-3 weeks | Round-trip through ONNX |
| **C. AI Hub remote compilation** (no local SDK) | no | yes | ~3-5 days | Latency + auth dependency |

The recommendation is **all three** — they cover different user
contexts (offline dev, CI, cloud-only). Path A is the fastest and
should land first; B + C are parallel follow-ups.

---

## 1. Software / library dependencies

### 1.1 Qualcomm AI Runtime (QAIRT) SDK

**What:** The whole SDK (~1 GB), specifically:
- `qairt-converter` — ONNX/TF/TFLite/PyTorch → DLC
- `qairt-quantizer` — DLC → quantized DLC (INT8 / INT4)
- `qnn-context-binary-generator` — DLC → QNN context binary (.bin) ⚠ **not yet wrapped**
- `qnn-model-lib-generator` — DLC → libQnnModel.so for compose-from-.so flow
- `snpe-dlc-info` — introspect DLC inputs/outputs
- Backend libraries: `libQnnHtp.dll/.so`, `libQnnHtpVxxStub.so`, `libQnnHtpVxxSkel.so`

**Why:** All real binary generation paths route through one of these tools. Without the SDK, only AI Hub (Path C) works.

**Where:** Already auto-discovered by `src/quad/sdk_manager.py` and applied via `_find_tool()` in `qairt_adapter.py`. No code change needed for discovery; just need to **add the new tool wrappers** (1.2 below).

**Status:** ✅ Available — installed via `quad sdk install <archive>` (Phase A.1).

**Depends on:** Qualcomm developer account + EULA acceptance (process dependency 5.1).

---

### 1.2 New CLI builders (Python wrappers)

**What:** Two new dataclass-based CLI builders in `src/quad/sdk_tools/tool_specs.py`:

- `QnnContextBinaryGeneratorArgs` — wraps `qnn-context-binary-generator` with all its flags:
  - `--model` (path to libQnnModel.so OR DLC)
  - `--backend` (libQnnHtp.dll / libQnnCpu.dll / libQnnGpu.dll)
  - `--binary_file` (output path)
  - `--config_file` (HTP backend extensions JSON)
  - `--output_dir`
  - `--profiling_level`
  - Target-SoC-specific flags

- `QnnModelLibGeneratorArgs` — wraps `qnn-model-lib-generator`:
  - `--cpp` / `--bin` / `--lib_targets`
  - `--lib_name`
  - `--output_dir`

**Why:** The existing `QAIRTAdapter` only goes ONNX → DLC. The DLC → context binary step (the actual deliverable for QNN deployment) has no Python wrapper.

**Where:** `src/quad/sdk_tools/tool_specs.py` (new dataclasses, ~80 lines each); also `src/quad/adapters/qairt_adapter.py` (new `_compile_to_context_binary()` method).

**Effort:** 1 day (CLI flag enumeration is well-documented in QAIRT SDK docs; just paperwork).

**Depends on:** 1.1 (SDK install) + 4.1 (CLI flag documentation).

---

### 1.3 ONNX writer / serializer

**What:** Code that converts a `quad.compiler.ir.IRGraph` back into an `onnx.ModelProto` so it can be fed to `qairt-converter`.

Required pieces:
- `IRGraph.to_onnx(weights: dict[str, np.ndarray]) -> onnx.ModelProto`
- Op-by-op lowering of QUAD IR ops to ONNX ops (most are 1:1 — `Conv` ↔ `Conv`, etc.)
- Tensor-shape propagation through the graph
- Initializer (weight) embedding
- Opset version tagging

**Why:** Path B (real IR → backend) needs the IR to round-trip through ONNX since QAIRT's converter only consumes ONNX/TF/TFLite/PyTorch — not QUAD IR directly.

**Where:** `src/quad/compiler/frontend_onnx.py` already has `compile_onnx() : ONNX → IR`. We need the inverse `compile_to_onnx() : IR → ONNX` next to it.

**Effort:** 3-5 days.

**Depends on:**
- `onnx` Python package (already in `[real]` extras)
- 2.1 (weight tensor storage in IR — currently absent — see below)

**Skipped if:** Path A (shell out to `qairt-converter` directly with the source ONNX, bypass the IR layer for compilation).

---

### 1.4 ONNX validation toolchain

**What:** `onnx.checker.check_model()` to verify the round-tripped graph is well-formed before handing to QAIRT.

**Why:** A subtle IR-lowering bug can produce ONNX that QAIRT silently accepts but compiles into wrong code. Pre-validation catches this with a stack trace instead of mystery output.

**Where:** Insert into `pipeline.py` between IR-to-ONNX and qairt-converter.

**Effort:** 1 hour (just one function call + error formatting).

**Depends on:** 1.3.

---

### 1.5 ONNX Runtime as an alternative backend

**What:** `onnxruntime-qnn` with the QNN execution provider — already in `[real]` extras. Lets us validate a compiled binary by running inference on it before declaring success.

**Why:** Sanity-check the produced binary. If `qnn-context-binary-generator` writes a corrupt file, ORT-QNN errors immediately rather than us discovering it on the target device.

**Where:** New `src/quad/compiler/validate_binary.py` module. Called as the last step of the pipeline.

**Effort:** 2 days (numerical-equivalence harness against the source ONNX).

**Depends on:** `onnxruntime-qnn` install (already in extras), 1.1 (backend libs), 1.2 (binary generator).

---

## 2. Code-level dependencies (what we have to write)

### 2.1 Weight-tensor storage in IR (BLOCKER for Path B)

**What:** `IRGraph` currently stores op metadata but **no actual tensor data** for weights/biases. Every `IRNode.attributes` is a Python dict, and there's no `weights: dict[str, np.ndarray]` on the graph.

**Why:** A real compile needs the weight bytes. Without them, the round-tripped ONNX has empty initializers and the converter emits a network that runs on garbage parameters.

**Where:** Two changes required:
- `src/quad/compiler/ir.py`: add `IRGraph.initializers: dict[str, IRTensor]` plus `data: bytes | np.ndarray` to `IRTensor`
- `src/quad/compiler/frontend_onnx.py:_parse_real_onnx()`: copy initializer bytes into the new field

**Effort:** 2-3 days (changes the IR schema; need to update QIR JSON serializer + 4-5 tests + 1 doc).

**Risk:** Memory: a 200 MB ONNX model becomes a 200 MB IR object. Need to decide whether to lazy-load initializers from a file-backed store. Probably yes for production.

**Depends on:** Nothing (independent change). Other paths can land in parallel.

---

### 2.2 Per-target compilation orchestration

**What:** A function that, given an `IRGraph` + target capability, decides whether to:
- Pass through to `qairt-converter` (Path A — needs source ONNX path, not IR)
- Round-trip via ONNX (Path B — needs 1.3 + 2.1)
- Submit to AI Hub (Path C)
- Raise (no path available)

The decision is per-target because different targets need different binary formats (.bin for HTP, .dlc for CPU/GPU SNPE, .tflite for the TFLite delegate).

**Where:** New `src/quad/compiler/backend.py`:
```python
def compile_for_target(
    ir: IRGraph,
    capability: ComputeCapability,
    *,
    source_onnx: Path | None = None,   # if available, prefer Path A
    use_aihub: bool = False,             # opt-in for Path C
    sdk_root: Path | None = None,
) -> bytes:
    ...
```

**Why:** Centralises the per-path logic. `pipeline.py` becomes a one-line dispatch.

**Effort:** 2-3 days.

**Depends on:** 1.2 (CLI builders), 1.3 (IR → ONNX), 4.2 (capability → tool flag mapping).

---

### 2.3 Per-target backend extensions config

**What:** `qnn-context-binary-generator` requires a JSON config file describing HTP backend extensions (graph optimizer level, compute precision, performance mode, etc.). We need a config-builder that produces this from a `ComputeCapability`:

```python
def build_htp_backend_config(
    capability: ComputeCapability,
    optimization_level: int = 3,
    performance_mode: str = "burst",
) -> dict[str, Any]:
    """Produce qnn-context-binary-generator --config_file content."""
```

**Where:** New `src/quad/compiler/htp_config.py`. Or extend the existing `src/quad/profiler/qhas.py:QHASConfig` (already a JSON-config builder).

**Effort:** 1 day.

**Depends on:** 4.3 (HTP backend extensions schema documentation).

---

### 2.4 Subprocess + asyncio plumbing

**What:** A reusable async-subprocess helper for the new tools. The `_run_command()` helper in `qairt_adapter.py:58-76` already does this; we should extract it to `src/quad/adapters/_subprocess.py` so both backend.py and qairt_adapter.py share it.

**Why:** One place to handle timeout/encoding/stderr-capture/log-redaction.

**Effort:** 0.5 day (refactor).

**Depends on:** Nothing.

---

### 2.5 Caching layer

**What:** A content-addressed cache that maps `(source-hash, target, options) → compiled-binary-path`. Compilation is slow (10-60 s); rerunning the same compile across MCP tool calls without caching is painful.

**Where:** New `src/quad/compiler/cache.py`. Uses `~/.quad/cache/` for storage (gitignored, like `.quad/sdk.json`).

**Effort:** 1-2 days.

**Depends on:** Nothing.

---

### 2.6 IR optimization passes (lazy)

**What:** Phase 0 of the pipeline currently has no optimization pass between frontend and backend. To produce competitive binaries, we'd want at least:
- Constant folding
- Dead code elimination
- Common subexpression elimination
- Layout propagation (NCHW ↔ NHWC)

**Why:** Optional for first cut — `qairt-converter` does its own optimisation. But high-quality compilation will need them eventually.

**Effort:** **Not in scope for closing T1.1.** Each pass is 2-5 days. Total ~3 weeks for a minimal optimizer, and the existing `src/quad/optimizer/` directory has scaffolds.

**Depends on:** 2.1 (weights in IR).

---

## 3. Knowledge / documentation dependencies

### 3.1 `qnn-context-binary-generator` CLI reference

**What:** Full flag list, output format, error semantics.

**Status:** ⚠ **Not currently documented in `src/quad/sdk_tools/`.** The 18-tool matrix covers every other tool but skips this one.

**Where to find it:** QAIRT SDK installation includes `${QAIRT_SDK_ROOT}/docs/QNN/general/context_binary_generator.html` and CLI `--help`.

**Effort:** 0.5 day to extract + integrate as a structured Python module (matching the existing `tool_specs.py` style).

**Risk:** SDK version drift — flag set has changed between 2.43 and 2.45. Need version-conditional builders.

---

### 3.2 HTP backend extensions JSON schema

**What:** The full schema for the `--config_file` JSON that the binary generator consumes. Includes:
- Compute precision (FP16 / INT8 / mixed)
- Graph optimizer level (1-3)
- Performance mode (burst / sustained / power_saver)
- Custom op registration (if UDOs are present)
- Soc-specific tuning params

**Where to find it:** QAIRT docs `QnnHtpBackendExtensions.html`. Some fields aren't fully documented and require Qualcomm support.

**Effort:** 1 day initial + ongoing as new fields are exposed.

---

### 3.3 ONNX → QUAD IR op coverage gaps

**What:** The op_coverage table (`src/quad/compiler/op_coverage.py`) lists what ONNX ops the backends support. We don't yet have:
- QUAD IR → ONNX op-name mapping for round-trip (Path B)
- Op-attribute lowering rules (e.g. ONNX `Conv` `kernel_shape` vs QUAD IR's representation)

**Effort:** 1-2 days to enumerate the 130+ HTP ops with their ONNX-equivalent attribute schemas.

---

### 3.4 Quantization parameters for round-trip

**What:** When INT8/INT4 quantization happens, the activation/weight scales need to round-trip through the ONNX intermediate. ONNX has `QuantizeLinear` / `DequantizeLinear` ops; AIMET produces `model.encodings` JSON; qairt-quantizer needs `--quantization_overrides`. These three formats need a converter.

**Effort:** 2-3 days.

**Depends on:** AIMET integration (Phase C delivered the scaffold; real backend pending).

---

## 4. CLI / API contract dependencies

### 4.1 Stable `qairt-converter` stdout/stderr format

**What:** Right now `_parse_latency()` and `_parse_layers()` in `qairt_adapter.py` use regex against the stdout of these tools. Same fragility applies to `qnn-context-binary-generator`. We need:
- Confirmation of the stable format across SDK versions, OR
- A `--json-output` / structured-output flag (some tools have it, not all)

**Risk:** If we ship regex parsers and the SDK output format changes in 2.46+, the backend silently produces wrong reports. Mitigation: version-pin the SDK + add a regression test that loads a real artifact.

---

### 4.2 Capability → tool-flag mapping table

**What:** Each `ComputeCapability` (qnpu_v3, hexagon_v66, hexagon_v75, hexagon_v79, qdsp_v66, etc.) maps to:
- A specific `--soc_model` flag for `qairt-converter`
- A specific backend library (libQnnHtpV68.so vs V73 vs V75 etc.)
- Specific HTP extensions config defaults

**Where:** `src/quad/compiler/capabilities.py` already has the capability list; we need to extend each entry with these tool-flag fields.

**Effort:** 1 day.

---

### 4.3 Op-coverage updates per SDK version

**What:** The HTP supported-ops set (`HTP_SUPPORTED_OPS` in `op_coverage.py`) is hand-maintained. New SDK versions add ops; we need a script that diffs the SDK's actual op support against our table and warns on drift.

**Effort:** 1 day.

---

## 5. Process / governance dependencies

### 5.1 Qualcomm developer account + EULA acceptance

**What:** The QAIRT SDK requires login + per-version EULA acceptance to download. There's no anonymous direct URL.

**Implications:**
- Local dev: each developer accepts once
- CI: needs a pre-stored token (the `QAIRT_DOWNLOAD_URL` + `QAIRT_DOWNLOAD_TOKEN` env vars in `setup_sdk.sh`)
- Tests that exercise the real backend can't run on stock GitHub Actions runners — need self-hosted Snapdragon runners OR mock-only

---

### 5.2 Test hardware

**What:** A Snapdragon X Elite Copilot+ PC for backend integration testing. CI on x86_64 Linux can't validate the produced HTP binary actually runs.

**Available now:** The Dell Latitude 7455 we used for the sample-app run is suitable.

**Implications:** Backend tests should be split:
- `unit/test_compiler/test_backend_unit.py` — runs everywhere (mocks subprocess)
- `integration/test_compiler/test_backend_real.py` — gated on `QAIRT_SDK_ROOT` + Snapdragon hardware (`pytest.mark.requires_sdk + requires_npu`)

---

### 5.3 Test models corpus

**What:** A small set of reference ONNX models for backend validation:
- MobileNetV2-1.0 (already downloaded for the sample app)
- ResNet50
- A transformer (e.g. distilbert) — exercises non-conv ops
- A model with at least one unsupported op (UDO test)

**Where:** Should live outside the repo (5-50 MB each). Use a CDN or scripted download from ONNX model zoo.

**Effort:** 1 day to set up the test fixture system.

---

### 5.4 Calibration datasets

**What:** Real INT8/INT4 paths need calibration data, not random noise. For each test model:
- 100-500 representative inputs
- Stored as `.npy` or `.raw`
- Bundled in a calibration archive (or re-downloaded for tests)

**Effort:** 2-3 days to assemble + version a calibration corpus.

**Depends on:** 5.3 (test models).

---

### 5.5 Numerical-equivalence acceptance threshold

**What:** Compiled binary outputs won't be bit-identical to the source ONNX (different math libraries, rounding, fused kernels). We need a policy:
- FP32 / FP16 paths: max abs error < 1e-3
- INT8: top-1 accuracy drop < 1.0% on classification
- INT4: top-1 accuracy drop < 3.0%

**Effort:** 1 day to bake into the test framework once thresholds are agreed.

---

## 6. Current QUAD plumbing — what we can reuse

This is the **good news** part: a lot of plumbing already exists from earlier phases.

| Component | Status | Reuse for backend? |
|---|---|---|
| `quad.adapters.qairt_adapter.QAIRTAdapter.convert_model` | ✅ Real (Phase A.4 + A.6) | **Yes — Path A's core** |
| `quad.adapters.aihub_adapter.AIHubAdapter.compile_for_device` | ✅ Real (Phase C) | **Yes — Path C's core** |
| `quad.adapters.aimet_adapter.AIMETAdapter` | 🟡 Mock + scaffold (Phase C) | Partial — full real backend pending |
| `quad.compiler.frontend_onnx.compile_onnx` | ✅ Real (uses `onnx.load`) | **Yes — frontend stays put** |
| `quad.compiler.op_coverage.compute_coverage` | ✅ Real (Phase E) | **Yes — pre-flight check** |
| `quad.compiler.capabilities` | ✅ Real | Yes (needs extending — see 4.2) |
| `quad.compiler.qbin.QBin` | ✅ Real (already a fat-binary container) | Yes |
| `quad.adapters._run_command` async helper | ✅ Real | Yes (refactor to shared module — 2.4) |
| `quad.adapters.model_inputs.create_input_list_for_model` | ✅ Real (Phase A.4) | Yes — for backend numerical-equivalence tests |
| `quad.sdk_manager` discovery | ✅ Real (Phase 0) | Yes — gates whether Path A/B are available |

---

## 7. Three implementation paths, ranked

### Path A — Shell out to `qairt-converter` (recommended first)

**The idea:** When `compile_model()` is given an ONNX path, skip the IR layer for compilation and re-invoke the existing `QAIRTAdapter.convert_model`. Then add the missing DLC → context binary step.

**New code:**
- `1.2` — `QnnContextBinaryGeneratorArgs` builder
- `2.4` — extracted subprocess helper
- New `pipeline.py` branch: when source is ONNX and a real backend is requested, route through QAIRTAdapter instead of placeholder.

**Skips:**
- `1.3` (no IR → ONNX needed; we have the source ONNX)
- `2.1` (no weight storage in IR needed)
- `2.6` (use qairt-converter's optimizer, not ours)

**Total effort: 3-5 days.**

**Caveat:** This is the *pragmatic* path. The IR layer becomes vestigial for compilation — it stays useful for analysis (op-coverage, layer counting) but isn't on the compile critical path.

### Path B — Real IR → ONNX → SDK (preserves IR)

**The idea:** `IRGraph` round-trips through ONNX. The same backend tool then handles it.

**New code:**
- `1.2`, `1.3`, `1.4`, `2.1`, `2.2`, `2.3`, `2.4` — full stack
- Optionally `1.5` (binary validation) and `2.6` (real optimizer)

**Skips:**
- Nothing — this is the comprehensive path.

**Total effort: 1.5-3 weeks (full stack).**

**When it matters:** When QUAD wants to do IR-level optimisations that ONNX can't represent (e.g. fusion patterns specific to Hexagon), and pass them straight to the backend without going through ONNX serialisation again.

### Path C — AI Hub remote compile

**The idea:** Skip local compilation entirely. Submit ONNX to AI Hub, get back a context binary.

**New code:**
- Already implemented in `aihub_adapter.compile_for_device` (Phase C).
- New `pipeline.py` branch: when `use_aihub=True` or `QUAD_AIHUB_BACKEND=qai_hub`, route through AI Hub.

**Skips:**
- Almost everything in 1.x, 2.x, 4.x — AI Hub does the work
- Still needs: 5.5 (acceptance thresholds for cloud-vs-local equivalence)

**Total effort: 3-5 days.**

**Caveats:**
- Network latency (10-60s per compile)
- Requires `QAI_HUB_API_KEY` (process dependency 5.1's analogue)
- Can't compile for unreleased SoCs / pre-release HTP versions

---

## 8. Sequencing — the critical path

Three milestones, each independently shippable:

### Milestone 1 — Path A working end-to-end (3-5 days)

```
1. Add QnnContextBinaryGeneratorArgs builder (1 day)
2. Extract _run_command helper to shared module (0.5 day)
3. Wire pipeline.py to call QAIRTAdapter for ONNX inputs (1 day)
4. Add the DLC -> context-binary tail step (0.5 day)
5. Unit tests with mocked subprocess (1 day)
6. Integration test on the Dell Latitude 7455 (1 day)
```

After M1: `compile_model('mobilenetv2.onnx')` produces a real
`.bin` that runs on the Hexagon HTP.

### Milestone 2 — Path C in parallel with M1 (3 days, parallelizable)

```
1. Add AI Hub backend branch in pipeline.py (1 day)
2. CLI flag QUAD_AIHUB_BACKEND=qai_hub (0.5 day)
3. Numerical-equivalence test (cloud-compiled vs source ONNX) (1.5 days)
```

### Milestone 3 — Path B (1.5-3 weeks, after M1)

```
Week 1: weight storage in IR (2.1) + IR → ONNX (1.3)
Week 2: backend orchestration (2.2) + HTP config (2.3) + cache (2.5)
Week 3: validation harness (1.5) + op coverage drift (4.3)
```

After M3: QUAD's IR is a first-class compilation target, not just an analysis representation.

---

## 9. Risks + unknowns

1. **SDK output format drift.** Mitigation: version-pin in `pyproject.toml`'s `[real]` extras + regression test against a vendored stdout fixture.

2. **HTP signature errors on Windows** (`transportStatus: 9 / 0x80000406`). Already handled in `qairt_adapter.py`, but the binary-generator path may surface it differently.

3. **Memory bloat in IR with weights** (2.1). May force a lazy/file-backed weight store. Decide before starting Milestone 3.

4. **Per-SoC backend libraries.** The right `libQnnHtpVxx.dll` depends on the device's HTP version (v66/v68/v73/v75/v79). The discovery → capability → backend-lib chain isn't fully wired.

5. **AI Hub API quotas + outage.** Path C must gracefully fall back to Path A on outage. CI shouldn't depend on AI Hub uptime.

6. **License / EULA boundaries.** The user must accept the QAIRT EULA before compilation. The installer covers this; backend tests need to verify the SDK install before invoking.

---

## 10. Summary checklist

Do these to close T1.1 with **Path A only** (the fast path):

- [ ] **1.2** Add `QnnContextBinaryGeneratorArgs` to `sdk_tools/tool_specs.py`
- [ ] **1.4** Add `onnx.checker.check_model` validation gate (1 hr)
- [ ] **2.4** Extract `_run_command` to shared module
- [ ] New `compile_for_target_via_qairt()` in `compiler/backend.py`
- [ ] Wire into `pipeline.py` — ONNX + real-backend → call new function
- [ ] **3.1** Document the binary-generator CLI in a structured module
- [ ] Unit tests with mocked subprocess (extend `test_compiler/test_compiler.py`)
- [ ] Integration test on the test laptop with real SDK installed
- [ ] **5.5** Write the numerical-equivalence acceptance threshold

Do these to also close it with **Path B** (preserves IR):

- [ ] **2.1** Add weight tensor storage to IR
- [ ] **1.3** Implement `compile_to_onnx(IRGraph) -> onnx.ModelProto`
- [ ] **2.2** Build per-target compilation orchestrator
- [ ] **2.3** HTP backend extensions config builder
- [ ] **1.5** Binary validation harness via onnxruntime-qnn
- [ ] **5.3 + 5.4** Test models + calibration corpus
- [ ] **3.4** Quantization params round-trip

Do these for **Path C** (AI Hub):

- [ ] AI Hub backend branch in `pipeline.py` (~30 lines)
- [ ] `compile_via_aihub()` integration test (depends on `QAI_HUB_API_KEY` in CI secrets)
- [ ] Equivalence test against local-SDK output

---

## Effort grand total

| Path | Best case | Worst case |
|---|---:|---:|
| A only | 3 days | 5 days |
| C only | 3 days | 5 days |
| A + C | 4 days | 7 days |
| A + C + B | 1.5 weeks | 3.5 weeks |

The pragmatic recommendation: **A + C in 1 week**, then B in a
follow-up sprint. That gets users a working compiler end-to-end with
both offline and cloud paths, while preserving the option to
upgrade to a fully-IR-driven flow later.
