# QUAD — End-to-End Gap Analysis

> Conducted 2026-05-07 against `main` at commit `5ea9253`.
> Scope: every layer of the QUAD stack from MCP request through SDK
> invocation, codegen, runtime, server, and packaging — what works,
> what's stubbed, what's missing entirely.

The headline number to keep in mind:

- **Mock-mode platform**: ~95% complete. All 5 MCP tools, the CLI, the
  installer, and the SDK manager are well-built and well-tested
  (1811/1819 passing).
- **Real-mode platform end-to-end**: ~30% complete. The plumbing
  (adapter dispatch, CLI invocation, output paths) is real, but
  several load-bearing pieces are placeholders.

This document inventories the gaps so we can plan closure work.

---

## Tier 1 — Blockers (real mode cannot work without these)

### 1.1 Compiler pipeline returns placeholder bytes

**File:** `src/quad/compiler/pipeline.py:66-72`

```python
qbin.add_target(
    target=cap.name,
    format="qnn" if "npu_v3" in cap.name else "snpe",
    data=b"QUAD_COMPILED_BINARY",  # Placeholder
)
```

`compile_model()` is presented in the CLI (`quad compile`) and the
architecture diagram as a real ONNX → `.qbin` pipeline. It currently:
- Builds a mock `IRGraph` (no real ONNX op decomposition)
- Writes literal bytes `b"QUAD_COMPILED_BINARY"` as the target binary
- Does not call any SDK backend

**Closure:** Wire `compile_onnx()` to actually parse the ONNX, lower to
QIR with real ops, and call `QnnContext_compose()` / QAIRT compiler API
for each target capability.

### 1.2 Inference server has no HTTP binding

**File:** `src/quad/serve/server.py:335-340` and `:220-232`

`ModelServer.start()` flips a `self._running = True` flag and returns —
no Flask/FastAPI/uvicorn/grpc binding. `infer()` produces synthetic
`np.random.randn(batch_size, 1000)` instead of calling the runtime.
`quad serve <model>` does nothing useful at runtime.

**Closure:** Pick a server framework (FastAPI is documented in
`pyproject.toml`-adjacent material), bind the load/predict endpoints,
delegate inference to `runtime.Model.infer()` once that's also real
(see 1.3).

### 1.3 Runtime is entirely numpy-backed

**Files:** `src/quad/runtime/device.py`, `tensor.py`, `model.py`,
`memory.py`, `power.py`, `stream.py`

Every class in `quad.runtime` is a numpy mock:
- `Device` reads from a hardcoded chipset profile dict (lines 11-35)
  rather than detecting actual hardware
- `Tensor` wraps a numpy array; no DMA, no device memory
- `Model.infer()` returns `np.random` outputs
- `PowerMonitor` synthesises power data from a per-device-type lookup

This means **everything below the adapter layer that imports
`quad.runtime` is also mock**. The `serve` module, the kernels
infrastructure (`quad.kernels.graph.Graph`), and any QBin-execution
path are all on top of these mocks.

**Closure:** Either (a) build real backends for `Device` / `Tensor` /
`Model` that call into QNN's C API via ctypes, or (b) bypass them in
real mode by routing through `QAIRTAdapter.execute_inference()` which
shells out to `snpe-net-run` (and is itself a stub — see 1.4).

### 1.4 `QAIRTAdapter.execute_inference` is a thin stub

**File:** `src/quad/adapters/qairt_adapter.py:520-540`

```python
async def execute_inference(self, model_path: str, input_data: Any) -> Any:
    tool = _find_tool("snpe-net-run")
    input_list = self._create_dummy_input_list(Path(model_path))   # !!
    cmd = [tool, "--container", model_path, "--input_list", input_list, "--use_dsp", "--perf_profile", "burst"]
    result = await _run_command(cmd, timeout=60)
    return {"status": ..., "stdout": result.stdout[:500], ...}
```

The method:
- Ignores `input_data` and uses `_create_dummy_input_list()` (line 542)
  which generates `np.random.randn(1, 3, 224, 224)` — wrong for any
  real model with non-`(1,3,224,224)` inputs and useless for output
  validation
- Returns truncated stdout (500 chars), not actual inference outputs

**Closure:** Marshal `input_data` into a real `.raw` file in the right
shape/dtype derived from the model metadata; parse `snpe-net-run`'s
output `.raw` files back into numpy arrays.

### 1.5 No AIMET adapter — quantization claims are hollow

**File:** missing — there is no `src/quad/adapters/aimet_adapter.py`.

The README, MCP tool docstrings, and `convert_model` schema all
support `quantization="int8" | "int4"`. In `qairt_adapter.py:195-216`
the `int8`/`int4` branch calls `qairt-quantizer` directly with a
**dummy random input list**, which means:
- Calibration is meaningless (random noise has no resemblance to real
  model inputs)
- INT4 quantization paths are entirely absent (no AIMET → no INT4)

**Closure:** Create `aimet_adapter.py` with:
- Real calibration dataset loader (CSV / HDF5 / image directory)
- AIMET PTQ workflow (`aimet_torch.quantsim` or `aimet_onnx`)
- AIMET QAT hooks for users with training pipelines
- INT4 path via `aimet_torch.quantsim.QuantizationSimModel`

### 1.6 No AI Hub adapter — cloud profiling claims are hollow

**File:** missing — there is no `src/quad/adapters/aihub_adapter.py`.

`QAI_HUB_API_KEY` is documented in `.env.example` and shows up in
`src/quad/cli/configure.py:204` as a config string, but the `qai_hub`
Python SDK is never imported anywhere. There is no client, no
`profile_on_device()`, no remote benchmarking.

**Closure:** Create `aihub_adapter.py` with `qai_hub` SDK
integration; expose `profile_on_device()` and a `compile_for_device()`
helper.

### 1.7 Templates not packaged — `pip install` from sdist will break

**File:** `pyproject.toml`

There is no `[tool.setuptools.package-data]` (or `hatch`/`pdm`
equivalent) entry that includes `templates/**/*.j2`. The 43 Jinja2
templates live at the repo root in `templates/` and are loaded at
runtime by `CodegenEngine` via `FileSystemLoader(template_dir)`. A
sdist build with the current config **will not include them**, so
`generate_code` would raise `TemplateNotFound` at runtime in any
end-user install.

**Closure:** Either move `templates/` under `src/quad/` and declare
`include-package-data = true`, or add an explicit
`[tool.setuptools.package-data]` (or `[tool.hatch.build.targets.wheel]
include = ["templates/**"]`) declaration.

### 1.8 Generated C++ inference scaffolds are not compilable

**Files:** `templates/qnn/inference_so.cpp.j2`,
`templates/qnn/inference_bin.cpp.j2`,
`templates/qnn/inference_tflite_delegate.cpp.j2`,
`templates/windows/cpp/inference.cpp.j2`,
`templates/android/jni/inference_jni.cpp.j2`

Validated empirically by running `examples/sample_app_real_hw.py` on
the test laptop: `inference.cpp` (91 lines) generated four
function-body TODOs (`initialize`, `load_model`, graph execute,
cleanup). The C++ validator (`codegen/validators.py:55-65`) only
checks bracket balance, so it accepts non-compilable output.

**Closure:** Fill in the QNN init / `dlopen` / `QnnInterface_getProviders`
/ `QnnGraph_execute` / `QnnContext_freeGraphs` calls in each template.
Strengthen the C++ validator to at least require declared identifiers
to be referenced (e.g. attempt a `gcc -fsyntax-only` if the toolchain
is on PATH).

---

## Tier 2 — High (real mode produces wrong / incomplete output)

### 2.1 Latency / per-layer parsing is regex-fragile and stubbed

**File:** `src/quad/adapters/qairt_adapter.py:560-593`

`_parse_latency` falls back to `5.0` ms when none of the three regex
patterns match (line 574). `_parse_layers` returns a single composite
layer when its regex doesn't match (lines 590-593). Any change in
`snpe-net-run` output format silently produces these defaults — the
caller has no signal that parsing failed.

**Closure:** Add a strict mode that raises `RealAdapterUnavailableError`
on parse failure. Better long term: prefer JSON output flags from the
SDK if/when they ship.

### 2.2 `QAIRTAdapter.detect_hardware` is a hardcoded fallback

**File:** `src/quad/adapters/qairt_adapter.py:98-128`

The method runs `qnn-platform-validator --help` (line 104) and then
always falls back to a hardcoded `linux` profile (lines 110-128). The
Windows and Android branches are not implemented. On the Snapdragon X
Elite test machine, this returns the linux fallback rather than the
machine's actual specs.

**Closure:** Implement Windows and Android branches; parse the actual
`qnn-platform-validator` output (not just `--help`); cross-check
against `Win32_Processor` / `getprop` / `/proc/cpuinfo` as we do in
`examples/sample_app_real_hw.py`.

### 2.3 `orchestrate_workload` output is unused by codegen

**Files:** `src/quad/tools/orchestrate_workload.py` (output) →
`src/quad/codegen/engine.py` (consumer)

The orchestrator returns a layer→runtime allocation map, but `CodegenEngine.render()` never reads it — every template emits single-target code (NPU OR CPU OR GPU, not heterogeneous). The orchestrate phase is documented but its result is dead-end metadata.

**Closure:** Pass `allocation_map` into the codegen request; emit
heterogeneous dispatch templates that load the model on multiple
backends and route ops accordingly.

### 2.4 `orchestrate_workload` crashes on linting profiles

**File:** `src/quad/tools/orchestrate_workload.py:34`

When called with `profiling_level="linting"`, the upstream profile
returns `layers=[]` (linting doesn't produce ms-based per-layer data).
Orchestrate iterates `report.layers` and produces an empty allocation,
or worse fails depending on heuristics. Defensive empty-handling missing.

**Closure:** Detect empty layers list and either (a) re-run profile in
`detailed` mode automatically, or (b) raise a clear error.

### 2.5 Phase 2 / Phase 3 platforms have transport but no adapters

**Files:** `src/quad/platforms/linux.py`, `src/quad/platforms/android.py`

The SSH-based Linux adapter and ADB-based Android adapter both have
working `run_command()` and `detect_device()` methods, but the
`AdapterFactory` only knows about `QAIRTAdapter` (which runs locally).
There is no path to use these for remote profiling/inference.

**Closure:** Either (a) extend `QAIRTAdapter` with a `target` argument
that routes commands through `LinuxPlatform` or `AndroidPlatform`, or
(b) create `QAIRTRemoteAdapter` subclasses.

### 2.6 No CI on Windows; 8 path-assertion failures unfixed

**File:** `.github/workflows/ci.yml`

CI runs only on `ubuntu-latest`. The 8 known Windows failures are all
in test code, not source code — they assert Unix paths against
`Path(...)` results that are correctly Windows-style on Windows. The
fixes are one-line each (use `Path.as_posix()`).

The 8 tests:
- `tests/unit/test_compiler/test_model_conversion.py::test_output_dlc_auto_generated`
- `tests/unit/test_profiler/test_qhas_profiler.py::test_get_profilelogs_dir_custom`
- `tests/unit/test_profiler/test_qhas_profiler.py::test_get_reader_lib_path`
- `tests/unit/test_profiler/test_qhas_profiler.py::test_reader_lib_contains_sdk_root`
- `tests/unit/test_udo/test_udo.py::test_sdk_bin_returns_correct_path`
- `tests/unit/test_udo/test_udo.py::test_sdk_bin_for_quant`
- `tests/unit/test_udo/test_udo.py::test_sdk_bin_for_converter`
- `tests/unit/test_benchmarks/test_mobilenet_ssd_benchmarking.py::test_returns_latest_results_symlink_if_exists`
  (this last one is a Windows symlink-privilege issue; needs
  `pytest.mark.skipif(sys.platform == 'win32')`)

**Closure:** Add `windows-latest` to the CI matrix; fix the 8 tests in
the same PR so CI stays green.

### 2.7 Package name mismatch

**File:** `pyproject.toml`

Package name is `quad-agent`. README claims `pip install qualcomm-ai-toolkit` is the goal. Both names exist in docs simultaneously.

**Closure:** Pick one. If `qualcomm-ai-toolkit` is the target, rename
the package now (before any PyPI publication) and update all references.

### 2.8 `_create_dummy_input_list` blocks honest profiling

**File:** `src/quad/adapters/qairt_adapter.py:542-558`

Every code path (`profile`, `execute_inference`, quantization
calibration) calls this helper which always generates
`np.random.randn(1, 3, 224, 224)` regardless of the model's actual
input shape or dtype. This means:
- Profiling timings are taken on garbage data (latency may differ from
  real-input latency for data-dependent ops)
- Quantization calibration is wrong (random noise → wrong
  quantization scales)
- Models that don't accept `(1,3,224,224)` will fail

**Closure:** Read input shape/dtype from the converted DLC (via
`snpe-dlc-info`) and either (a) generate plausible inputs of the right
shape, or (b) accept a `--calibration-data` path argument.

---

## Tier 3 — Medium (polish, error handling, breadth)

### 3.1 Typed exceptions defined but inconsistently used

**Files:** `src/quad/exceptions.py` (16 well-typed exception classes)
vs. consumers that throw built-in `RuntimeError` / `ValueError` /
`EnvironmentError` instead.

Sampled offenders: `compiler/model_conversion.py`, `profiler/qhas.py`,
`adapters/qairt_adapter.py`, `udo/manager.py`, `accuracy/metrics.py`.

**Closure:** ruff custom rule banning bare `RuntimeError` /
`ValueError` in `src/quad/`; require typed exceptions from
`quad.exceptions`.

### 3.2 No auto-generated API documentation

There's no `sphinx`/`mkdocs` config. Every public function/class has
docstrings (sampled coverage is good) but they're invisible without
reading source. README + design PDFs are the only entry points.

**Closure:** Add `mkdocs` + `mkdocstrings`; auto-build on push to
`main` and publish to GitHub Pages.

### 3.3 Plugins folder is incomplete

- `plugins/vscode/` has a real TypeScript extension that connects to
  the MCP server (≈250 lines), but command handlers above ≈line 100
  are stubs.
- `plugins/arduino/` and `plugins/android-studio/` are referenced in
  the README but **not present** in the repo.

**Closure:** Either remove the references or fill in the plugins.

### 3.4 Model registry has no real models

**File:** `src/quad/serve/model_registry.py`

Five entries (`mobilenetv2`, `resnet50`, `yolov8n`, `whisper-tiny`,
`llama-7b`) point to paths like `models/yolov8n.qbin` that don't
exist. There's no `download_model(name)` function backed by S3 or a
CDN.

**Closure:** Either decide model registry is metadata-only (and
document it as such), or implement the download/cache path.

### 3.5 `deploy.sh` is SNPE-only and fragile

**File:** `deploy.sh:206-228`

Skel naming logic for HTP v65/v66 vs v68+ is hardcoded. No QNN
backend support. No version verification on the target. Works in the
lab; not production-grade.

**Closure:** Add QNN deploy path; query device for HTP version via
`adb getprop` / `cat /sys/devices/...` instead of trusting the env var.

### 3.6 `quad detect` is documented but is also a thin mock

**File:** `src/quad/cli/main.py:144-152` → `quad.runtime.list_devices`

`list_devices()` returns three hardcoded entries (Hexagon NPU, Adreno
X1-85, Oryon ARM64) on every machine. Doesn't actually probe local
hardware.

**Closure:** Implement local-only PnP/`/proc/cpuinfo`/`getprop` probing
behind `list_devices()`.

### 3.7 `get_supported_ops` is a hardcoded list

**File:** `src/quad/adapters/qairt_adapter.py:497-518`

130-op list copy-pasted from SDK docs. Doesn't change with SDK
version. Real adapter should query the SDK at runtime.

**Closure:** Run `qairt-converter --help` or
`snpe-dlc-info --supported-ops` and parse.

---

## Tier 4 — Low (nice to have)

- **`mypy --strict` not enforced in CI** — pyproject says strict, CI
  ignores it (`ci.yml` runs mypy with `--ignore-missing-imports`)
- **Pre-commit not enforced as a CI gate** — `.pre-commit-config.yaml`
  exists but local-only
- **No coverage report uploaded** — coverage target is 85% but CI
  doesn't track or fail on regression
- **README's roadmap "Phase 4: Cross-Platform 0%"** is honest but
  doesn't link to specific work items
- **`logging.py` exists** but not every module imports `structlog`
  consistently (some still call `print` in error paths — sampled
  in `udo/manager.py`)

---

## Recommended closure order

If the goal is "real-mode end-to-end works on the test laptop with the
QAIRT SDK installed":

1. **Tier 1.7** (templates package_data) — 30 min, prerequisite to anything that ships
2. **Tier 2.6** (Windows CI + 8 path-assertion fixes) — 2 hours, makes the test surface trustworthy
3. **Tier 2.8** (real `_create_dummy_input_list` based on DLC introspection) — 1 day, unblocks honest profiling
4. **Tier 1.8** (fill in QNN C++ template TODOs) — 2-3 days, makes `generate_code` produce shippable code
5. **Tier 1.4** (`execute_inference` real I/O marshalling) — 1 day, unblocks `quad serve` and `infer_batch`
6. **Tier 1.2 + 1.3** (HTTP server + real runtime backends) — 1-2 weeks
7. **Tier 1.5 + 1.6** (AIMET + AI Hub adapters) — 2 weeks each, in parallel
8. **Tier 1.1** (real compiler pipeline) — 2-3 weeks; this is the largest and depends on the runtime
9. **Tier 2.x** items as they come up
10. **Tier 3 / 4** — polish, can be parallel to user-facing work

---

## What's actually solid (so we don't break it)

The following are well-built and shouldn't be touched casually:
- `src/quad/sdk_manager.py` — discovery + install + state file (29 tests)
- `src/quad/adapters/factory.py` — strict mode, fallback tagging,
  real-mode-ready introspection (14 tests)
- `src/quad/cli/main.py` — CLI orchestration with `mode`, `sdk`,
  `doctor --real-mode`
- `install.sh` + `bootstrap.ps1` + `bootstrap.bat` + `setup_sdk.sh` —
  end-to-end installer
- The 5 MCP tool registrations in `src/quad/server/__init__.py` —
  clean dispatch, sensible defaults
- The model conversion config (`src/quad/compiler/model_conversion.py`)
  with all 18 SDK CLI flag builders — 125 tests, this is reference-quality
- Profiling levels (`profiler/levels.py`, `profiler/linting.py`,
  `profiler/qhas.py`) — well-structured, well-tested
- Template engine (`codegen/engine.py`) — Jinja2 + validators; only
  the *contents* of templates need work, not the engine

---

## Summary scorecard

| Layer | Mock-mode | Real-mode | Tier of biggest gap |
|---|---|---|---|
| MCP server + 5 tools | 95% | 50% | T1 (parsers, calibration) |
| AdapterFactory + SDK manager | 95% | 95% | T4 |
| `QAIRTAdapter` | n/a | 60% | T1 (execute_inference, calibration), T2 (parsers) |
| AIMET integration | n/a | 0% | T1 (entire adapter missing) |
| AI Hub integration | n/a | 0% | T1 (entire adapter missing) |
| Phase 2 (Linux QCS2210) | 70% | 10% | T2 (transport exists, not wired) |
| Phase 3 (Android SM8750) | 70% | 10% | T2 (transport exists, not wired) |
| Compiler pipeline | 60% | 10% | T1 (placeholder bytes) |
| Runtime (Device/Tensor/Model) | 80% | 10% | T1 (numpy mocks) |
| Codegen engine | 95% | n/a | n/a |
| Codegen templates | 60% C++ / 80% Py | n/a | T1 (C++ scaffolds) |
| Inference server | 50% | 0% | T1 (no HTTP) |
| Model registry | 80% | n/a | T3 (no downloads) |
| `deploy.sh` | n/a | 60% | T3 (SNPE-only, fragile) |
| Installer | 95% | 95% | T4 |
| CLI surface | 90% | 80% | T3 (`quad detect` mock) |
| VS Code plugin | 50% | n/a | T3 |
| Tests / CI | 1811/1819 pass on Linux | n/a | T2 (no Windows CI) |
| Packaging | 80% | n/a | T1 (templates not in sdist) |
| Documentation | 75% | n/a | T3 (no API ref site) |

**Overall:** Mock-mode is shippable for demos. Real-mode end-to-end
needs ~6-10 person-weeks of work, mostly concentrated in Tiers 1.1,
1.2, 1.3, 1.5, 1.6.
