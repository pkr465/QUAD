# QUAD — Real Hardware Enablement Guide

> Audience: developers ready to flip QUAD off mock mode and run on real
> Qualcomm hardware (Snapdragon X Elite AI PC, Snapdragon 8 Elite phone,
> or Arduino UNO Q / QCS2210). Updated 2026-05-07.

This guide is the single source of truth for moving QUAD from mock to
real-hardware mode. It is meant to be followed top-to-bottom on a fresh
machine.

---

## 1. The mock → real switch in one paragraph

Every adapter call (`hardware_detect`, `convert_model`, `profile_workload`,
…) routes through `AdapterFactory` (`src/quad/adapters/factory.py`). The
factory looks at `ServerConfig.adapter_mode` (either `"mock"` or `"real"`,
sourced from `quad.toml` or `QUAD_ADAPTER_MODE`). In real mode it
constructs `QAIRTAdapter` (`src/quad/adapters/qairt_adapter.py`), which
shells out to the real Qualcomm CLI tools (`qairt-converter`,
`snpe-net-run`, `qnn-platform-validator`, …). If real mode is requested
but the SDK is missing, the factory **falls back to mock and tags the
adapter** with `fell_back_from_real=True` — set `QUAD_STRICT_REAL=1` to
make the factory raise `RealAdapterUnavailableError` instead.

---

## 2. Pre-requisites checklist

| Requirement | Verify with | Where it comes from |
| --- | --- | --- |
| Python 3.10+ | `python --version` | OS package manager |
| QUAD installed | `python -c "import quad; print(quad.__version__)"` | `pip install -e .` from this repo |
| QAIRT SDK 2.45+ unpacked locally | `ls $QAIRT_SDK_ROOT/bin/` | <https://softwarecenter.qualcomm.com> |
| SDK env activated | `which qairt-converter` returns a path | `source $QAIRT_SDK_ROOT/bin/envsetup.sh` |
| ADSP library path set (HTP/DSP) | `echo $ADSP_LIBRARY_PATH` | `export ADSP_LIBRARY_PATH=...` |
| Target device reachable | platform-specific (see §5) | n/a |

If you skip any of these, `quad doctor --real-mode` will tell you which
one and exit non-zero.

---

## 3. The 1-step enablement playbook

The recommended path is **`./install.sh --qairt-archive <path>`** which
bundles every step into a single command:

```bash
# 1. Download the SDK archive from the Qualcomm developer portal
#    (requires a developer account + EULA acceptance — there is no
#    public direct-download URL):
#      https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
#      https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai

# 2. Run the installer — does everything else
./install.sh --qairt-archive ~/Downloads/qairt-2.45.0.260326.zip
```

That single command:

* Creates `.venv/` and installs QUAD (`pip install -e .[dev]`)
* Unpacks the archive into `./sdks/qairt-<version>/` (gitignored)
* Sets `QAIRT_SDK_ROOT` / `QNN_SDK_ROOT` / `SNPE_ROOT` for the current shell
* Generates `quad.toml` and `.claude/settings.json`
* Generates `activate.sh` so future shells re-resolve the SDK automatically
* Runs `pytest -q` and prints the result
* Reports the path to use, e.g.:

  ```
  Platform:  quad-agent installed
  QAIRT SDK: qairt 2.45.0.260326 (real mode available)
             root: ./sdks/qairt-2.45.0.260326
  Tests:     1811 passed, 8 failed in 9.92s
  ```

If you didn't pass `--qairt-archive`, the installer tries every other
strategy automatically (existing `QAIRT_SDK_ROOT`, vendor-default
locations, `~/Downloads/qairt*.zip`, `QAIRT_DOWNLOAD_URL` +
`QAIRT_DOWNLOAD_TOKEN` for CI). If nothing's found, QUAD installs in
mock mode and prints clear next-step instructions including the URLs.

### Adding the SDK after the fact

If you ran `./install.sh --mock-only` first or only got the SDK later:

```bash
quad sdk install ~/Downloads/qairt-2.45.0.260326.zip
export QUAD_ADAPTER_MODE=real
quad mode                  # should report `real-mode: READY`
quad doctor --real-mode    # full pre-flight; exits non-zero on issues
```

**Power users** who already have the SDK installed at a vendor default
location (`C:\Qualcomm\AIStack\QAIRT\<ver>` on Windows or
`/opt/qcom/aistack/qairt/<ver>` on Linux) or who set `QAIRT_SDK_ROOT`
themselves can skip steps 2–3 — the discovery is a no-op when the env
var is already populated.

The MCP server runs the same discovery on every startup and writes the
resolved SDK info to `.quad/sdk.json` for inspection.

`quad mode` is the fastest way to see whether the next adapter call will
hit real hardware:

```
$ quad mode
adapter_mode:    real
strict:          True
real-mode:       READY
  reason:        Real mode active. SDK root: /opt/qairt/2.45.0.260326
```

`quad doctor --real-mode` runs the full pre-flight (Python, SDK env vars,
CLI tools in `PATH`, `ADSP_LIBRARY_PATH`, QHAS prerequisites, adapter-mode
consistency) and **escalates SDK warnings to errors** so missing tools
fail the check instead of being ignored.

---

## 4. Configuration reference

### 4.1 Environment variables (override TOML; useful in CI)

| Variable | Purpose | Example |
| --- | --- | --- |
| `QUAD_ADAPTER_MODE` | `mock` \| `real` | `real` |
| `QUAD_STRICT_REAL` | If `1`, raise instead of falling back to mock | `1` |
| `QAIRT_SDK_ROOT` | Primary SDK path (preferred) | `/opt/qairt/2.45.0.260326` |
| `QNN_SDK_ROOT` | Alternative QNN-only path | (rarely needed) |
| `SNPE_ROOT` | Legacy SNPE path | (legacy) |
| `ADSP_LIBRARY_PATH` | Skel `.so` discovery (HTP/DSP) | `$QAIRT_SDK_ROOT/lib/aarch64-android;/vendor/lib/rfsa/adsp` |
| `QAI_HUB_API_KEY` | AI Hub cloud profiling | (from <https://app.aihub.qualcomm.com>) |
| `ANDROID_NDK_ROOT` | Android AAR build | `/opt/android-ndk-r26` |
| `ANDROID_SERIAL` | Pin to a specific phone | `adb devices` to discover |
| `TARGET_IP` / `TARGET_USER` / `TARGET_SSH_KEY` | Linux target over SSH | `192.168.1.42` / `root` / `~/.ssh/id_rsa` |

### 4.2 `quad.toml` — minimum real-mode block

```toml
[server]
adapter_mode = "real"
log_level = "info"

[adapters.qairt]
sdk_path = "/opt/qairt/2.45.0.260326"
version = "2.45.0"

[dsp]
hexagon_version = "v75"   # v75 = 8 Gen 3 / X Elite, v79 = 8 Elite
target_os = "android"
```

---

## 5. Per-platform notes

### Phase 1 — AI PC (Windows on Snapdragon X Elite)

```powershell
$env:QAIRT_SDK_ROOT = "C:\Qualcomm\AIStack\QAIRT\2.45.0.260326"
& "$env:QAIRT_SDK_ROOT\bin\envsetup.ps1"
$env:QUAD_ADAPTER_MODE = "real"
quad mode
quad doctor --real-mode
```

Watch out for: Windows DSP signature errors (`transportStatus: 9 / 0x80000406`).
QUAD detects these in `_parse_stderr` and prints the recovery steps
(co-locate `libqnnhtpvXX.cat` with `libSnpeHtpVXXSkel.so`; do not modify
either file).

### Phase 2 — Arduino UNO Q (Linux QCS2210)

```bash
export TARGET_IP=192.168.1.50
export TARGET_USER=root
export TARGET_SSH_KEY=~/.ssh/id_rsa
ssh $TARGET_USER@$TARGET_IP "uname -a"   # confirm reachable
quad doctor --real-mode
```

`profile_workload` will cross-compile and push the model + skel `.so` to
`$deploy_dest` over SSH (default `/tmp/snpeexample`).

### Phase 3 — Mobile (Android, Snapdragon 8 Elite)

```bash
adb devices                              # confirm device shows up
export ANDROID_SERIAL=$(adb devices | awk 'NR==2{print $1}')
export ANDROID_NDK_ROOT=/opt/android-ndk-r26
quad doctor --real-mode
```

---

## 6. Known gaps (current blockers — keep updated)

These are the `QAIRTAdapter` paths still using stub parsing. Wiring them
up unblocks real-hardware mode end-to-end. Cross-reference with
`TODO.md` Priority 1.

| Gap | File / line | What's needed |
| --- | --- | --- |
| `convert_model` output parser | `src/quad/adapters/qairt_adapter.py:130` | `qairt-converter` stdout/stderr format on success and failure |
| `profile()` standard parser | `src/quad/adapters/qairt_adapter.py:263` | `snpe-diagview` text output schema |
| `detect_hardware()` parser | `src/quad/adapters/qairt_adapter.py:98` | `qnn-platform-validator` stdout |
| Calibration data loader | `src/quad/adapters/qairt_adapter.py:542` | Replace `np.random.randn` dummy with real calibration set |
| QNN C++ pipeline (alt to CLI) | `src/quad/qnn/inference_pipeline.py:85` | ctypes/cffi bindings for `libQnnHtp.so` |

When any of those parsers lands, no other code change is required to
flip the corresponding tool from mock to real — the adapter dispatch is
already wired.

---

## 7. Smoke test (after enablement)

```bash
# Detect hardware (real)
python -c "
import asyncio
from quad.adapters.factory import AdapterFactory
from quad.config import load_config
cfg = load_config()
factory = AdapterFactory(cfg, strict=True)
adapter = factory.get_adapter('qairt')
print(asyncio.run(adapter.detect_hardware('linux')))
"

# End-to-end via the existing sample app
python examples/sample_app.py
```

If `factory.real_mode_ready()` returns `(True, ...)` and the sample app
runs without raising `RealAdapterUnavailableError`, you're on real
hardware.

---

## 8. Rollback to mock (if real mode is broken)

```bash
unset QUAD_STRICT_REAL
export QUAD_ADAPTER_MODE=mock
quad mode    # should report adapter_mode: mock
```

Mock mode is always safe — no SDK, no hardware, deterministic synthetic
data. CI runs in mock mode by default.
