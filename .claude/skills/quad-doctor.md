---
name: quad-doctor
description: Use when the user asks "is QUAD set up correctly", "diagnose my install", "why isn't real mode working", "check the SDK", or invokes /quad-doctor. Runs the diagnostic suite and explains each warning + the exact fix.
trigger: "doctor"
---

# QUAD Doctor Skill

Run the QUAD environment diagnostic suite and translate the output
into actionable next steps.

## Steps

1. Run the bash command `quad doctor` (or `quad doctor --real-mode`
   for a strict pre-flight if the user wants to run on real hardware).

2. Format the output with `quad.ui.format_doctor(checks)` — gives a
   table with ✓/⚠/✗ status icons.

3. Summarise: total passed / warnings / errors at the top.

## For each non-pass check, surface the exact fix

| Check                            | Common fix                                      |
| :---                             | :---                                            |
| Python version                   | `winget install Python.Python.3.12`             |
| QUAD package                     | `pip install -e .`                              |
| quad.toml                        | `cp configs/quad.toml.example quad.toml`        |
| Dependencies                     | `pip install -e .[dev]`                         |
| Device detection                 | `quad detect --refresh`                         |
| Template directory               | Re-run `pip install -e .` (force-include)       |
| DLC compatibility                | `export QAIRT_SDK_ROOT=/path/to/qairt`          |
| WER support                      | Upgrade to QAIRT 2.28+                          |
| SDK env vars                     | `quad sdk install ~/Downloads/qairt-X.Y.Z.zip`  |
| SDK tools in PATH                | `source ./activate.sh`                          |
| DSP env (ADSP_LIBRARY_PATH)      | Set per `docs/REAL_HARDWARE.md`                 |
| Android tools                    | `winget install Google.AndroidStudio.Tools`     |
| QHAS prerequisites               | `quad sdk status` to verify reader lib present  |
| AIMET integration                | `pip install aimet-torch`                       |
| AI Hub integration               | Set `QAI_HUB_API_KEY` from app.aihub.qualcomm.com |
| Adapter mode (real)              | `export QUAD_ADAPTER_MODE=real`                 |

## Special handling

- **All passing** — congratulate, suggest `quad-quickstart` next.
- **Real-mode pre-flight failed** — the user almost certainly needs
  to install QAIRT. Walk through `docs/REAL_HARDWARE.md` step by step.
- **Warnings only** — explain that QUAD will still run (mock mode),
  and the warnings are about which features are degraded.

## Output template

```markdown
### Doctor report

|  | Check                       | Detail                                  |
| :-: | :---                      | :---                                    |
| ✓ | Python version              | Python 3.12.10 (>= 3.10)                |
| ✓ | QUAD package                | quad v0.3.0 importable                  |
| ⚠ | SDK env vars                | No SDK env vars set. Real mode disabled. |
| ✓ | Adapter mode (real)         | Real mode active. SDK root: ./sdks/...  |

**Summary:** 13 passed · 2 warnings · 0 errors

**Action items:**
1. `SDK env vars` is a warning — install QAIRT to enable real mode:
   ```
   quad sdk install ~/Downloads/qairt-2.45.0.260326.zip
   ```
   See [docs/REAL_HARDWARE.md](docs/REAL_HARDWARE.md) for the full setup.
```
