---
name: quad-detect
description: Use when the user asks "what hardware do I have", "is this a Snapdragon", "do I have an NPU", "what's on this machine", or invokes /quad-detect. Calls the hardware_detect MCP tool and renders a rich summary with TOPS/TFLOPS, available runtimes, and SDK status.
trigger: "detect"
---

# QUAD Hardware Detection Skill

The user wants to know what compute hardware QUAD sees on their
machine.

## Steps

1. Call `mcp__quad__hardware_detect` with `platform="windows"` (use
   `linux` or `android` if they explicitly mention those). On Windows
   the platform value still gets the right answer — the MCP server
   internally probes the local host via PowerShell.

2. Also run the bash command `quad detect` for the OS-level probe
   that includes RAM, MHz, OS string, NPU PnP device.

3. Format the response with `quad.ui.format_device(profile)` for
   the chipset table.

4. Show available compute units as utilisation-bar primer rows:

   ```
   [NPU]  Hexagon NPU       ░░░░░░░░░░░░░░░░░░░  45 TOPS available
   [GPU]  Adreno X1-85       ░░░░░░░░░░░░░░░░░░░  4.6 TFLOPS available
   [CPU]  Oryon (12 cores)   ░░░░░░░░░░░░░░░░░░░  baseline
   ```

5. Add `quad mode` output so the user immediately sees whether real
   mode is READY.

6. End with 1-2 tips from `quad.tips.get_tips_for("detect")`.

## Special cases

- **Non-Qualcomm machine** — note that mock mode is fully functional
  for development. Recommend `./install.sh --mock-only` and the
  `quad-quickstart` flow.

- **Snapdragon detected but no SDK** — point at
  `docs/REAL_HARDWARE.md` and explain that QUAD will run in mock
  mode until they install QAIRT.

- **NPU detected but `quad mode` says NOT READY** — run
  `quad doctor --real-mode` to enumerate exactly what's missing.

## Example response

```markdown
### Hardware: Snapdragon X Elite X1E80100 (Windows)

|              | Detail                                       |
| :---         | :---                                         |
| **CPU**      | 12 × Oryon ARM64 @ 4.012 GHz                 |
| **GPU**      | Adreno X1-85 (4.6 TFLOPS)                    |
| **NPU**      | Hexagon NPU (45 TOPS)                        |
| **RAM**      | 31.6 GB                                      |
| **Runtimes** | cpu, gpu, npu                                |

`███████████████████████░` 45 TOPS NPU available

**SDK:** real-mode READY (qairt 2.45.0.260326 @ ./sdks/qairt-…)

💡 _Snapdragon X-series Copilot+ PC detected. Real-mode workflows are supported here._
```
