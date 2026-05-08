---
name: quad-quickstart
description: Use when the user asks to "get started with QUAD", "set up Qualcomm AI", "configure my Snapdragon laptop for QUAD", or invokes /quad-quickstart. Walks through hardware detection -> SDK check -> sample inference end-to-end with rich summaries at each step.
trigger: "quickstart"
---

# QUAD Quickstart Skill

You're guiding the user from cold-start to running inference on Qualcomm
hardware (real or mock). The QUAD MCP server gives you 5 tools plus
`quad mode` / `quad sdk` / `quad doctor`. Use them in this order:

## Step-by-step flow

1. **Detect hardware** — call `mcp__quad__hardware_detect` with `platform="windows"`
   (or `linux` / `android`). Show the user a table with chipset, CPU/GPU/NPU,
   RAM, available runtimes.

2. **Check SDK readiness** — run the bash command `quad mode` via the Bash tool
   and `quad sdk status`. Tell the user whether they're in real or mock mode.

3. **Recommend a sample model** based on the detected device:
   - Snapdragon X Elite → `mobilenetv2-12.onnx` (the same model that the
     real-hardware sample app uses)
   - Other → same default

4. **Convert** — call `mcp__quad__convert_model` with `quantization="int8"` and
   `input_layout="nchw", channel_order="rgb"` (correct for ONNX models).
   Show the conversion result table including image format guidance.

5. **Profile** — call `mcp__quad__profile_workload` with `runtime="auto"`,
   `profiling_level="detailed"`. Show latency stats, throughput, utilisation.

6. **Linting profile** — second call with `profiling_level="linting"`. Show
   any HTP bottlenecks discovered.

7. **Orchestrate** — call `mcp__quad__orchestrate_workload` with each of the
   three power modes (`performance`, `balanced`, `efficiency`) and present a
   side-by-side comparison.

8. **Generate code** — call `mcp__quad__generate_code` with the user's chosen
   platform/language. List the generated files.

## Output style

- Use markdown tables for every numeric output. Don't paste raw JSON.
- After each step include 1-2 contextual tips (you can fetch them via
  `from quad.tips import get_tips_for`).
- End with a "Next steps" block: link to the `serve` skill, mention `quad
  doctor --real-mode` for a strict pre-flight, suggest an example model
  to try next.

## Prereqs to check first

- If `quad mode` reports `NOT READY`, point at `docs/REAL_HARDWARE.md`
  before continuing.
- If they have no QAIRT SDK, the workflow still works in mock mode —
  call this out clearly so the user knows the numbers aren't real
  hardware measurements.

## Example invocation

> "I just installed QUAD on my Snapdragon X Elite laptop. What do I do next?"

→ run `mcp__quad__hardware_detect`, then walk through steps 2-8.
