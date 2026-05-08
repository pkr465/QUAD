---
name: quad-codegen
description: Use when the user asks to "generate inference code", "write a C++ app for this model", "give me Python boilerplate", "create an Android JNI binding", or invokes /quad-codegen. Picks the right (platform, language, sdk) combo and emits compilable code.
trigger: "codegen"
---

# QUAD Code Generation Skill

Emit platform-specific inference application code from a converted
model. The generated code is real (not a TODO scaffold) — it does
real QNN init, model load, I/O marshalling, graph execute, and
cleanup.

## Steps

1. Determine `(platform, language, sdk)`:
   - **Windows + C++ + QNN** — Windows-on-Snapdragon native app
     (the most-tested path, real QNN context binary loading)
   - **Windows + Python + QNN** — Python via onnxruntime-qnn
   - **Linux + Python + SNPE** — Yocto / generic Linux on QCS2210
   - **Linux + arduino_sketch + SNPE** — Arduino UNO Q
   - **Android + Kotlin + SNPE** — Android library (.aar)
   - **Android + JNI + SNPE** — JNI bridge for hybrid apps

2. Call `mcp__quad__generate_code` with those args + `model_path`.

3. List the source files emitted. For each, show:
   - Filename
   - Line count
   - One-line description (e.g. "Real QNN inference engine with
     dlopen/dlsym, contextCreateFromBinary, graphExecute,
     QNN_TENSOR_SET_CLIENT_BUF I/O marshalling, cleanup")

4. Surface the `build_instructions` field — paste exactly so the
   user can copy-run.

5. Surface the `dependencies` list — pip installs, CMake versions,
   gradle artifacts, etc.

6. End with 1-2 tips from `quad.tips.get_tips_for("codegen")`.

## What's actually in the generated code

Phase B (T1.8) of the gap-closure plan replaced the Windows C++
template with a real implementation:

  * Real backend library loading via dlopen/LoadLibrary
  * QnnInterface_getProviders + version-checked provider selection
  * Full QNN init: logCreate -> backendCreate -> deviceCreate -> contextCreate
  * Model load: contextCreateFromBinary for .bin files
    (compose-from-.so path documented but raises clear error)
  * Real I/O marshalling:
    - graphGetInputTensors / graphGetOutputTensors -> discover schema
    - QNN_TENSOR_GET_DIMENSIONS / GET_RANK / GET_DATA_TYPE -> sizing
    - read_raw_file -> populate input buffers
    - QNN_TENSOR_SET_CLIENT_BUF -> bind buffers
    - QnnGraph_execute -> actual inference call
    - write_raw_file -> dump outputs
  * Cleanup in reverse order, all best-effort

Other templates (linux/python, linux/arduino, android/kotlin,
android/jni, qnn/.so / .bin / .tflite) are at varying levels of
completeness — a few still have TODO bodies. If the user gets a
template with TODOs, recommend filling them with reference to
`src/quad/qnn/inference_pipeline.py` which documents the full
6-step pipeline.

## Validating output

- The codegen engine runs `quad.codegen.validators.validate_output`
  on every emitted file, which checks:
  - Brackets balanced (after stripping string literals)
  - No unrendered Jinja2 placeholders
  - Optional: `gcc -fsyntax-only` if `QUAD_VALIDATE_CPP_SYNTAX=1`
  - Optional: reject any TODO markers if `QUAD_STRICT_TODOS=1`

If validation fails, the tool raises `TemplateRenderError` with the
specific issue.
