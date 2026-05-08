---
name: quad-convert
description: Use when the user asks to "convert a model", "compile this ONNX", "quantize my PyTorch model", "convert to QNN/SNPE", or invokes /quad-convert. Walks through model conversion with calibration data + image format guidance.
trigger: "convert"
---

# QUAD Model Conversion Skill

Convert ONNX / PyTorch / TensorFlow / TFLite models to QNN context
binaries (.bin) or SNPE DLC (.dlc) files, with optional INT8/INT4
quantization.

## Pre-conversion checklist

1. **Source file exists?** If not, ask the user to share the path.
2. **Source format?** Inspect file extension (.onnx / .pt / .pb / .tflite).
   PyTorch users need to export to ONNX first — use `torch.onnx.export`.
3. **Target SDK?** Default to `qnn` on Windows (Snapdragon X Elite),
   `snpe` on Linux/Android. Ask if unclear.
4. **Quantization?** Recommend `int8` by default. Use `int4` only with
   real calibration data + AIMET (`use_aimet=True`).
5. **Image input?** Ask about layout (NCHW from PyTorch / NHWC from
   TensorFlow) and channel order (RGB / BGR). Set `input_layout=` and
   `channel_order=` accordingly so QUAD inserts the right transposes.

## Steps

1. Call `mcp__quad__convert_model` with the right args. Always
   provide:
   - `source_format`
   - `model_path`
   - `target_sdk`
   - `quantization`
   - `input_layout` (default `nchw` for ONNX from PyTorch)
   - `channel_order` (default `rgb`)

2. Format the response with `quad.ui.format_conversion(result)`. The
   table shows original/output size, compression ratio, supported-ops
   percent, conversion time, and target SDK.

3. If `unsupported_ops` is non-empty, surface it prominently — those
   ops will fall back to CPU and may dominate latency.

4. Surface the `image_format_notes` from the response — they tell the
   user how to preprocess inputs at inference time. This is one of the
   biggest sources of "model returns garbage" bug reports.

5. If the user is doing INT4 + has no `calibration_data`, raise the
   AIMET tip: "Random-data calibration produces wrong scales. Provide
   real calibration samples via `calibration_data=`."

6. Suggest the next step:
   - "Now profile it on the NPU: `quad-profile <output_path>`"
   - "Want to allocate it across CPU/GPU/NPU? Use `quad-orchestrate`"

7. End with 1-2 tips from `quad.tips.get_tips_for("convert")`.

## Common gotchas to call out

- **PyTorch model_path with .pt** — PyTorch frontend uses a mock IR
  for now. Recommend exporting to ONNX first.
- **Symbolic batch dim** — ONNX models with `dim_param` need
  `input_dimensions={"input": "1,3,224,224"}` to bind a concrete shape.
- **MNIST / single-channel** — provide `input_layout` correctly; the
  MNIST preset is `{1, 1, 28, 28}` not `{1, 3, 224, 224}`.
- **Mean values** — many ImageNet models expect normalisation
  (e.g. mean=`[123.675, 116.28, 103.53]`). Pass via `mean_values=`.

## Example response

```markdown
### Model converted: `output/mobilenetv2-12.dlc`

|                  |                    |
| :---             | :---               |
| Original size    | 13.3 MB            |
| Output size      | 3.3 MB             |
| Compression      | 4.05× smaller      |
| Quantization     | int8               |
| Supported ops    | 100%               |
| Conversion time  | 1.33 s             |
| Target SDK       | qairt              |

**Image format guidance:**
- Input layout is NCHW (PyTorch). SNPE requires NHWC — transpose
  before inference: `np.transpose(img, (0, 2, 3, 1))`
- Channel order is RGB — standard for most modern frameworks.

💡 _Pass real calibration_data to INT8/INT4 conversion. Random noise
produces wrong quantization scales._
```
