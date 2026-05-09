# QUAD Usage Guide

> **QUAD** — Qualcomm Unified Agent for Developers
> A unified AI computing platform for all Qualcomm SoCs.

This guide covers the full QUAD platform — from installation to production inference serving.

---

## Quick Start (3 commands)

```bash
./install.sh          # Install + verify (runs the full test suite)
./launch.sh           # Start MCP server (mock mode)
quad quickstart       # Interactive zero-to-inference wizard
```

> The three CLI entry points after install: `quad` (the developer toolchain),
> `quad-server` (long-running MCP server), `quad-client` (lightweight Claude
> Code provisioner — see [`docs/CLIENT_INSTALL.md`](CLIENT_INSTALL.md)).

---

## Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- pip
- Git

### Automated Install

```bash
git clone git@github.qualcomm.com:pavanr/QUAD.git && cd QUAD
./install.sh
```

**What `install.sh` does:**
1. Verifies Python 3.10+
2. Creates `.venv/` virtual environment
3. Installs `quad-agent` package + dev dependencies
4. Copies `configs/quad.toml.example` → `quad.toml`
5. Verifies `.claude/settings.json` for MCP auto-detection
6. Runs the full test suite

### Manual Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp configs/quad.toml.example quad.toml
pytest tests/ -v
```

---

## Claude Code MCP Auto-Detection

QUAD includes `.claude/settings.json` **committed to the repository**. Claude Code automatically discovers and registers the QUAD MCP server when opening this project.

```json
{
  "mcpServers": {
    "quad": {
      "command": "python",
      "args": ["-m", "quad.mcp.server"],
      "env": {"QUAD_ADAPTER_MODE": "mock"}
    }
  }
}
```

> `quad.mcp.server` is the canonical module path. `quad.server.main` works
> too — `quad.server` is a thin backward-compat shim that re-exports the
> FastMCP app.

**No manual configuration needed.** Just run `./launch.sh` and use Claude Code normally.

---

## Platform Modules

### 1. Runtime — `quad.runtime`

Core programming interface (like CUDA Runtime API):

```python
import quad.runtime as quad

# Device discovery
device = quad.Device("npu")          # Or "auto" (NPU > GPU > CPU)
devices = quad.list_devices()        # [npu, gpu, cpu]
assert quad.is_available("npu")

# Load model (one line)
model = quad.load("model.onnx", device="npu", power_budget_mw=3000)

# Inference
input_t = quad.Tensor.rand(1, 3, 224, 224, device="npu")
output = model(input_t)              # Returns Tensor on device

# Async inference
future = model.infer_async(input_t)
result = future.result()

# Power modes
model.set_power_mode("efficiency")   # performance | balanced | efficiency
```

### 2. Compiler — `quad.compiler`

Portable IR and cross-target compilation (like nvcc + PTX):

```python
from quad.compiler import compile_model, compile_onnx, QBin

# Compile to fat binary (all targets)
qbin = compile_model("model.onnx", output_path="model.qbin", targets="all")

# Compile for specific chipset
qbin = compile_model("model.onnx", targets=["qnpu_v3", "qdsp_v66"])

# Portable mode (IR only, JIT at load time)
qbin = compile_model("model.onnx", portable=True)

# Inspect IR
from quad.compiler import compile_onnx
ir = compile_onnx("model.onnx")
print(f"Nodes: {ir.num_nodes}, Inputs: {ir.inputs}, Outputs: {ir.outputs}")
```

**Compute Capabilities:**
| Name | Chipset | NPU TOPS | INT4 |
|------|---------|----------|------|
| `qnpu_v3` | Snapdragon X Elite | 45 | Yes |
| `qnpu_v3_mobile` | Snapdragon 8 Elite | 48 | Yes |
| `qnpu_v2` | Snapdragon 8 Gen 3 | 36 | No |
| `qdsp_v66` | QCS2210 | 1 | No |

### 3. Libraries — `quad.libs`

Optimized compute primitives (like cuDNN + cuBLAS):

```python
from quad.libs import nn, blas
from quad.runtime import Tensor, Device

# Neural network operations
conv = nn.Conv2d(3, 64, kernel_size=3, device="npu")
output = conv(Tensor.rand(1, 3, 224, 224, device="npu"))

attention = nn.FlashAttention(embed_dim=768, num_heads=12, device="npu")
output = attention(query, key, value)

fused = nn.FusedConvBnRelu(64, 128, kernel_size=3, device="npu")
output = fused(input_tensor)

# BLAS operations
C = blas.gemm(A, B, device="npu")
C_batch = blas.batched_gemm(A_batch, B_batch, device="npu")
```

**Available Operations:** `nn.list_ops()` → Conv2d, Linear, MultiHeadAttention, LayerNorm, FusedConvBnRelu, FlashAttention

### 4. Optimizer — `quad.optimizer`

Graph optimization pipeline (like TensorRT):

```python
from quad.optimizer import optimize_model

result = optimize_model(
    "model.onnx",
    target="qnpu_v3",
    quantization="int8",
    power_budget_mw=5000,
)

print(f"Nodes: {result.original_nodes} → {result.optimized_nodes}")
print(f"Speedup: {result.estimated_speedup:.1f}x")
print(f"Power reduction: {result.estimated_power_reduction_pct:.0f}%")
print(f"Passes: {result.passes_applied}")
```

**Optimization Passes:**
- `FusionPass` — Conv+BN+ReLU → single kernel
- `ConstantFoldingPass` — pre-compute static expressions
- `DeadCodePass` — remove unused nodes
- `MemoryPlanningPass` — buffer reuse optimization

### 5. Profiler — `quad.profiler`

Deep hardware profiling (like Nsight Systems + Compute):

```python
from quad.profiler import profile_model

# Full profiling (all levels)
summary = profile_model("model.qbin", level="deep", device="npu")

# Roofline analysis
print(summary.roofline.diagnosis)        # "compute-bound" or "memory-bound"
print(summary.roofline.achieved_pct)     # % of peak hardware utilization
print(summary.roofline.recommendation)   # Automated suggestion

# Power breakdown
print(summary.power_trace.avg_power_mw)
print(summary.power_trace.breakdown_pct) # {"npu": 62%, "gpu": 15%, "cpu": 23%}

# Battery impact
impact = summary.power_trace.battery_impact(battery_mah=5000, voltage=3.7)
print(f"Battery life: {impact.hours_at_workload:.1f} hours")

# Top bottleneck kernels
for k in summary.kernel_report.top_kernels(5):
    print(f"  {k.name}: {k.latency_us}µs ({k.bottleneck})")
```

**Profiling Levels:**
| Level | What It Shows |
|-------|--------------|
| `system` | CPU+GPU+NPU+DMA timeline, idle gaps |
| `kernel` | Per-op roofline, utilization, stall reasons |
| `deep` | All of the above + memory + power |

### 6. Kernels — `quad.kernels`

Custom NPU programming (like CUDA C++):

```python
from quad.kernels import kernel, compile_kernel, Graph
from quad.runtime import Tensor

# Define custom kernel
@kernel
def fused_gelu(x: Tensor, output: Tensor):
    """Custom fused GELU on Hexagon NPU."""
    for i in range(x.size):
        val = x[i]
        output[i] = 0.5 * val * (1 + tanh(0.7978845 * (val + 0.044715 * val**3)))

# Compile for target
compiled = compile_kernel(fused_gelu, target="hexagon_v73")

# Execute
input_t = Tensor.rand(1, 768, device="npu")
output_t = Tensor.zeros(1, 768, device="npu")
compiled(input_t, output_t)

# Register as custom op (usable in ONNX graphs)
compiled.register_as_op("FusedGeLU")
```

**QUAD Graphs** (captured execution — like CUDA Graphs):

```python
from quad.kernels import Graph

# Capture execution sequence
with Graph.capture() as g:
    y = model_a(x)
    z = model_b(y)

# Replay with minimal overhead (sub-100µs)
for frame in video_stream:
    x.copy_from(frame)
    g.replay()
```

### 7. Serve — `quad.serve`

Production inference server (like Triton):

```python
from quad.serve import ModelServer

# Create server
server = ModelServer(port=8080, power_budget_mw=10000)

# Load models on different compute units
server.load_model("yolo", "models/yolo.qbin", device="npu:0")
server.load_model("resnet", "models/resnet.qbin", device="npu:1")

# Inference
response = server.infer("yolo", {"image": input_data})
print(f"Latency: {response.latency_ms}ms")

# Batch inference
responses = server.infer_batch("resnet", batch_of_inputs)

# Health & metrics
print(server.health())   # HealthStatus(status="healthy", models=2)
print(server.metrics())  # avg_latency, throughput_rps, power_mw
```

**Model Zoo** (pre-registered models):

```python
from quad.serve.model_registry import ModelRegistry

registry = ModelRegistry()
models = registry.search("classification")
# Returns: mobilenetv2, resnet50 with published latency/accuracy
```

### 8. CLI — `quad` command

```bash
# Getting started
quad quickstart              # Interactive wizard: detect → compile → profile → generate
quad doctor                  # Run 16 environment diagnostics
quad doctor --real-mode      # Strict pre-flight (exits non-zero on any SDK issue)
quad version                 # Show QUAD version

# SDK management
quad sdk status              # Show the active SDK + version + bin dir
quad sdk discover            # Scan all standard locations
quad sdk install <archive>   # Unpack a downloaded SDK archive into ./sdks/
quad mode                    # Show adapter mode + real-mode readiness
quad mode --set real         # Print 'export QUAD_ADAPTER_MODE=real' for shell-eval

# Compilation
quad compile model.onnx --output model.qbin --targets all
quad compile model.onnx --targets qnpu_v3,qdsp_v66
quad compile model.onnx --portable                  # IR only (JIT at load)
quad compile model.onnx --quantization int8         # INT8 via the active backend
quad compile model.onnx --backend qairt             # Force the real qairt backend
quad compile model.onnx --coverage-only             # IR + per-target op-coverage report

# Optimization
quad optimize model.onnx --target qnpu_v3 --quantization int8 --power-budget 5000

# Profiling
quad profile model.qbin --level deep --device npu
quad profile model.qbin --level system              # Timeline-only (fastest)

# Benchmarking
quad benchmark                                      # All default models
quad benchmark --device npu --models mobilenetv2 --models resnet50

# Serving (model_path is positional)
quad serve model.qbin --port 8080 --device npu --name mnet

# Hardware
quad detect                  # Show available compute units
quad detect --refresh        # Re-probe; bypass the discovery cache
```

---

## MCP Agent Tools (via Claude Code)

When using Claude Code in this project, 5 MCP tools are available via natural language:

| Tool | What to Ask Claude |
|------|--------------------|
| `hardware_detect` | "What Qualcomm hardware is available?" |
| `convert_model` | "Convert model.onnx to QNN format with INT8" |
| `profile_workload` | "Profile this model on the NPU" |
| `orchestrate_workload` | "Allocate layers for best power efficiency" |
| `generate_code` | "Generate Python inference code for this model" |

---

## Testing

```bash
make test                    # Full suite with coverage
make test-unit               # Unit tests only (fast)
pytest tests/integration/    # Integration tests
pytest tests/e2e/            # End-to-end pipeline
pytest -k "profiler"         # Run tests matching pattern
pytest tests/ --cov=quad --cov-report=html  # Coverage report
```

**Test layout** (`tests/`):

| Module                | What's tested                                          |
| --------------------- | ------------------------------------------------------ |
| `unit/test_runtime`   | Device, Tensor, Model, Stream, Memory, Power           |
| `unit/test_compiler`  | IR, QBin, capabilities, ONNX frontend, pipeline        |
| `unit/test_libs`      | Conv2d, Linear, Attention, BLAS ops                    |
| `unit/test_optimizer` | Fusion, dead code, memory planning, pipeline           |
| `unit/test_profiler`  | Roofline, kernel, power, memory, system                |
| `unit/test_kernels`   | DSL, primitives, Graph capture/replay                  |
| `unit/test_serve`     | Server, registry, deployment                           |
| `unit/test_cli`       | quickstart, doctor, benchmark, mode                    |
| `unit/test_tools`     | All 5 MCP tools through the mock adapter               |
| `unit/test_models`    | Pydantic schema validation                             |
| `unit/test_adapters`  | Mock + QAIRT adapter, factory                          |
| `unit/test_codegen`   | Templates, validators                                  |
| `integration/`        | Cross-module flows                                     |
| `e2e/`                | Full pipeline, TTFI, cross-platform                    |

`pytest --collect-only -q` prints the live count — that's the source of truth.

---

## Configuration

### quad.toml

```toml
[server]
adapter_mode = "mock"     # "mock" (development) or "real" (hardware)
log_level = "info"        # debug | info | warning | error
log_format = "console"    # "json" for production
model_output_dir = "./output"
template_dir = "./templates"

[adapters.qnn]
sdk_path = ""             # Fill when QNN SDK installed

[adapters.snpe]
sdk_path = ""             # Fill when SNPE SDK installed

[adapters.ai_hub]
api_key_env = "QAI_HUB_API_KEY"
```

### Environment Overrides

```bash
QUAD_ADAPTER_MODE=mock    # Force mock mode
QUAD_LOG_LEVEL=debug      # Verbose logging
QAI_HUB_API_KEY=sk-...   # AI Hub authentication
```

---

## Launch Options

```bash
./launch.sh                    # Default: mock + stdio (Claude Code)
./launch.sh --mock --verbose   # Mock with debug logging
./launch.sh --sse              # SSE transport (IDE plugins)
./launch.sh --real             # Real mode (requires SDKs)
./launch.sh --help             # All options
```

---

## Development Workflow

```bash
source .venv/bin/activate      # Activate environment
# ... make code changes ...
make test                      # Run tests
make lint                      # Ruff + mypy
make format                    # Auto-format
./launch.sh --verbose          # Test server
```

| Command | Description |
|---------|-------------|
| `make serve` | Start MCP server |
| `make test` | All tests + coverage |
| `make test-unit` | Unit tests only |
| `make lint` | Ruff check + mypy |
| `make format` | Auto-format code |
| `make clean` | Remove build artifacts |
| `make install` | Install + pre-commit hooks |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `quad: command not found` | `source ./activate.sh` (or `source .venv/bin/activate`); `pip install -e ".[dev]"` re-registers the `quad` script |
| `ModuleNotFoundError: quad` | `source .venv/bin/activate && pip install -e .` |
| `quad.toml not found` | `cp configs/quad.toml.example quad.toml` |
| Tests fail on import | `pip install -e ".[dev]"` |
| Server won't start | `python -m quad.mcp.server` for the direct error |
| `bash: command not found` on Windows | Run `.\bootstrap.ps1` first (installs Git Bash via winget) |
| Permission denied | `chmod +x install.sh launch.sh` |
| Template render error | Run from the QUAD root directory |
| Real-mode infer falls back to mock | `quad doctor --real-mode` prints the exact missing piece |

Run `quad doctor` for automated environment diagnostics.

---

## Uninstalling

```bash
# 1. Drop the bundled Claude Code skills (settings.json is preserved)
quad-client uninstall

# 2. Remove the Python package
pip uninstall quad-agent

# 3. (Optional) Wipe the local checkout state
rm -rf .venv sdks/ .quad/ output/ quad.toml .claude/settings.json
```

`quad-client uninstall` deliberately keeps `.claude/settings.json` —
delete it manually only if you're fully detaching Claude Code from this
project.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI: quad quickstart | doctor | benchmark | compile | serve     │
├─────────────────────────────────────────────────────────────────┤
│  MCP Agent: 5 tools (Claude Code natural-language interface)     │
├─────────────────────────────────────────────────────────────────┤
│  Serve: ModelServer | Registry | Deploy                          │
├─────────────────────────────────────────────────────────────────┤
│  Optimizer: Fusion | ConstantFold | DeadCode | MemoryPlan        │
├─────────────────────────────────────────────────────────────────┤
│  Libraries: QualcommDNN | QualcommBLAS | QualcommFFT             │
├─────────────────────────────────────────────────────────────────┤
│  Runtime: Device | Tensor | Model | Stream | Memory | Power      │
├─────────────────────────────────────────────────────────────────┤
│  Compiler: ONNX Frontend | QUAD IR | QBin | JIT                  │
├─────────────────────────────────────────────────────────────────┤
│  Kernels: @kernel DSL | HVX Primitives | QUAD Graphs             │
├─────────────────────────────────────────────────────────────────┤
│  Profiler: Roofline | Kernel | Power | Memory | System           │
├─────────────────────────────────────────────────────────────────┤
│  Hardware: CPU (Oryon/Kryo) | GPU (Adreno) | NPU (Hexagon)       │
└─────────────────────────────────────────────────────────────────┘
```
