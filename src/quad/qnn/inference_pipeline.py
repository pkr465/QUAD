"""QNN SDK Inference Pipelines — initialization sequences and code guidance.

Based on QNN SDK API Review (Nov 12, 2024, 80-63442-50 Oct 17 2024 base).

Three model formats supported by QNN C++ API:
  .so    — model compiled to shared object; contains QnnModel_composeGraphs()
  .bin   — context binary cache; faster init, backend-specific
  .tflite — TFLite model via QNN Delegate API

.so Pipeline (6 steps):
  1. Load libQnnBackend.so → get QNN interfaces
  2. Load model.so → get QnnModel_composeGraphs()
  3. Init: QnnLog → QnnBackend → QnnDevice → QnnContext
  4. Init QnnGraph via QnnModel_composeGraphs() → Finalize
  5. Execute inference
  6. Free: Graph → Context → Device → Backend → Log

.bin Pipeline (6 steps):
  1. Load libQnnBackend.so → get QNN interfaces
  2. Load libQnnSystem.so → get system interfaces
  3. Init: QnnLog → QnnBackend → QnnDevice
  4. Load QnnContext binary cache (model.bin)
  5. Execute inference
  6. Free resources

  NOTE: model.bin is backend-specific. The backend used when loading MUST
  match the backend used when the cache was saved.

.tflite Pipeline via QNN Delegate (4 steps):
  1. Load model.tflite → build TensorFlow Lite interpreter
  2. Register QNN delegate with interpreter
  3. Execute inference
  4. Delete QNN delegate

Key insight — why model.bin exists:
  Context init + graph composition + finalization is time-consuming.
  model.bin caches the prepared+finalized context so subsequent loads
  skip these steps → significantly faster initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Model Format Enum
# ══════════════════════════════════════════════════════════════════════════════

class QnnModelFormat(str, Enum):
    """Supported QNN model formats."""
    SO = "so"           # Shared object — requires QnnModel_composeGraphs()
    BIN = "bin"         # Context binary cache — backend-specific, fast load
    TFLITE = "tflite"   # TFLite via QNN Delegate API


class QnnBackendLib(str, Enum):
    """QNN backend shared libraries."""
    CPU = "libQnnCpu.so"
    GPU = "libQnnGpu.so"
    HTP = "libQnnHtp.so"       # Hexagon Tensor Processor (NPU)
    DSP = "libQnnDsp.so"
    HTA = "libQnnHta.so"
    SAVER = "libQnnSaver.so"   # For saving context binary cache


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Step Descriptors
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineStep:
    """One step in a QNN inference pipeline."""
    number: int
    name: str
    description: str
    key_calls: list[str] = field(default_factory=list)   # C++ API calls
    notes: list[str] = field(default_factory=list)


# ── .so Pipeline ─────────────────────────────────────────────────────────────

SO_PIPELINE_STEPS: list[PipelineStep] = [
    PipelineStep(
        number=1,
        name="Load Backend Library",
        description=(
            "Dynamically load the QNN backend shared library (e.g. libQnnHtp.so) "
            "and retrieve QNN interface function pointers."
        ),
        key_calls=["dlopen(libQnnBackend.so)", "QnnInterface_getProviders()"],
        notes=["One of: libQnnCpu.so, libQnnGpu.so, libQnnHtp.so, libQnnDsp.so"],
    ),
    PipelineStep(
        number=2,
        name="Load Model Library",
        description=(
            "Dynamically load model.so and retrieve the QnnModel_composeGraphs() "
            "function pointer. This function will compose the QNN graph."
        ),
        key_calls=["dlopen(model.so)", "dlsym(QnnModel_composeGraphs)"],
        notes=["model.so is generated from model.cpp by the QNN SDK converter"],
    ),
    PipelineStep(
        number=3,
        name="Initialize QNN Environment",
        description=(
            "Initialize QNN components in order: Log first, then Backend, "
            "Device, and Context."
        ),
        key_calls=[
            "QnnLog_create()",
            "QnnBackend_create()",
            "QnnDevice_create()",
            "QnnContext_create()",
        ],
        notes=["QnnLog can be initialized before QnnBackend"],
    ),
    PipelineStep(
        number=4,
        name="Compose and Finalize Graph",
        description=(
            "Call QnnModel_composeGraphs() to create and populate the QNN graph "
            "inside the context. Then finalize the graph to prepare for execution."
        ),
        key_calls=[
            "QnnModel_composeGraphs(backend, interface, context, ...)",
            "QnnGraph_finalize(graph)",
        ],
        notes=["Graph must be finalized before execution"],
    ),
    PipelineStep(
        number=5,
        name="Execute Inference",
        description=(
            "Set input tensors, execute the graph, and retrieve output tensors."
        ),
        key_calls=[
            "QnnTensor_createContextTensor() / QnnTensor_createGraphTensor()",
            "QnnGraph_execute(graph, inputs, n_inputs, outputs, n_outputs, profile, signal)",
        ],
        notes=[
            "Input tensors populated with raw input data",
            "Output tensors contain inference results after execute",
        ],
    ),
    PipelineStep(
        number=6,
        name="Free Resources",
        description="Release all QNN resources in reverse initialization order.",
        key_calls=[
            "QnnGraph_free()",
            "QnnContext_free()",
            "QnnDevice_free()",
            "QnnBackend_free()",
            "QnnLog_free()",
        ],
        notes=["Always free in reverse order of initialization"],
    ),
]


# ── .bin Pipeline ─────────────────────────────────────────────────────────────

BIN_PIPELINE_STEPS: list[PipelineStep] = [
    PipelineStep(
        number=1,
        name="Load Backend Library",
        description=(
            "Dynamically load the QNN backend library and retrieve interfaces. "
            "Must use the SAME backend as when model.bin was saved."
        ),
        key_calls=["dlopen(libQnnBackend.so)", "QnnInterface_getProviders()"],
        notes=["Backend must match the backend used when the binary cache was created"],
    ),
    PipelineStep(
        number=2,
        name="Load System Library",
        description=(
            "Load libQnnSystem.so to obtain system-level interfaces needed "
            "for reading the context binary cache."
        ),
        key_calls=["dlopen(libQnnSystem.so)", "QnnSystemInterface_getProviders()"],
        notes=["libQnnSystem.so is required for .bin loading (not needed for .so)"],
    ),
    PipelineStep(
        number=3,
        name="Initialize QNN Environment",
        description=(
            "Initialize Log, Backend, and Device. Note: context is NOT created "
            "here — it is loaded from the binary in step 4."
        ),
        key_calls=[
            "QnnLog_create()",
            "QnnBackend_create()",
            "QnnDevice_create()",
        ],
        notes=["No QnnContext_create() — context is loaded from binary cache"],
    ),
    PipelineStep(
        number=4,
        name="Load Context Binary Cache",
        description=(
            "Read model.bin and restore the QNN context with pre-composed and "
            "pre-finalized graphs. Skips the time-consuming graph composition step."
        ),
        key_calls=[
            "QnnContext_createFromBinary(backend, device, config, binary_buffer, size, context)",
        ],
        notes=[
            "This step saves considerable initialization time vs .so loading",
            "Binary buffer is backend-specific — binary from HTP cannot be loaded on CPU",
        ],
    ),
    PipelineStep(
        number=5,
        name="Execute Inference",
        description="Set input tensors, execute the graph, retrieve outputs.",
        key_calls=[
            "QnnGraph_execute(graph, inputs, n_inputs, outputs, n_outputs, profile, signal)",
        ],
        notes=["Graphs are already finalized — no compose/finalize needed"],
    ),
    PipelineStep(
        number=6,
        name="Free Resources",
        description="Release QNN resources.",
        key_calls=[
            "QnnContext_free()",
            "QnnDevice_free()",
            "QnnBackend_free()",
            "QnnLog_free()",
        ],
        notes=["No QnnGraph_free() needed — context free handles graphs"],
    ),
]


# ── .tflite / QNN Delegate Pipeline ──────────────────────────────────────────

TFLITE_DELEGATE_PIPELINE_STEPS: list[PipelineStep] = [
    PipelineStep(
        number=1,
        name="Load TFLite Model and Build Interpreter",
        description=(
            "Load the .tflite model file and create a TensorFlow Lite interpreter "
            "using the standard TFLite API."
        ),
        key_calls=[
            "tflite::FlatBufferModel::BuildFromFile(model_path)",
            "tflite::InterpreterBuilder(*model, resolver)(&interpreter)",
        ],
        notes=["Standard TFLite model loading — no QNN-specific calls yet"],
    ),
    PipelineStep(
        number=2,
        name="Register QNN Delegate",
        description=(
            "Create and register the QNN delegate with the TFLite interpreter. "
            "The delegate redirects supported ops to QNN hardware acceleration."
        ),
        key_calls=[
            "TfLiteQnnDelegateOptions options = TfLiteQnnDelegateOptionsDefault()",
            "TfLiteDelegate* delegate = TfLiteQnnDelegateCreate(&options)",
            "interpreter->ModifyGraphWithDelegate(delegate)",
        ],
        notes=[
            "Configure options to select backend (CPU/GPU/HTP)",
            "Unsupported ops fall back to TFLite CPU execution",
        ],
    ),
    PipelineStep(
        number=3,
        name="Execute Inference",
        description=(
            "Run inference using the standard TFLite Invoke() call. "
            "The QNN delegate handles hardware-accelerated ops transparently."
        ),
        key_calls=[
            "interpreter->AllocateTensors()",
            "// populate input tensors",
            "interpreter->Invoke()",
            "// read output tensors",
        ],
        notes=["Standard TFLite Invoke — QNN acceleration is transparent"],
    ),
    PipelineStep(
        number=4,
        name="Delete QNN Delegate",
        description="Clean up by deleting the QNN delegate.",
        key_calls=["TfLiteQnnDelegateDelete(delegate)"],
        notes=[],
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Registry
# ══════════════════════════════════════════════════════════════════════════════

INFERENCE_PIPELINES: dict[QnnModelFormat, list[PipelineStep]] = {
    QnnModelFormat.SO: SO_PIPELINE_STEPS,
    QnnModelFormat.BIN: BIN_PIPELINE_STEPS,
    QnnModelFormat.TFLITE: TFLITE_DELEGATE_PIPELINE_STEPS,
}


@dataclass
class QnnInferencePipeline:
    """Complete inference pipeline specification for a given model format."""
    model_format: QnnModelFormat
    steps: list[PipelineStep]
    required_libraries: list[str]
    key_difference: str  # What makes this pipeline unique

    def get_step(self, number: int) -> Optional[PipelineStep]:
        return next((s for s in self.steps if s.number == number), None)

    def all_key_calls(self) -> list[str]:
        """Flat list of all key API calls across all steps."""
        calls = []
        for step in self.steps:
            calls.extend(step.key_calls)
        return calls

    def describe(self) -> str:
        """Return human-readable pipeline description."""
        lines = [
            f"QNN Inference Pipeline — {self.model_format.value.upper()} format",
            "=" * 50,
            f"Required libraries: {', '.join(self.required_libraries)}",
            f"Key difference: {self.key_difference}",
            "",
        ]
        for step in self.steps:
            lines.append(f"Step {step.number}: {step.name}")
            lines.append(f"  {step.description}")
            if step.key_calls:
                lines.append(f"  Calls: {'; '.join(step.key_calls[:2])}")
            if step.notes:
                lines.append(f"  Notes: {step.notes[0]}")
            lines.append("")
        return "\n".join(lines)


PIPELINES = {
    QnnModelFormat.SO: QnnInferencePipeline(
        model_format=QnnModelFormat.SO,
        steps=SO_PIPELINE_STEPS,
        required_libraries=["libQnnBackend.so", "model.so"],
        key_difference=(
            "Loads graph composition function from model.so; "
            "most flexible but requires compose+finalize at every startup."
        ),
    ),
    QnnModelFormat.BIN: QnnInferencePipeline(
        model_format=QnnModelFormat.BIN,
        steps=BIN_PIPELINE_STEPS,
        required_libraries=["libQnnBackend.so", "libQnnSystem.so", "model.bin"],
        key_difference=(
            "Loads pre-compiled context binary cache (model.bin); "
            "skips graph composition → faster init. "
            "Binary is backend-specific."
        ),
    ),
    QnnModelFormat.TFLITE: QnnInferencePipeline(
        model_format=QnnModelFormat.TFLITE,
        steps=TFLITE_DELEGATE_PIPELINE_STEPS,
        required_libraries=["libQnnTFLiteDelegate.so", "model.tflite"],
        key_difference=(
            "Uses TFLite Interpreter + QNN Delegate; "
            "QNN acceleration is transparent to the TFLite API."
        ),
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Context Binary Cache (model.bin) Notes
# ══════════════════════════════════════════════════════════════════════════════

BIN_CACHE_NOTES: dict[str, Any] = {
    "purpose": (
        "Save considerable initialization time. Context init + graph composition "
        "+ finalization is time-consuming. model.bin caches the prepared and "
        "finalized context for fast subsequent loads."
    ),
    "generation": (
        "Generated by running the .so pipeline once with QnnBackend = QnnSaver "
        "(libQnnSaver.so), which intercepts the finalized context and writes it "
        "to a binary file."
    ),
    "backend_specific": (
        "The binary description is determined by each backend individually. "
        "A model.bin saved with HTP backend cannot be loaded with CPU backend. "
        "Backend MUST match between save and load."
    ),
    "equivalent_to_dlc": (
        "In SNPE/QAIRT terms, model.bin corresponds to the offline-prepared "
        "cached DLC generated by snpe-dlc-graph-prepare."
    ),
    "loading_api": "QnnContext_createFromBinary()",
    "saving_api": (
        "Use QnnSaver backend (libQnnSaver.so) during initial .so execution; "
        "it serializes the context instead of executing it."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

QNN_INFERENCE_NOTES: dict[str, Any] = {
    "description": (
        "QNN SDK C++ API for AI model inference. Three supported formats: "
        ".so (composed graph), .bin (cached context), .tflite (via delegate)."
    ),
    "api_language": "C++ (C-style for portability across platforms)",
    "supported_formats": {
        "so": "Shared object with QnnModel_composeGraphs(); most flexible",
        "bin": "Context binary cache; fast init, backend-specific",
        "tflite": "TFLite via QNN Delegate; transparent acceleration",
    },
    "backend_libraries": {
        "cpu": "libQnnCpu.so",
        "gpu": "libQnnGpu.so",
        "htp": "libQnnHtp.so",
        "dsp": "libQnnDsp.so",
        "system": "libQnnSystem.so (required for .bin loading)",
        "saver": "libQnnSaver.so (for saving context binary cache)",
    },
    "so_vs_bin_key_difference": (
        ".so: load backend + model.so → compose graph → finalize → execute. "
        ".bin: load backend + system → load cached context → execute (skips compose+finalize)."
    ),
    "initialization_order": [
        "QnnLog (optional, can be before QnnBackend)",
        "QnnBackend",
        "QnnDevice",
        "QnnContext (create new OR load from binary)",
        "QnnGraph (compose + finalize — .so only; already done in .bin)",
    ],
    "bin_cache_notes": BIN_CACHE_NOTES,
    "qnn_sample_app": (
        "QNN Sample App Tutorial: Create and build a sample C++ application. "
        "See QNN SDK documentation for full source code."
    ),
    "tflite_references": [
        "TensorFlow-lite minimal example",
        "QNN Delegate Tutorial",
    ],
    "recommended_alternative": (
        "For most use cases, use qrb_inference_manager instead of raw QNN API — "
        "it abstracts the initialization complexity."
    ),
}
