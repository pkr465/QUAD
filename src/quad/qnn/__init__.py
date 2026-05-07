"""QUAD QNN — QNN SDK API reference, pipeline definitions, and code generation."""

from quad.qnn.api import (
    QnnApiComponent,
    QnnCategory,
    QNN_API_COMPONENTS,
    get_components_by_category,
    get_backend_specialized_components,
    get_pre_backend_components,
)
from quad.qnn.inference_pipeline import (
    BIN_CACHE_NOTES,
    BIN_PIPELINE_STEPS,
    INFERENCE_PIPELINES,
    PIPELINES,
    PipelineStep,
    QnnBackendLib,
    QnnInferencePipeline,
    QnnModelFormat,
    QNN_INFERENCE_NOTES,
    SO_PIPELINE_STEPS,
    TFLITE_DELEGATE_PIPELINE_STEPS,
)

__all__ = [
    # API components
    "QnnApiComponent",
    "QnnCategory",
    "QNN_API_COMPONENTS",
    "get_components_by_category",
    "get_backend_specialized_components",
    "get_pre_backend_components",
    # Inference pipelines
    "BIN_CACHE_NOTES",
    "BIN_PIPELINE_STEPS",
    "INFERENCE_PIPELINES",
    "PIPELINES",
    "PipelineStep",
    "QnnBackendLib",
    "QnnInferencePipeline",
    "QnnModelFormat",
    "QNN_INFERENCE_NOTES",
    "SO_PIPELINE_STEPS",
    "TFLITE_DELEGATE_PIPELINE_STEPS",
]
