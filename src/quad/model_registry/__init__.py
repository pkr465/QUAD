"""QUAD model registry — production ONNX provisioning for plans.

Public API:

    >>> from quad.model_registry import (
    ...     ModelEntry, fetch_model, list_models, list_for_plan,
    ...     resolve_model_path, register_entry,
    ... )
    >>> path = fetch_model("mobilenetv2")          # downloads to ~/.quad/models/
    >>> path = resolve_model_path("llama3_8b_prefill")   # reads $LLAMA3_8B_PREFILL_ONNX

Adding a model for a future plan is a one-line manifest edit (or a
``register_entry()`` call from code) — no other QUAD changes needed.
"""
from quad.model_registry.manifest import (
    ModelEntry,
    list_for_plan,
    list_models,
    register_entry,
    registry_path,
    resolve_entry,
)
from quad.model_registry.fetcher import (
    ModelFetchError,
    fetch_model,
    resolve_model_path,
)

__all__ = [
    "ModelEntry",
    "ModelFetchError",
    "fetch_model",
    "list_for_plan",
    "list_models",
    "register_entry",
    "registry_path",
    "resolve_entry",
    "resolve_model_path",
]
