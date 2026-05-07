"""QUAD Serve — Request and Response models for inference."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InferenceRequest:
    """A single inference request to a model.

    Attributes:
        model_name: Target model name.
        inputs: Dictionary mapping input names to numpy arrays.
        request_id: Unique request identifier (auto-generated if not provided).
        priority: Request priority (0=normal, higher=more urgent).
    """

    model_name: str
    inputs: dict[str, np.ndarray]
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0

    def __post_init__(self):
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.inputs:
            raise ValueError("inputs must not be empty")


@dataclass
class InferenceResponse:
    """Response from an inference request.

    Attributes:
        outputs: Dictionary mapping output names to numpy arrays.
        latency_ms: Inference latency in milliseconds.
        model_name: Name of the model that produced this response.
        request_id: ID of the corresponding request.
    """

    outputs: dict[str, np.ndarray]
    latency_ms: float
    model_name: str
    request_id: str = ""

    @property
    def output_names(self) -> list[str]:
        """List of output tensor names."""
        return list(self.outputs.keys())

    @property
    def num_outputs(self) -> int:
        """Number of output tensors."""
        return len(self.outputs)


@dataclass
class BatchRequest:
    """A batch of inference requests for efficient processing.

    Attributes:
        requests: List of individual inference requests.
        max_wait_ms: Maximum time to wait for batch to fill before executing.
    """

    requests: list[InferenceRequest]
    max_wait_ms: float = 10.0

    @property
    def batch_size(self) -> int:
        """Number of requests in the batch."""
        return len(self.requests)

    @property
    def model_name(self) -> str:
        """Target model (all requests in batch must target same model)."""
        if not self.requests:
            return ""
        return self.requests[0].model_name

    def validate(self) -> bool:
        """Ensure all requests target the same model."""
        if not self.requests:
            return False
        model = self.requests[0].model_name
        return all(r.model_name == model for r in self.requests)
