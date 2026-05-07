"""QUAD Serve — Main Inference Server.

Production inference server for hosting QUAD models, equivalent to NVIDIA Triton
but optimized for Qualcomm on-device deployment.

Usage:
    server = ModelServer(port=8080, power_budget_mw=10000)
    server.load_model("yolo", "models/yolo.qbin", device="npu:0")
    server.load_model("resnet", "models/resnet.qbin", device="npu:1")
    server.start()  # Blocking
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quad.runtime.device import Device
from quad.serve.request import InferenceRequest, InferenceResponse, BatchRequest


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """Configuration for the QUAD inference server.

    Attributes:
        port: HTTP port to listen on.
        host: Host address to bind to.
        power_budget_mw: Total power budget in milliwatts (None = unlimited).
        max_batch_size: Maximum batch size for dynamic batching.
        model_repo_path: Path to model repository directory.
    """

    port: int = 8080
    host: str = "0.0.0.0"
    power_budget_mw: int | None = None
    max_batch_size: int = 32
    model_repo_path: str = "./models"


@dataclass
class ModelInfo:
    """Information about a loaded model.

    Attributes:
        name: Model name.
        path: Path to model binary.
        device: Device the model is loaded on.
        version: Model version.
        loaded_at: Timestamp when the model was loaded.
        num_inferences: Number of inferences performed.
    """

    name: str
    path: str
    device: str
    version: int = 1
    loaded_at: float = field(default_factory=time.time)
    num_inferences: int = 0


@dataclass
class HealthStatus:
    """Server health status.

    Attributes:
        status: One of "healthy", "degraded", "unhealthy".
        models_loaded: Number of models currently loaded.
        uptime_s: Server uptime in seconds.
    """

    status: str  # "healthy", "degraded", "unhealthy"
    models_loaded: int = 0
    uptime_s: float = 0.0


@dataclass
class ServerMetrics:
    """Server performance metrics.

    Attributes:
        total_requests: Total number of inference requests served.
        avg_latency_ms: Average inference latency in milliseconds.
        p99_latency_ms: 99th percentile latency.
        throughput_rps: Throughput in requests per second.
        power_mw: Current estimated power consumption.
    """

    total_requests: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    power_mw: float = 0.0


# ---------------------------------------------------------------------------
# ModelServer
# ---------------------------------------------------------------------------


class ModelServer:
    """QUAD production inference server.

    Hosts models on NPU/GPU/CPU devices and handles inference requests
    with dynamic batching, power management, and health monitoring.

    Usage:
        server = ModelServer(port=8080, power_budget_mw=10000)
        server.load_model("yolo", "models/yolo.qbin", device="npu:0")
        server.load_model("resnet", "models/resnet.qbin", device="npu:1")
        server.start()
    """

    def __init__(
        self,
        port: int = 8080,
        host: str = "0.0.0.0",
        power_budget_mw: int | None = None,
    ):
        self._config = ServerConfig(port=port, host=host, power_budget_mw=power_budget_mw)
        self._models: dict[str, ModelInfo] = {}
        self._running = False
        self._start_time: float | None = None
        self._total_inferences: int = 0
        self._latencies: list[float] = []

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(
        self,
        name: str,
        path: str,
        device: str = "npu",
        version: int = 1,
    ) -> None:
        """Load a model into the server for inference.

        Args:
            name: Model name identifier.
            path: Path to model binary (.qbin).
            device: Target device (e.g. "npu", "npu:0", "gpu", "cpu").
            version: Model version number.

        Raises:
            ValueError: If model name is already loaded.
        """
        if name in self._models:
            raise ValueError(f"Model '{name}' is already loaded. Unload first.")

        # Validate device string
        _ = Device(device)

        self._models[name] = ModelInfo(
            name=name,
            path=path,
            device=device,
            version=version,
            loaded_at=time.time(),
            num_inferences=0,
        )

    def unload_model(self, name: str) -> None:
        """Unload a model from the server.

        Args:
            name: Model name to unload.

        Raises:
            KeyError: If model is not loaded.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not loaded")
        del self._models[name]

    def list_models(self) -> list[ModelInfo]:
        """List all loaded models.

        Returns:
            List of ModelInfo objects for all loaded models.
        """
        return list(self._models.values())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, model_name: str, inputs: dict[str, np.ndarray]) -> InferenceResponse:
        """Run inference on a loaded model.

        Args:
            model_name: Name of the loaded model.
            inputs: Dictionary mapping input names to numpy arrays.

        Returns:
            InferenceResponse with output tensors and timing info.

        Raises:
            KeyError: If model is not loaded.
            ValueError: If inputs are empty.
        """
        if model_name not in self._models:
            raise KeyError(f"Model '{model_name}' is not loaded")
        if not inputs:
            raise ValueError("inputs must not be empty")

        model_info = self._models[model_name]
        start = time.perf_counter()

        # Mock inference: produce output based on input shapes
        outputs = {}
        for input_name, input_array in inputs.items():
            # Simulate model output (e.g., classification logits)
            if len(input_array.shape) == 4:
                # Image input -> classification output
                batch_size = input_array.shape[0]
                outputs["output"] = np.random.randn(batch_size, 1000).astype(np.float32)
            elif len(input_array.shape) == 2:
                # Sequence input -> sequence output
                outputs["output"] = np.random.randn(*input_array.shape).astype(np.float32)
            else:
                outputs["output"] = np.random.randn(*input_array.shape).astype(np.float32)

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Update stats
        model_info.num_inferences += 1
        self._total_inferences += 1
        self._latencies.append(elapsed_ms)

        return InferenceResponse(
            outputs=outputs,
            latency_ms=elapsed_ms,
            model_name=model_name,
            request_id=str(uuid.uuid4()),
        )

    def infer_batch(
        self, model_name: str, batch: list[dict[str, np.ndarray]]
    ) -> list[InferenceResponse]:
        """Run batch inference on a loaded model.

        Args:
            model_name: Name of the loaded model.
            batch: List of input dictionaries.

        Returns:
            List of InferenceResponse objects, one per input.

        Raises:
            KeyError: If model is not loaded.
            ValueError: If batch is empty.
        """
        if not batch:
            raise ValueError("batch must not be empty")
        return [self.infer(model_name, inputs) for inputs in batch]

    # ------------------------------------------------------------------
    # Health & Metrics
    # ------------------------------------------------------------------

    def health(self) -> HealthStatus:
        """Get server health status.

        Returns:
            HealthStatus with current server state.
        """
        uptime = 0.0
        if self._start_time is not None:
            uptime = time.time() - self._start_time

        num_models = len(self._models)

        if not self._running:
            status = "unhealthy"
        elif num_models == 0:
            status = "degraded"
        else:
            status = "healthy"

        return HealthStatus(
            status=status,
            models_loaded=num_models,
            uptime_s=uptime,
        )

    def metrics(self) -> ServerMetrics:
        """Get server performance metrics.

        Returns:
            ServerMetrics with aggregated statistics.
        """
        avg_latency = 0.0
        p99_latency = 0.0
        throughput = 0.0

        if self._latencies:
            avg_latency = sum(self._latencies) / len(self._latencies)
            sorted_latencies = sorted(self._latencies)
            p99_idx = int(len(sorted_latencies) * 0.99)
            p99_latency = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]

        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                throughput = self._total_inferences / elapsed

        # Estimate power based on loaded models
        power_mw = sum(
            Device(m.device).power_typical_mw for m in self._models.values()
        )

        return ServerMetrics(
            total_requests=self._total_inferences,
            avg_latency_ms=avg_latency,
            p99_latency_ms=p99_latency,
            throughput_rps=throughput,
            power_mw=power_mw,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the inference server.

        In mock mode, this sets the server to running state without blocking.
        In production, this would start the HTTP server event loop.
        """
        self._running = True
        self._start_time = time.time()

    def stop(self) -> None:
        """Stop the inference server and release resources."""
        self._running = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the server is currently running."""
        return self._running

    @property
    def num_models(self) -> int:
        """Number of models currently loaded."""
        return len(self._models)

    @property
    def total_inferences(self) -> int:
        """Total number of inferences served since start."""
        return self._total_inferences

    @property
    def config(self) -> ServerConfig:
        """Server configuration."""
        return self._config
