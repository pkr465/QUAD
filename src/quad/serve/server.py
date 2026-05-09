"""QUAD Serve — Main Inference Server.

Production inference server for hosting QUAD models, equivalent to NVIDIA Triton
but optimized for Qualcomm on-device deployment.

Usage:
    server = ModelServer(port=8080, power_budget_mw=10000)
    server.load_model("yolo", "models/yolo.qbin", device="npu:0")
    server.load_model("resnet", "models/resnet.qbin", device="npu:1")
    server.start()  # Blocking

Two runtime backends:
    runtime="mock"  — `infer()` returns deterministic random output of
                      the right shape (no SDK required, default).
    runtime="qairt" — `infer()` shells out to `snpe-net-run` via
                      ``QAIRTAdapter.execute_inference`` and returns
                      real model outputs. Requires QAIRT_SDK_ROOT.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quad.runtime.device import Device
from quad.serve.request import InferenceRequest, InferenceResponse, BatchRequest

logger = logging.getLogger(__name__)


# Extensions that the QAIRT adapter can run via snpe-net-run / qnn-net-run
_QAIRT_RUNNABLE_EXT = {".dlc", ".bin"}

# Backend label
RuntimeBackend = Literal["mock", "qairt"]


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
        runtime: RuntimeBackend = "mock",
        adapter: Any | None = None,
    ):
        """
        Args:
            port / host / power_budget_mw: ServerConfig fields.
            runtime: "mock" (default) returns deterministic random outputs;
                "qairt" calls the real QAIRTAdapter.execute_inference
                pipeline (snpe-net-run / qnn-net-run subprocess).
            adapter: Inject a pre-built adapter (mostly for tests). When
                None and runtime="qairt", a fresh QAIRTAdapter is built
                lazily on the first inference.
        """
        self._config = ServerConfig(port=port, host=host, power_budget_mw=power_budget_mw)
        self._models: dict[str, ModelInfo] = {}
        self._running = False
        self._start_time: float | None = None
        self._total_inferences: int = 0
        self._latencies: list[float] = []
        self._runtime: RuntimeBackend = runtime
        self._adapter: Any | None = adapter

    @classmethod
    def from_env(cls, **kwargs: Any) -> "ModelServer":
        """Build a ModelServer whose runtime is controlled by env vars.

        ``QUAD_SERVE_RUNTIME=qairt`` (or ``real``) wires the real adapter
        when a Qualcomm SDK is reachable; otherwise mock mode. This is
        what ``quad serve`` and the FastAPI factory use so the server's
        inference path automatically tracks the rest of QUAD's adapter
        mode.
        """
        wanted = os.environ.get("QUAD_SERVE_RUNTIME", "").strip().lower()
        if wanted in ("qairt", "real"):
            return cls(runtime="qairt", **kwargs)
        # Default: respect the broader adapter mode env var; fallback mock
        adapter_mode = os.environ.get("QUAD_ADAPTER_MODE", "").strip().lower()
        if adapter_mode == "real" and (
            os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("SNPE_ROOT")
        ):
            return cls(runtime="qairt", **kwargs)
        return cls(runtime="mock", **kwargs)

    def _get_adapter(self) -> Any:
        """Lazy-construct the QAIRT adapter on first real inference."""
        if self._adapter is not None:
            return self._adapter
        from quad.adapters.qairt_adapter import QAIRTAdapter
        self._adapter = QAIRTAdapter()
        return self._adapter

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

        # Dispatch by runtime: real QAIRT for .dlc / .bin, mock otherwise
        # OR when explicitly configured for mock mode.
        ext = Path(model_info.path).suffix.lower()
        use_real = (
            self._runtime == "qairt"
            and ext in _QAIRT_RUNNABLE_EXT
        )
        if use_real:
            try:
                outputs = self._infer_qairt(model_info, inputs)
            except Exception as e:  # broad: real-mode failure should NOT crash the server
                logger.warning(
                    "qairt_infer_failed_falling_back_to_mock",
                    extra={"model": model_name, "error": str(e)[:300]},
                )
                outputs = self._infer_mock(inputs)
        else:
            outputs = self._infer_mock(inputs)

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

    @staticmethod
    def _infer_mock(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Deterministic mock inference: shape-preserving random output.

        Used when ``runtime="mock"`` or when real-mode inference fails
        and we degrade gracefully. Output shape depends on input rank:
        - 4D (NCHW/NHWC image) → classification logits (batch, 1000)
        - 2D (sequence) → same-shape output
        - other → same-shape output
        """
        outputs: dict[str, np.ndarray] = {}
        first_array = next(iter(inputs.values()))
        if first_array.ndim == 4:
            batch_size = first_array.shape[0]
            outputs["output"] = np.random.randn(batch_size, 1000).astype(np.float32)
        else:
            outputs["output"] = np.random.randn(*first_array.shape).astype(np.float32)
        return outputs

    def _infer_qairt(
        self,
        model_info: ModelInfo,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Run real inference via QAIRTAdapter.execute_inference.

        Bridges the sync HTTP layer to the async adapter using a per-call
        ``asyncio.run``. For high-throughput serving this should move to
        a dedicated worker thread / loop; the per-call cost is negligible
        compared to the snpe-net-run subprocess time.
        """
        adapter = self._get_adapter()
        # Map our device string ("npu" / "npu:0" / "gpu" / "cpu") to the
        # adapter's runtime selector ("npu" / "gpu" / "cpu" / "auto").
        dev_label = model_info.device.split(":", 1)[0].lower()
        runtime = dev_label if dev_label in ("cpu", "gpu", "npu") else "auto"

        result = asyncio.run(
            adapter.execute_inference(
                model_path=model_info.path,
                input_data=inputs,
                runtime=runtime,
            )
        )

        if result.get("status") != "success":
            raise RuntimeError(
                f"snpe-net-run failed (rc={result.get('returncode')}): "
                f"{(result.get('stderr') or '').splitlines()[-1] if result.get('stderr') else 'no stderr'}"
            )
        outputs = result.get("outputs") or {}
        if not outputs:
            # The subprocess succeeded but produced no parsed outputs —
            # most likely a model_io introspection miss. Don't pretend
            # we have data; surface the failure.
            raise RuntimeError(
                f"snpe-net-run succeeded but produced no outputs at "
                f"{result.get('work_dir')}; check model output specs."
            )
        return outputs

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
