"""HTTP/FastAPI binding for ModelServer.

Closes GAP_ANALYSIS T1.2 (partial): the previous ModelServer.start()
was a no-op (just flipped a boolean). This module adds a FastAPI app
factory plus a uvicorn-based start_http() helper, so `quad serve` can
actually listen for inference requests.

The HTTP layer is a *thin shell* over ModelServer — all the actual
inference / model-management logic stays in server.py.

FastAPI + uvicorn are in the [real] extras of pyproject.toml. If a
caller imports this module without those packages installed, the
import falls through cleanly until ``build_app()`` is called; then we
raise an ImportError with a clear next-step.

Endpoints:
  POST /infer           — single inference
  POST /infer/batch     — batched inference
  GET  /health          — liveness + loaded-models count
  GET  /metrics         — aggregate stats (latency, throughput, power)
  GET  /models          — list loaded models
  POST /models/{name}/load    — load a model
  DELETE /models/{name}       — unload a model
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from quad.serve.server import ModelServer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # avoid hard import of fastapi at module load
    from fastapi import FastAPI as _FastAPI


# ─── Pydantic schemas (module level so FastAPI can see them) ─────────────────


class TensorJSON(BaseModel):
    """Numpy ndarray serialised for JSON transport."""

    shape: list[int]
    dtype: str
    data_b64: str = Field(
        description="base64-encoded raw little-endian bytes of the tensor"
    )

    @classmethod
    def from_ndarray(cls, arr: np.ndarray) -> "TensorJSON":
        return cls(
            shape=list(arr.shape),
            dtype=str(arr.dtype),
            data_b64=base64.b64encode(arr.tobytes()).decode("ascii"),
        )

    def to_ndarray(self) -> np.ndarray:
        raw = base64.b64decode(self.data_b64)
        return np.frombuffer(raw, dtype=np.dtype(self.dtype)).reshape(self.shape)


class InferRequest(BaseModel):
    model_name: str
    inputs: dict[str, TensorJSON]


class InferBatchRequest(BaseModel):
    model_name: str
    batch: list[dict[str, TensorJSON]]


class InferResponse(BaseModel):
    model_name: str
    request_id: str
    latency_ms: float
    outputs: dict[str, TensorJSON]


class LoadModelRequest(BaseModel):
    path: str
    device: str = "npu"
    version: int = 1


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    uptime_s: float


class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    power_mw: float


class ModelEntry(BaseModel):
    name: str
    path: str
    device: str
    version: int
    num_inferences: int


# ─── FastAPI app factory ─────────────────────────────────────────────────────


def build_app(server: ModelServer | None = None) -> "_FastAPI":
    """Construct a FastAPI app bound to a given ModelServer.

    Args:
        server: An existing ModelServer (so callers can preload models
            before the HTTP layer starts). Defaults to a fresh server.

    Returns:
        FastAPI app ready to be served by uvicorn.

    Raises:
        ImportError: with a clear next-step message if FastAPI isn't
            installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as e:
        raise ImportError(
            "FastAPI not installed — cannot build HTTP app. "
            "Install via: pip install fastapi uvicorn  "
            "(or pip install -e .[real])"
        ) from e

    state = server if server is not None else ModelServer()
    if not state.is_running:
        state.start()

    app = FastAPI(
        title="QUAD Inference Server",
        description="Production inference server for Qualcomm AI hardware.",
        version="0.4.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        h = state.health()
        return HealthResponse(
            status=h.status,
            models_loaded=h.models_loaded,
            uptime_s=h.uptime_s,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        m = state.metrics()
        return MetricsResponse(
            total_requests=m.total_requests,
            avg_latency_ms=m.avg_latency_ms,
            p99_latency_ms=m.p99_latency_ms,
            throughput_rps=m.throughput_rps,
            power_mw=m.power_mw,
        )

    @app.get("/models", response_model=list[ModelEntry])
    def list_models() -> list[ModelEntry]:
        return [
            ModelEntry(
                name=m.name,
                path=m.path,
                device=m.device,
                version=m.version,
                num_inferences=m.num_inferences,
            )
            for m in state.list_models()
        ]

    @app.post("/models/{name}/load", status_code=201)
    def load_model(name: str, body: LoadModelRequest) -> dict[str, str]:
        try:
            state.load_model(name, body.path, device=body.device, version=body.version)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"status": "loaded", "model": name}

    @app.delete("/models/{name}", status_code=204)
    def unload_model(name: str) -> None:
        try:
            state.unload_model(name)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest) -> InferResponse:
        try:
            inputs = {k: v.to_ndarray() for k, v in req.inputs.items()}
            result = state.infer(req.model_name, inputs)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return InferResponse(
            model_name=result.model_name,
            request_id=result.request_id,
            latency_ms=result.latency_ms,
            outputs={k: TensorJSON.from_ndarray(v) for k, v in result.outputs.items()},
        )

    @app.post("/infer/batch", response_model=list[InferResponse])
    def infer_batch(req: InferBatchRequest) -> list[InferResponse]:
        try:
            batch = [{k: v.to_ndarray() for k, v in inp.items()} for inp in req.batch]
            results = state.infer_batch(req.model_name, batch)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return [
            InferResponse(
                model_name=r.model_name,
                request_id=r.request_id,
                latency_ms=r.latency_ms,
                outputs={k: TensorJSON.from_ndarray(v) for k, v in r.outputs.items()},
            )
            for r in results
        ]

    return app


# ─── start_http helper (uvicorn entry point) ────────────────────────────────


def start_http(
    server: ModelServer | None = None,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Run the FastAPI app under uvicorn (blocking).

    Used by ``quad serve``. For programmatic / async usage, build the
    app via ``build_app()`` and run uvicorn yourself.

    Args:
        server: An optional ModelServer (so callers can preload models)
        host: Bind host (default 0.0.0.0)
        port: Bind port (default 8080)

    Raises:
        ImportError: if uvicorn is not installed.
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "uvicorn not installed — cannot serve HTTP. "
            "Install via: pip install uvicorn  (or pip install -e .[real])"
        ) from e

    app = build_app(server)
    logger.info(
        "quad_serve_http_starting",
        extra={"host": host, "port": port, "models_loaded": (server.num_models if server else 0)},
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
