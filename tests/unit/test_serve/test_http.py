"""Tests for the FastAPI HTTP layer (T1.2).

Uses httpx + FastAPI's TestClient (in-process, no real network).
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

# Skip the entire module if FastAPI isn't installed (it's in [real] extras)
fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from quad.serve.http import build_app
from quad.serve.server import ModelServer


@pytest.fixture
def server() -> ModelServer:
    s = ModelServer()
    s.start()
    return s


@pytest.fixture
def client(server: ModelServer) -> TestClient:
    return TestClient(build_app(server))


# ─── Tensor JSON round-trip ─────────────────────────────────────────────────


def _encode_array(arr: np.ndarray) -> dict:
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def _decode_array(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data_b64"])
    return np.frombuffer(raw, dtype=np.dtype(payload["dtype"])).reshape(payload["shape"])


# ─── Health & metrics ──────────────────────────────────────────────────────


class TestHealthAndMetrics:
    def test_health_when_no_models(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        # No models → "degraded" (server up but nothing to serve)
        assert body["status"] == "degraded"
        assert body["models_loaded"] == 0
        assert body["uptime_s"] >= 0

    def test_health_after_loading_model(
        self, client: TestClient, server: ModelServer
    ) -> None:
        server.load_model("m", "fake.qbin", device="npu")
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "healthy"
        assert body["models_loaded"] == 1

    def test_metrics_initial_state(self, client: TestClient) -> None:
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["total_requests"] == 0
        assert body["avg_latency_ms"] == 0.0


# ─── Model management ──────────────────────────────────────────────────────


class TestModelManagement:
    def test_list_models_empty(self, client: TestClient) -> None:
        r = client.get("/models")
        assert r.status_code == 200
        assert r.json() == []

    def test_load_model(self, client: TestClient) -> None:
        r = client.post(
            "/models/foo/load",
            json={"path": "fake.qbin", "device": "npu", "version": 1},
        )
        assert r.status_code == 201
        assert r.json()["model"] == "foo"

        # Now appears in list
        r2 = client.get("/models")
        assert len(r2.json()) == 1
        assert r2.json()[0]["name"] == "foo"

    def test_load_duplicate_returns_409(self, client: TestClient) -> None:
        client.post("/models/foo/load", json={"path": "x", "device": "npu"})
        r = client.post("/models/foo/load", json={"path": "x", "device": "npu"})
        assert r.status_code == 409

    def test_unload_model(self, client: TestClient) -> None:
        client.post("/models/foo/load", json={"path": "x", "device": "npu"})
        r = client.delete("/models/foo")
        assert r.status_code == 204
        assert client.get("/models").json() == []

    def test_unload_unknown_returns_404(self, client: TestClient) -> None:
        r = client.delete("/models/missing")
        assert r.status_code == 404


# ─── Inference ─────────────────────────────────────────────────────────────


class TestInference:
    def test_single_inference_returns_outputs(
        self, client: TestClient, server: ModelServer
    ) -> None:
        server.load_model("m", "fake.qbin", device="npu")
        inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
        body = {
            "model_name": "m",
            "inputs": {"input": _encode_array(inp)},
        }
        r = client.post("/infer", json=body)
        assert r.status_code == 200
        resp = r.json()
        assert resp["model_name"] == "m"
        assert "request_id" in resp
        assert resp["latency_ms"] > 0
        assert "output" in resp["outputs"]
        # Decode output and check shape
        out = _decode_array(resp["outputs"]["output"])
        # Mock backend produces (batch, 1000) for 4D image inputs
        assert out.shape[0] == 1
        assert out.shape[1] == 1000

    def test_inference_unknown_model_returns_404(self, client: TestClient) -> None:
        body = {
            "model_name": "missing",
            "inputs": {"x": _encode_array(np.zeros(3))},
        }
        r = client.post("/infer", json=body)
        assert r.status_code == 404

    def test_inference_empty_inputs_returns_400(
        self, client: TestClient, server: ModelServer
    ) -> None:
        server.load_model("m", "fake.qbin", device="npu")
        r = client.post("/infer", json={"model_name": "m", "inputs": {}})
        assert r.status_code == 400

    def test_batch_inference(
        self, client: TestClient, server: ModelServer
    ) -> None:
        server.load_model("m", "fake.qbin", device="npu")
        batch_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        body = {
            "model_name": "m",
            "batch": [
                {"input": _encode_array(batch_input)},
                {"input": _encode_array(batch_input)},
                {"input": _encode_array(batch_input)},
            ],
        }
        r = client.post("/infer/batch", json=body)
        assert r.status_code == 200
        results = r.json()
        assert len(results) == 3
        # Each gets a unique request id
        assert len({res["request_id"] for res in results}) == 3

    def test_metrics_incremented_after_inference(
        self, client: TestClient, server: ModelServer
    ) -> None:
        server.load_model("m", "fake.qbin", device="npu")
        body = {
            "model_name": "m",
            "inputs": {"input": _encode_array(np.zeros((1, 3, 224, 224), dtype=np.float32))},
        }
        client.post("/infer", json=body)
        r = client.get("/metrics")
        assert r.json()["total_requests"] == 1


# ─── build_app contract ────────────────────────────────────────────────────


class TestBuildApp:
    def test_build_app_creates_running_server_when_not_started(self) -> None:
        s = ModelServer()
        # Don't call s.start() — build_app should auto-start
        assert not s.is_running
        _ = build_app(s)
        assert s.is_running

    def test_build_app_reuses_existing_server(self) -> None:
        s = ModelServer()
        s.start()
        s.load_model("preexisting", "x.qbin")
        client = TestClient(build_app(s))
        assert len(client.get("/models").json()) == 1
