"""Tests for Sprint 2: ModelServer routes .dlc/.bin inference through
the QAIRT adapter when runtime='qairt'.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quad.serve.server import ModelServer


# ─── A fake QAIRT adapter so we don't need a real SDK ──────────────────────


class _FakeAdapter:
    def __init__(self, *, status: str = "success", outputs: dict[str, np.ndarray] | None = None) -> None:
        self._status = status
        self._outputs = outputs or {"output": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}
        self.calls: list[dict[str, Any]] = []

    async def execute_inference(
        self,
        *,
        model_path: str,
        input_data: Any = None,
        runtime: str = "auto",
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.calls.append(
            dict(model_path=model_path, runtime=runtime, has_input=input_data is not None)
        )
        return {
            "status": self._status,
            "returncode": 0 if self._status == "success" else 1,
            "outputs": self._outputs if self._status == "success" else {},
            "stdout": "",
            "stderr": "" if self._status == "success" else "snpe-net-run failed",
            "model_io": {},
            "work_dir": "/tmp/fake",
        }


# ─── runtime dispatch ──────────────────────────────────────────────────────


class TestRuntimeDispatch:
    def test_dlc_routes_to_real_adapter(self, tmp_path: Path) -> None:
        adapter = _FakeAdapter()
        s = ModelServer(runtime="qairt", adapter=adapter)
        s.start()
        model_path = tmp_path / "x.dlc"
        model_path.write_bytes(b"")
        s.load_model("m", str(model_path), device="npu")

        out = s.infer("m", {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})

        assert len(adapter.calls) == 1
        assert adapter.calls[0]["runtime"] == "npu"
        assert adapter.calls[0]["model_path"] == str(model_path)
        # Real outputs propagate to the response (not np.random)
        np.testing.assert_array_equal(out.outputs["output"], adapter._outputs["output"])

    def test_qbin_routes_to_mock_when_extension_unsupported(self, tmp_path: Path) -> None:
        # .qbin is QUAD's IR-compiled binary — QAIRT can't run it directly,
        # so even in qairt mode we fall through to mock.
        adapter = _FakeAdapter()
        s = ModelServer(runtime="qairt", adapter=adapter)
        s.start()
        model_path = tmp_path / "y.qbin"
        model_path.write_bytes(b"")
        s.load_model("m", str(model_path), device="npu")

        out = s.infer("m", {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})

        assert adapter.calls == []  # adapter never invoked
        assert out.outputs["output"].shape == (1, 1000)  # mock classification head

    def test_mock_runtime_always_uses_mock(self, tmp_path: Path) -> None:
        adapter = _FakeAdapter()
        s = ModelServer(runtime="mock", adapter=adapter)
        s.start()
        model_path = tmp_path / "x.dlc"
        model_path.write_bytes(b"")
        s.load_model("m", str(model_path), device="npu")

        out = s.infer("m", {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})

        assert adapter.calls == []
        assert out.outputs["output"].shape == (1, 1000)

    def test_real_failure_falls_back_to_mock(self, tmp_path: Path) -> None:
        adapter = _FakeAdapter(status="error")
        s = ModelServer(runtime="qairt", adapter=adapter)
        s.start()
        model_path = tmp_path / "x.dlc"
        model_path.write_bytes(b"")
        s.load_model("m", str(model_path), device="npu")

        # Should not raise — falls back to mock and logs the failure.
        out = s.infer("m", {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)})
        assert len(adapter.calls) == 1
        assert out.outputs["output"].shape == (1, 1000)


# ─── env factory ───────────────────────────────────────────────────────────


class TestFromEnv:
    def test_explicit_serve_runtime_qairt(self, monkeypatch) -> None:
        monkeypatch.setenv("QUAD_SERVE_RUNTIME", "qairt")
        s = ModelServer.from_env()
        assert s._runtime == "qairt"

    def test_no_env_defaults_to_mock(self, monkeypatch) -> None:
        monkeypatch.delenv("QUAD_SERVE_RUNTIME", raising=False)
        monkeypatch.delenv("QUAD_ADAPTER_MODE", raising=False)
        s = ModelServer.from_env()
        assert s._runtime == "mock"

    def test_real_adapter_mode_with_sdk_root_picks_qairt(self, monkeypatch, tmp_path: Path) -> None:
        monkeypatch.delenv("QUAD_SERVE_RUNTIME", raising=False)
        monkeypatch.setenv("QUAD_ADAPTER_MODE", "real")
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(tmp_path))
        s = ModelServer.from_env()
        assert s._runtime == "qairt"

    def test_real_adapter_mode_without_sdk_root_falls_back_to_mock(
        self, monkeypatch
    ) -> None:
        monkeypatch.delenv("QUAD_SERVE_RUNTIME", raising=False)
        monkeypatch.setenv("QUAD_ADAPTER_MODE", "real")
        monkeypatch.delenv("QAIRT_SDK_ROOT", raising=False)
        monkeypatch.delenv("SNPE_ROOT", raising=False)
        s = ModelServer.from_env()
        assert s._runtime == "mock"


# ─── HTTP layer health endpoint exposes runtime ────────────────────────────


class TestHealthExposesRuntime:
    def test_health_reports_runtime_label(self) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from quad.serve.http import build_app

        s = ModelServer(runtime="qairt", adapter=_FakeAdapter())
        s.start()
        client = TestClient(build_app(s))
        body = client.get("/health").json()
        assert body["runtime"] == "qairt"
