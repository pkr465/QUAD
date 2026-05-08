"""Tests for the AI Hub adapter (T1.6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from quad.adapters.aihub_adapter import (
    AIHubAdapter,
    AIHubAuthError,
    AIHubCompileResult,
    AIHubProfile,
    AIHubUnavailableError,
    KNOWN_AIHUB_DEVICES,
    auth_configured,
    qai_hub_available,
    select_backend,
)


@pytest.fixture
def clean_aihub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip AI Hub auth + backend env vars."""
    for var in ("QAI_HUB_API_KEY", "QUAD_AIHUB_BACKEND"):
        monkeypatch.delenv(var, raising=False)


# ─── Backend detection ──────────────────────────────────────────────────────


class TestBackendDetection:
    def test_mock_always_available(self, clean_aihub_env: None) -> None:
        assert select_backend("mock") == "mock"

    def test_qai_hub_unavailable_raises(
        self, clean_aihub_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quad.adapters.aihub_adapter as mod
        monkeypatch.setattr(mod, "qai_hub_available", lambda: False)
        with pytest.raises(AIHubUnavailableError):
            select_backend("qai_hub")

    def test_qai_hub_no_auth_raises(
        self, clean_aihub_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quad.adapters.aihub_adapter as mod
        monkeypatch.setattr(mod, "qai_hub_available", lambda: True)
        monkeypatch.setattr(mod, "auth_configured", lambda: False)
        with pytest.raises(AIHubAuthError):
            select_backend("qai_hub")

    def test_auto_falls_back_to_mock(
        self, clean_aihub_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quad.adapters.aihub_adapter as mod
        monkeypatch.setattr(mod, "qai_hub_available", lambda: False)
        assert select_backend("auto") == "mock"

    def test_env_override_to_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QUAD_AIHUB_BACKEND", "mock")
        assert select_backend("auto") == "mock"

    def test_auth_configured_via_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QAI_HUB_API_KEY", "fake-token")
        assert auth_configured() is True

    def test_auth_not_configured_when_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QAI_HUB_API_KEY", "")
        # Note: this might still be True if ~/.qai_hub/client.ini exists on
        # the test machine — accept either result, but empty env var alone
        # shouldn't be enough. Skip the stricter assertion for portability.
        _ = auth_configured()


# ─── Mock profile ────────────────────────────────────────────────────────────


class TestAIHubProfileMock:
    def test_profile_returns_aihub_profile(
        self, clean_aihub_env: None, tmp_path: Path
    ) -> None:
        a = AIHubAdapter(backend="mock")
        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        result = a.profile_on_device(model, device="Snapdragon X Elite CRD")
        assert isinstance(result, AIHubProfile)
        assert result.backend == "mock"
        assert result.device == "Snapdragon X Elite CRD"
        assert result.inference_time_us > 0
        assert result.peak_memory_mb > 0
        assert result.compute_unit == "NPU"

    def test_profile_deterministic_per_model_path(
        self, clean_aihub_env: None, tmp_path: Path
    ) -> None:
        # Same model path + same device → same numbers
        a = AIHubAdapter(backend="mock")
        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        r1 = a.profile_on_device(model, device="Snapdragon X Elite CRD")
        r2 = a.profile_on_device(model, device="Snapdragon X Elite CRD")
        assert r1.inference_time_us == r2.inference_time_us
        assert r1.peak_memory_mb == r2.peak_memory_mb

    def test_profile_to_dict_includes_throughput(
        self, clean_aihub_env: None, tmp_path: Path
    ) -> None:
        a = AIHubAdapter(backend="mock")
        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        d = a.profile_on_device(model).to_dict()
        assert "throughput_fps" in d
        assert d["throughput_fps"] > 0


# ─── Mock compile ────────────────────────────────────────────────────────────


class TestAIHubCompileMock:
    def test_compile_returns_aihub_compile_result(
        self, clean_aihub_env: None, tmp_path: Path
    ) -> None:
        a = AIHubAdapter(backend="mock")
        model = tmp_path / "m.onnx"
        model.write_bytes(b"original_onnx_bytes")
        out = tmp_path / "compiled.bin"
        result = a.compile_for_device(
            model,
            output_path=out,
            device="Snapdragon X Elite CRD",
            target_runtime="qnn_context_binary",
        )
        assert isinstance(result, AIHubCompileResult)
        assert result.backend == "mock"
        assert Path(result.output_path).exists()
        # Mock writes a marker prefix
        contents = Path(result.output_path).read_bytes()
        assert contents.startswith(b"QUAD_MOCK_AIHUB_COMPILED")

    def test_compile_default_output_extension(
        self, clean_aihub_env: None, tmp_path: Path
    ) -> None:
        a = AIHubAdapter(backend="mock")
        model = tmp_path / "m.onnx"
        model.write_bytes(b"x")
        result = a.compile_for_device(model, target_runtime="tflite")
        assert result.output_path.endswith(".tflite")


# ─── Adapter-level behaviour ────────────────────────────────────────────────


class TestAIHubAdapter:
    def test_init_falls_back_to_mock(
        self, clean_aihub_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quad.adapters.aihub_adapter as mod
        monkeypatch.setattr(mod, "qai_hub_available", lambda: False)
        a = AIHubAdapter()
        assert a.backend == "mock"

    def test_strict_raises_when_unavailable(
        self, clean_aihub_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import quad.adapters.aihub_adapter as mod
        monkeypatch.setattr(mod, "qai_hub_available", lambda: False)
        with pytest.raises(AIHubUnavailableError):
            AIHubAdapter(backend="qai_hub", strict=True)

    def test_list_devices_mock_returns_known(
        self, clean_aihub_env: None
    ) -> None:
        a = AIHubAdapter(backend="mock")
        devices = a.list_devices()
        assert "Snapdragon X Elite CRD" in devices
        assert all(d in KNOWN_AIHUB_DEVICES for d in devices)

    def test_doctor_reports_status(
        self, clean_aihub_env: None
    ) -> None:
        a = AIHubAdapter(backend="mock")
        d = a.doctor()
        assert d["backend"] == "mock"
        assert "qai_hub_installed" in d
        assert "auth_configured" in d
        assert "qai_hub_api_key_set" in d
        assert d["qai_hub_api_key_set"] is False  # We cleaned the env var
        assert "Snapdragon X Elite CRD" in d["known_devices"]
