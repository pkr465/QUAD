"""Tests for AdapterFactory."""

from __future__ import annotations

import pytest

from quad.adapters.factory import (
    AdapterFactory,
    RealAdapterUnavailableError,
)
from quad.adapters.mock_adapter import MockAdapter
from quad.models.config import ServerConfig


@pytest.fixture
def clean_sdk_env(monkeypatch):
    """Strip any SDK env vars + strict-mode flag for hermetic tests."""
    for var in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT", "QUAD_STRICT_REAL"):
        monkeypatch.delenv(var, raising=False)


class TestAdapterFactory:
    def test_mock_mode_returns_mock_adapter(self, clean_sdk_env) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        adapter = factory.get_adapter("qnn")
        assert isinstance(adapter, MockAdapter)

    def test_caches_adapter_instances(self, clean_sdk_env) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        a1 = factory.get_adapter("qnn")
        a2 = factory.get_adapter("qnn")
        assert a1 is a2

    def test_different_sdks_get_different_adapters(self, clean_sdk_env) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        a1 = factory.get_adapter("qnn")
        a2 = factory.get_adapter("snpe")
        assert a1 is not a2

    def test_mode_property(self, clean_sdk_env) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        assert factory.mode == "mock"

    def test_real_mode_falls_back_to_mock(self, clean_sdk_env) -> None:
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config)
        adapter = factory.get_adapter("qnn")
        # When SDK is missing, falls back to mock so dev/CI keeps working
        assert isinstance(adapter, MockAdapter)

    def test_fallback_mock_is_tagged(self, clean_sdk_env) -> None:
        """Fallback mock should be tagged so callers can detect it."""
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config)
        adapter = factory.get_adapter("qnn")
        assert getattr(adapter, "fell_back_from_real", False) is True
        assert "QAIRT_SDK_ROOT" in getattr(adapter, "fallback_reason", "")

    def test_intentional_mock_is_not_tagged(self, clean_sdk_env) -> None:
        """Mock-mode adapter should NOT carry the fallback tag."""
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        adapter = factory.get_adapter("qnn")
        assert getattr(adapter, "fell_back_from_real", False) is False

    def test_strict_real_raises_when_sdk_missing(self, clean_sdk_env) -> None:
        """Strict mode should refuse the silent fallback."""
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config, strict=True)
        with pytest.raises(RealAdapterUnavailableError) as exc:
            factory.get_adapter("qnn")
        assert "QAIRT_SDK_ROOT" in str(exc.value)

    def test_strict_real_via_env_var(self, clean_sdk_env, monkeypatch) -> None:
        """QUAD_STRICT_REAL=1 should enable strict mode without explicit kwarg."""
        monkeypatch.setenv("QUAD_STRICT_REAL", "1")
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config)
        assert factory.strict is True
        with pytest.raises(RealAdapterUnavailableError):
            factory.get_adapter("qnn")

    def test_strict_real_does_not_apply_in_mock_mode(self, clean_sdk_env, monkeypatch) -> None:
        """Strict flag must not break intentional mock mode."""
        monkeypatch.setenv("QUAD_STRICT_REAL", "1")
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        # Mock mode bypasses real-adapter creation entirely
        assert isinstance(factory.get_adapter("qnn"), MockAdapter)

    def test_real_mode_ready_reports_mock_mode(self, clean_sdk_env) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        ready, reason = factory.real_mode_ready()
        assert ready is False
        assert "mock" in reason

    def test_real_mode_ready_reports_missing_sdk(self, clean_sdk_env) -> None:
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config)
        ready, reason = factory.real_mode_ready()
        assert ready is False
        assert "QAIRT_SDK_ROOT" in reason

    def test_real_mode_ready_when_sdk_present(self, clean_sdk_env, tmp_path, monkeypatch) -> None:
        """Real mode reports ready when QAIRT_SDK_ROOT points at a real dir."""
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(tmp_path))
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config)
        ready, reason = factory.real_mode_ready()
        assert ready is True
        assert str(tmp_path) in reason

    def test_real_mode_rejects_missing_sdk_path(self, clean_sdk_env, monkeypatch) -> None:
        """Env var pointing at a non-existent path should NOT count as ready."""
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/definitely/does/not/exist/qairt")
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config)
        ready, _ = factory.real_mode_ready()
        assert ready is False
