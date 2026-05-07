"""Tests for AdapterFactory."""

from __future__ import annotations

import pytest

from quad.adapters.factory import AdapterFactory
from quad.adapters.mock_adapter import MockAdapter
from quad.models.config import ServerConfig


class TestAdapterFactory:
    def test_mock_mode_returns_mock_adapter(self) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        adapter = factory.get_adapter("qnn")
        assert isinstance(adapter, MockAdapter)

    def test_caches_adapter_instances(self) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        a1 = factory.get_adapter("qnn")
        a2 = factory.get_adapter("qnn")
        assert a1 is a2

    def test_different_sdks_get_different_adapters(self) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        a1 = factory.get_adapter("qnn")
        a2 = factory.get_adapter("snpe")
        assert a1 is not a2

    def test_mode_property(self) -> None:
        config = ServerConfig(adapter_mode="mock")
        factory = AdapterFactory(config)
        assert factory.mode == "mock"

    def test_real_mode_falls_back_to_mock(self) -> None:
        config = ServerConfig(adapter_mode="real")
        factory = AdapterFactory(config)
        adapter = factory.get_adapter("qnn")
        # Until real adapters are implemented, falls back to mock
        assert isinstance(adapter, MockAdapter)
