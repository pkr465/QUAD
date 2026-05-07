"""Adapter factory — config-driven selection of mock vs real SDK adapters."""

from __future__ import annotations

import os

from quad.adapters.base import SDKAdapter
from quad.adapters.mock_adapter import MockAdapter
from quad.models.config import ServerConfig


class AdapterFactory:
    """Creates the appropriate SDK adapter based on configuration.

    In mock mode (default), all adapters return simulated responses.
    In real mode, uses QAIRTAdapter (requires QAIRT_SDK_ROOT set).
    """

    def __init__(self, config: ServerConfig):
        self._config = config
        self._mode = config.adapter_mode
        self._cache: dict[str, SDKAdapter] = {}

    def get_adapter(self, sdk: str = "auto") -> SDKAdapter:
        """Get an adapter for the specified SDK.

        Args:
            sdk: SDK name ("qnn", "snpe", "qairt", "auto") or others.

        Returns:
            Appropriate SDKAdapter instance (mock or real).
        """
        cache_key = f"{self._mode}:{sdk}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        adapter: SDKAdapter
        if self._mode == "mock":
            adapter = MockAdapter(sdk_name=sdk)
        else:
            adapter = self._create_real_adapter(sdk)

        self._cache[cache_key] = adapter
        return adapter

    def _create_real_adapter(self, sdk: str) -> SDKAdapter:
        """Create a real SDK adapter. Falls back to mock if SDK not available."""
        # Try QAIRT adapter (covers both QNN and SNPE)
        if sdk in ("qnn", "snpe", "qairt", "auto"):
            sdk_root = os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("SNPE_ROOT")
            if sdk_root:
                from quad.adapters.qairt_adapter import QAIRTAdapter
                return QAIRTAdapter(sdk_root=sdk_root)

        # Fallback to mock with warning
        import structlog
        logger = structlog.get_logger()
        logger.warning(
            "real_adapter_unavailable",
            sdk=sdk,
            reason="SDK not found or not configured. Falling back to mock.",
        )
        return MockAdapter(sdk_name=sdk)

    @property
    def mode(self) -> str:
        """Current adapter mode (mock or real)."""
        return self._mode
