"""Adapter factory — config-driven selection of mock vs real SDK adapters.

The factory is the single switch between mock and real SDK execution. Real
adapters require a Qualcomm SDK installation reachable via
``QAIRT_SDK_ROOT`` (or legacy ``SNPE_ROOT``). When real mode is requested
but no SDK is available the factory falls back to the mock adapter so
mock-mode CI keeps working — but it loudly tags the returned adapter
(``adapter.fell_back_from_real`` / ``adapter.fallback_reason``) so callers
that care can detect and react.

Set ``QUAD_STRICT_REAL=1`` (or pass ``strict=True``) to disable the
fallback and raise ``RealAdapterUnavailableError`` instead. Use this in
real-hardware CI to fail fast when the SDK is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

from quad.adapters.base import SDKAdapter
from quad.adapters.mock_adapter import MockAdapter
from quad.models.config import ServerConfig


class RealAdapterUnavailableError(RuntimeError):
    """Raised in strict mode when a real adapter was requested but the SDK is missing."""


def _truthy_env(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy string."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_sdk_root() -> str | None:
    """Return the first SDK root env var that points at a real directory."""
    for var in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT"):
        val = os.environ.get(var, "").strip()
        if val and Path(val).exists():
            return val
    return None


class AdapterFactory:
    """Creates the appropriate SDK adapter based on configuration.

    In mock mode (default), all adapters return simulated responses.
    In real mode, uses QAIRTAdapter (requires QAIRT_SDK_ROOT set).

    Args:
        config: ServerConfig — drives ``adapter_mode``.
        strict: If True, raise ``RealAdapterUnavailableError`` when real mode
            is requested but the SDK is missing instead of silently falling
            back to mock. Defaults to the value of the ``QUAD_STRICT_REAL``
            env var.
    """

    def __init__(self, config: ServerConfig, strict: bool | None = None):
        self._config = config
        self._mode = config.adapter_mode
        self._strict = strict if strict is not None else _truthy_env("QUAD_STRICT_REAL")
        self._cache: dict[str, SDKAdapter] = {}

    def get_adapter(self, sdk: str = "auto") -> SDKAdapter:
        """Get an adapter for the specified SDK.

        Args:
            sdk: SDK name ("qnn", "snpe", "qairt", "auto") or others.

        Returns:
            Appropriate SDKAdapter instance (mock or real).

        Raises:
            RealAdapterUnavailableError: if strict mode is on, real mode is
                requested, and no SDK is found.
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
        """Create a real SDK adapter. Falls back to mock if SDK not available.

        The returned mock adapter (in fallback case) is tagged with
        ``fell_back_from_real=True`` and a human-readable
        ``fallback_reason`` so callers can detect the fallback without
        inspecting log output.
        """
        # Try QAIRT adapter (covers both QNN and SNPE)
        if sdk in ("qnn", "snpe", "qairt", "auto"):
            sdk_root = _resolve_sdk_root()
            if sdk_root:
                from quad.adapters.qairt_adapter import QAIRTAdapter
                return QAIRTAdapter(sdk_root=sdk_root)

            reason = (
                "QAIRT_SDK_ROOT (or QNN_SDK_ROOT / SNPE_ROOT) is not set or "
                "points at a missing directory. Run: source activate_qairt.sh "
                "or set the env var to your SDK install path."
            )
        else:
            reason = f"No real adapter implemented for sdk={sdk!r}"

        if self._strict:
            raise RealAdapterUnavailableError(
                f"Real adapter requested but unavailable: {reason}"
            )

        # Fallback to mock with prominent warning
        import structlog
        logger = structlog.get_logger()
        logger.warning(
            "real_adapter_unavailable_falling_back_to_mock",
            sdk=sdk,
            reason=reason,
            hint="Set QUAD_STRICT_REAL=1 to fail fast instead of falling back.",
        )
        adapter = MockAdapter(sdk_name=sdk)
        # Tag the adapter so callers can tell apart "intentional mock"
        # from "fallback mock". The tags are simple attributes — adding
        # them does not break the SDKAdapter protocol.
        adapter.fell_back_from_real = True  # type: ignore[attr-defined]
        adapter.fallback_reason = reason  # type: ignore[attr-defined]
        return adapter

    @property
    def mode(self) -> str:
        """Current adapter mode (mock or real)."""
        return self._mode

    @property
    def strict(self) -> bool:
        """Whether strict mode is enabled (raises on missing SDK)."""
        return self._strict

    def real_mode_ready(self) -> tuple[bool, str]:
        """Return (ready, reason) for whether real mode would actually work.

        Useful for ``quad doctor`` / ``quad mode`` to report the readiness
        without instantiating the adapter or running an inference.
        """
        if self._mode != "real":
            return False, f"adapter_mode is {self._mode!r}; set to 'real' to enable hardware."
        sdk_root = _resolve_sdk_root()
        if not sdk_root:
            return False, (
                "QAIRT_SDK_ROOT / QNN_SDK_ROOT / SNPE_ROOT not set or path missing."
            )
        return True, f"Real mode active. SDK root: {sdk_root}"
