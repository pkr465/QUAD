"""Shared pytest fixtures for QUAD test suite."""

from __future__ import annotations

import os
import pytest

# Ensure tests always use mock mode regardless of .env contents.
os.environ.setdefault("QUAD_ADAPTER_MODE", "mock")


@pytest.fixture
def mock_config() -> dict:
    """Minimal config for testing."""
    return {
        "server": {
            "adapter_mode": "mock",
            "log_level": "debug",
            "log_format": "console",
            "model_output_dir": "./output",
            "template_dir": "./templates",
        },
    }
