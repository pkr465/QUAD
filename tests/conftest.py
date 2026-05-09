"""Shared pytest fixtures for QUAD test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure tests always use mock mode regardless of .env contents.
os.environ.setdefault("QUAD_ADAPTER_MODE", "mock")

# Several tests resolve template paths relative to the current working
# directory (e.g. ``FileSystemLoader("templates/snpe/cpp")``). Pin CWD
# to the repo root so the suite runs from anywhere — parent dir, a
# subdirectory of tests/, or wherever a developer happens to be.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRIOR_CWD = os.getcwd()
os.chdir(_REPO_ROOT)


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
