"""Configuration loader for QUAD server."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from quad.models.config import ServerConfig

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def load_dotenv(env_path: str | Path | None = None) -> None:
    """Load .env file into os.environ.

    Reads KEY=VALUE pairs from .env and sets them in the environment.
    Does NOT override existing environment variables (existing wins).

    Resolution order:
      1. Specified env_path
      2. .env in current directory
      3. Skip silently if no file found
    """
    if env_path is None:
        env_path = Path(".env")

    env_path = Path(env_path)
    if not env_path.exists():
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY=VALUE
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")  # Strip surrounding quotes
            # Only set if not already in environment (env vars win over .env)
            if key and key not in os.environ:
                os.environ[key] = value


def load_config(
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
) -> ServerConfig:
    """Load QUAD configuration from TOML file + .env + environment variables.

    Resolution order (highest priority first):
    1. QUAD_* environment variables (already set in shell)
    2. .env file values (never override existing env vars)
    3. quad.toml [server] section values
    4. ServerConfig defaults

    Args:
        config_path: Path to quad.toml (default: ./quad.toml)
        env_path: Path to .env file (default: ./.env)
    """
    # Step 1: Load .env first (sets os.environ entries if not already set)
    load_dotenv(env_path)

    # Step 2: Load TOML config
    file_config: dict[str, Any] = {}

    if config_path is None:
        config_path = Path("quad.toml")

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
        file_config = raw.get("server", {})

    # Step 3: Merge — env vars (via Pydantic Settings) override file config
    return ServerConfig(**file_config)
