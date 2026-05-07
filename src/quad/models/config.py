"""Configuration model for QUAD server."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AdapterConfig(BaseModel):
    """SDK adapter configuration."""

    sdk_path: str = ""
    version: str = ""
    target_arch: str = ""
    tools_path: str = ""
    api_key_env: str = ""
    base_url: str = ""


class PlatformConfig(BaseModel):
    """Platform-specific configuration."""

    enabled: bool = True
    device_type: str = "local"
    ssh_host: str = ""
    ssh_user: str = "root"
    ssh_key: str = ""
    device_serial: str = ""
    adb_path: str = "adb"


class ServerConfig(BaseSettings):
    """QUAD server configuration — loaded from quad.toml + QUAD_* env vars."""

    adapter_mode: Literal["mock", "real"] = "mock"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_format: Literal["json", "console"] = "console"
    model_output_dir: str = "./output"
    template_dir: str = "./templates"
    qnn_sdk_path: str | None = Field(default=None, alias="QUAD_QNN_SDK_PATH")
    snpe_sdk_path: str | None = Field(default=None, alias="QUAD_SNPE_SDK_PATH")
    hexagon_sdk_path: str | None = Field(default=None, alias="QUAD_HEXAGON_SDK_PATH")
    ai_hub_api_key: str | None = Field(default=None, alias="QAI_HUB_API_KEY")
    cache_hardware_detection: bool = True
    default_platform: str | None = None

    # CPU Fixed Point Mode — execute quantized DLC on CPU without dequantization
    # Set in quad.toml [server] enable_cpu_fxp = true
    # Or env var: QUAD_ENABLE_CPU_FXP=true
    enable_cpu_fxp: bool = False

    # Default PD type (unsigned = default SNPE2, signed = requires signed skels)
    pd_type: Literal["unsigned", "signed"] = "unsigned"

    # Default performance profile for inference
    performance_profile: str = "high_performance"

    model_config = {"env_prefix": "QUAD_", "extra": "ignore", "env_ignore_empty": True}
