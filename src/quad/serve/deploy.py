"""QUAD Serve — Deployment automation for on-device models.

Handles deployment to local devices, remote targets via SSH,
and Android devices via ADB.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DeployResult:
    """Result of a model deployment operation.

    Attributes:
        success: Whether deployment succeeded.
        deployed_path: Path where model was deployed on target.
        deployment_time_s: Time taken for deployment in seconds.
        device_info: Information about the target device.
    """

    success: bool
    deployed_path: str
    deployment_time_s: float
    device_info: str


def deploy_model(
    model_path: str,
    target_device: str = "local",
    ssh_host: str | None = None,
) -> DeployResult:
    """Deploy a compiled model to a target device.

    Supports three deployment modes:
    - local: Copies model to local serving directory
    - remote (SSH): Transfers model to remote device via SCP
    - android (ADB): Pushes model to Android device

    Args:
        model_path: Path to the compiled model binary (.qbin).
        target_device: Target type — "local", "remote", or "android".
        ssh_host: SSH host for remote deployment (e.g., "user@device.local").

    Returns:
        DeployResult with deployment status and details.

    Raises:
        ValueError: If target_device is invalid or ssh_host missing for remote.
        FileNotFoundError: If model_path does not exist (in real mode).
    """
    start = time.perf_counter()
    model_name = Path(model_path).name

    if target_device == "local":
        result = _deploy_local(model_path, model_name)
    elif target_device == "remote":
        if not ssh_host:
            raise ValueError("ssh_host is required for remote deployment")
        result = _deploy_remote(model_path, model_name, ssh_host)
    elif target_device == "android":
        result = _deploy_android(model_path, model_name)
    else:
        raise ValueError(
            f"Invalid target_device: '{target_device}'. "
            f"Must be 'local', 'remote', or 'android'."
        )

    elapsed = time.perf_counter() - start
    result.deployment_time_s = elapsed
    return result


def _deploy_local(model_path: str, model_name: str) -> DeployResult:
    """Deploy model to local serving directory (mock)."""
    serving_dir = "/opt/quad/models"
    deployed_path = f"{serving_dir}/{model_name}"

    # In real mode: shutil.copy2(model_path, deployed_path)
    # Mock mode: simulate success
    return DeployResult(
        success=True,
        deployed_path=deployed_path,
        deployment_time_s=0.0,
        device_info="local: Snapdragon X Elite (NPU + GPU + CPU)",
    )


def _deploy_remote(model_path: str, model_name: str, ssh_host: str) -> DeployResult:
    """Deploy model to remote device via SCP (mock)."""
    remote_path = f"/opt/quad/models/{model_name}"

    # In real mode: subprocess.run(["scp", model_path, f"{ssh_host}:{remote_path}"])
    # Mock mode: simulate successful transfer
    return DeployResult(
        success=True,
        deployed_path=f"{ssh_host}:{remote_path}",
        deployment_time_s=0.0,
        device_info=f"remote: {ssh_host} (Snapdragon 8 Gen 3)",
    )


def _deploy_android(model_path: str, model_name: str) -> DeployResult:
    """Deploy model to Android device via ADB (mock)."""
    android_path = f"/data/local/tmp/quad/{model_name}"

    # In real mode: subprocess.run(["adb", "push", model_path, android_path])
    # Mock mode: simulate successful push
    return DeployResult(
        success=True,
        deployed_path=android_path,
        deployment_time_s=0.0,
        device_info="android: Snapdragon 8 Gen 3 (SM8650) via ADB",
    )
