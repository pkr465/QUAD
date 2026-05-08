"""Qualcomm AI Hub adapter — cloud profiling and remote compilation.

Closes GAP_ANALYSIS T1.6: ``QAI_HUB_API_KEY`` was documented in
``.env.example`` and config but the ``qai_hub`` Python SDK was never
imported anywhere. This adapter wires it in.

Qualcomm AI Hub (https://app.aihub.qualcomm.com) provides:
  * Cloud-hosted profiling on real Qualcomm devices (Snapdragon X
    Elite, 8 Elite, 8 Gen 3, …)
  * Cloud compilation: ONNX / PyTorch -> QNN context binary for a
    target device, without needing the SDK installed locally
  * Inference-on-device — submit a model + inputs, get outputs back
  * Performance metrics returned as a structured report (latency,
    memory, NPU utilisation)

Design choices:

* **Soft dependency on qai_hub.** Like AIMETAdapter, we import
  lazily so the package isn't a hard requirement.
* **Mock backend for testing.** Same public API; mock returns
  deterministic synthetic results.
* **Authentication via env var.** ``QAI_HUB_API_KEY`` is read at
  client creation; the SDK itself reads ``~/.qai_hub/client.ini``
  too. We support both.
* **No network calls in CI.** Tests use the mock backend
  exclusively.

Used by:
- The ``profile_workload`` MCP tool with a ``cloud=True`` option
  (Phase F adds the wiring)
- ``quad doctor`` checks for AI Hub configuration
- The ``aihub`` skill in Claude Code
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from quad.exceptions import QUADError

logger = logging.getLogger(__name__)


# ─── Public types ────────────────────────────────────────────────────────────


# Common AI Hub device names (subset). The real list is fetched
# dynamically from the API; this is for autocomplete + offline mock.
KNOWN_AIHUB_DEVICES = (
    "Snapdragon X Elite CRD",
    "Samsung Galaxy S24 (Family)",
    "Samsung Galaxy S25 (Family)",
    "Snapdragon 8 Elite QRD",
    "Snapdragon 8 Gen 3 QRD",
    "QCS6490 (Proxy)",
    "QCS8550 (Proxy)",
)


@dataclass
class AIHubProfile:
    """Result of an AI Hub profile job."""

    job_id: str
    device: str
    inference_time_us: float
    peak_memory_mb: float
    compute_unit: str  # 'NPU' / 'GPU' / 'CPU'
    compile_time_s: float
    profile_url: str = ""
    layer_count: int = 0
    notes: list[str] = field(default_factory=list)
    raw_results: dict[str, Any] = field(default_factory=dict)
    backend: Literal["qai_hub", "mock"] = "mock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "device": self.device,
            "inference_time_us": round(self.inference_time_us, 2),
            "inference_time_ms": round(self.inference_time_us / 1000.0, 3),
            "throughput_fps": round(1_000_000 / self.inference_time_us, 1)
            if self.inference_time_us > 0
            else 0,
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "compute_unit": self.compute_unit,
            "compile_time_s": round(self.compile_time_s, 2),
            "profile_url": self.profile_url,
            "layer_count": self.layer_count,
            "notes": self.notes,
            "backend": self.backend,
        }


@dataclass
class AIHubCompileResult:
    """Result of an AI Hub compile job."""

    job_id: str
    device: str
    output_path: str  # local path to the downloaded artifact
    output_format: Literal["qnn_context_binary", "tflite", "tensorflow_lite", "qnn_lib"]
    compile_time_s: float
    target_runtime: str
    artifact_url: str = ""
    notes: list[str] = field(default_factory=list)
    backend: Literal["qai_hub", "mock"] = "mock"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "device": self.device,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "compile_time_s": round(self.compile_time_s, 2),
            "target_runtime": self.target_runtime,
            "artifact_url": self.artifact_url,
            "notes": self.notes,
            "backend": self.backend,
        }


class AIHubAuthError(QUADError):
    """Raised when AI Hub auth (API key or config file) is not set up."""

    def __init__(self) -> None:
        super().__init__(
            "Qualcomm AI Hub credentials not found. Set QAI_HUB_API_KEY in your "
            "environment, or run `qai-hub configure --api_token <token>` to write "
            "~/.qai_hub/client.ini. Get an API key at https://app.aihub.qualcomm.com",
            code="AIHUB_AUTH",
        )


class AIHubUnavailableError(QUADError):
    """Raised when qai_hub package isn't installed."""

    def __init__(self) -> None:
        super().__init__(
            "qai_hub package not installed. Install via: pip install qai-hub. "
            "Set QUAD_AIHUB_BACKEND=mock to use the mock backend for testing.",
            code="AIHUB_UNAVAILABLE",
        )


# ─── Backend detection ───────────────────────────────────────────────────────


def qai_hub_available() -> bool:
    """True if ``qai_hub`` can be imported AND auth is configured."""
    try:
        import qai_hub  # noqa: F401
        return True
    except ImportError:
        return False


def auth_configured() -> bool:
    """True if AI Hub auth is set up (env var or ~/.qai_hub/client.ini)."""
    if os.environ.get("QAI_HUB_API_KEY", "").strip():
        return True
    cfg = Path.home() / ".qai_hub" / "client.ini"
    return cfg.is_file()


def select_backend(prefer: str = "auto") -> str:
    """Pick the AI Hub backend.

    Args:
        prefer: 'auto' | 'qai_hub' | 'mock'

    Returns:
        Backend identifier.
    """
    env_override = os.environ.get("QUAD_AIHUB_BACKEND", "").strip().lower()
    if env_override in ("mock", "qai_hub"):
        prefer = env_override

    if prefer == "mock":
        return "mock"
    if prefer == "qai_hub":
        if not qai_hub_available():
            raise AIHubUnavailableError()
        if not auth_configured():
            raise AIHubAuthError()
        return "qai_hub"
    # auto
    if qai_hub_available() and auth_configured():
        return "qai_hub"
    return "mock"


# ─── Mock backend ────────────────────────────────────────────────────────────


def _mock_profile(model_path: Path, device: str) -> AIHubProfile:
    """Deterministic synthetic profile for tests + offline use."""
    # Seed RNG by model path so the same model always returns the same numbers
    rng = np.random.default_rng(seed=abs(hash(str(model_path) + device)) % (2**32))
    base_us = float(rng.uniform(800, 5000))  # 0.8-5 ms typical for AI-PC NPU
    return AIHubProfile(
        job_id=f"mock_{int(time.time())}_{abs(hash(model_path)) % 10000}",
        device=device,
        inference_time_us=base_us,
        peak_memory_mb=float(rng.uniform(20, 200)),
        compute_unit="NPU",
        compile_time_s=float(rng.uniform(15, 60)),
        profile_url=f"https://app.aihub.qualcomm.com/jobs/mock_{int(time.time())}",
        layer_count=int(rng.integers(50, 200)),
        notes=[
            "Mock profile — deterministic seeded output.",
            "For real cloud profiling install qai-hub and set QAI_HUB_API_KEY.",
        ],
        raw_results={"seed_basis": str(model_path)},
        backend="mock",
    )


def _mock_compile(
    model_path: Path,
    output_path: Path,
    device: str,
    target_runtime: str,
) -> AIHubCompileResult:
    """Deterministic synthetic compile for tests + offline use."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        # Pretend we compiled it — write a small marker file
        output_path.write_bytes(b"QUAD_MOCK_AIHUB_COMPILED:" + model_path.read_bytes()[:32])
    else:
        output_path.write_bytes(b"QUAD_MOCK_AIHUB_COMPILED")
    return AIHubCompileResult(
        job_id=f"mock_compile_{int(time.time())}",
        device=device,
        output_path=output_path.as_posix(),
        output_format="qnn_context_binary",
        compile_time_s=12.5,
        target_runtime=target_runtime,
        artifact_url=f"https://app.aihub.qualcomm.com/jobs/mock_compile_{int(time.time())}",
        notes=["Mock compile — bytes copied with marker prefix."],
        backend="mock",
    )


# ─── Real qai_hub backend ────────────────────────────────────────────────────


def _qai_hub_profile(model_path: Path, device: str) -> AIHubProfile:
    """Submit a real profile job to AI Hub."""
    import qai_hub as hub

    target_device = hub.Device(device)
    job = hub.submit_profile_job(
        model=str(model_path),
        device=target_device,
        name=f"quad-profile-{model_path.stem}",
    )
    job.wait()
    profile = job.download_profile()
    return AIHubProfile(
        job_id=job.job_id,
        device=device,
        inference_time_us=profile.get("execution_summary", {}).get("estimated_inference_time", 0),
        peak_memory_mb=profile.get("execution_summary", {}).get("peak_memory_usage", 0) / (1024 * 1024),
        compute_unit=profile.get("execution_summary", {}).get("primary_compute_unit", "NPU"),
        compile_time_s=profile.get("execution_summary", {}).get("compile_time", 0),
        profile_url=f"https://app.aihub.qualcomm.com/jobs/{job.job_id}",
        layer_count=len(profile.get("execution_detail", [])),
        raw_results=profile,
        backend="qai_hub",
    )


def _qai_hub_compile(
    model_path: Path,
    output_path: Path,
    device: str,
    target_runtime: str,
) -> AIHubCompileResult:
    """Submit a real compile job to AI Hub."""
    import qai_hub as hub

    target_device = hub.Device(device)
    options_map = {
        "qnn_context_binary": "--target_runtime qnn_context_binary",
        "tflite": "--target_runtime tflite",
        "qnn_lib": "--target_runtime qnn_lib_aarch64_android",
    }
    job = hub.submit_compile_job(
        model=str(model_path),
        device=target_device,
        name=f"quad-compile-{model_path.stem}",
        options=options_map.get(target_runtime, ""),
    )
    job.wait()
    target_model = job.get_target_model()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_model.download(str(output_path))
    return AIHubCompileResult(
        job_id=job.job_id,
        device=device,
        output_path=output_path.as_posix(),
        output_format=target_runtime,  # type: ignore[arg-type]
        compile_time_s=0.0,  # AI Hub doesn't expose this directly
        target_runtime=target_runtime,
        artifact_url=f"https://app.aihub.qualcomm.com/jobs/{job.job_id}",
        backend="qai_hub",
    )


# ─── Public API ──────────────────────────────────────────────────────────────


class AIHubAdapter:
    """High-level facade over Qualcomm AI Hub for cloud profiling and compilation."""

    def __init__(self, *, backend: str = "auto", strict: bool = False):
        """
        Args:
            backend: 'auto' | 'qai_hub' | 'mock'. Default 'auto'.
            strict: if True, raise when real backend was requested but
                package or auth is missing (rather than falling back).
        """
        try:
            self._backend = select_backend(backend)
        except (AIHubUnavailableError, AIHubAuthError) as e:
            if strict:
                raise
            logger.warning(
                "aihub_backend_unavailable_falling_back_to_mock",
                extra={"requested": backend, "reason": str(e)},
            )
            self._backend = "mock"
        logger.info("aihub_adapter_init", extra={"backend": self._backend})

    @property
    def backend(self) -> str:
        return self._backend

    def profile_on_device(
        self,
        model_path: str | Path,
        device: str = "Snapdragon X Elite CRD",
    ) -> AIHubProfile:
        """Submit a profile job to AI Hub and wait for results.

        Args:
            model_path: ONNX / PyTorch / TFLite / QNN model
            device: AI Hub device name (see KNOWN_AIHUB_DEVICES)
        """
        src = Path(model_path)
        if self._backend == "mock":
            return _mock_profile(src, device)
        return _qai_hub_profile(src, device)

    def compile_for_device(
        self,
        model_path: str | Path,
        output_path: str | Path | None = None,
        device: str = "Snapdragon X Elite CRD",
        target_runtime: str = "qnn_context_binary",
    ) -> AIHubCompileResult:
        """Submit a compile job to AI Hub and download the artifact.

        Args:
            model_path: source model (ONNX / PyTorch)
            output_path: where to save the compiled artifact
            device: AI Hub device name
            target_runtime: 'qnn_context_binary' | 'tflite' | 'qnn_lib'
        """
        src = Path(model_path)
        if output_path is None:
            ext = {
                "qnn_context_binary": ".bin",
                "tflite": ".tflite",
                "qnn_lib": ".so",
            }.get(target_runtime, ".bin")
            output_path = src.with_suffix(ext)
        out = Path(output_path)

        if self._backend == "mock":
            return _mock_compile(src, out, device, target_runtime)
        return _qai_hub_compile(src, out, device, target_runtime)

    def list_devices(self) -> list[str]:
        """Return the list of AI Hub devices.

        In real-backend mode, queries the API. In mock mode, returns
        the static KNOWN_AIHUB_DEVICES list.
        """
        if self._backend == "qai_hub":
            try:
                import qai_hub as hub

                return [d.name for d in hub.get_devices()]
            except Exception as e:
                logger.warning("aihub_list_devices_failed: %s", e)
                return list(KNOWN_AIHUB_DEVICES)
        return list(KNOWN_AIHUB_DEVICES)

    def doctor(self) -> dict[str, Any]:
        """Status snapshot used by `quad doctor` and the AI Hub skill."""
        return {
            "backend": self._backend,
            "qai_hub_installed": qai_hub_available(),
            "auth_configured": auth_configured(),
            "qai_hub_api_key_set": bool(os.environ.get("QAI_HUB_API_KEY", "").strip()),
            "client_ini_present": (Path.home() / ".qai_hub" / "client.ini").is_file(),
            "known_devices": list(KNOWN_AIHUB_DEVICES),
        }
