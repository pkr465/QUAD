"""QUAD Model — unified model loading and inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from quad.runtime.device import Device
from quad.runtime.tensor import Tensor

# Type alias: input tensor name → new shape
InputShapeMap = dict[str, tuple[int, ...]]


class Model:
    """Loaded model ready for inference.

    Wraps compiled model binary and provides a callable interface.
    Supports power budget constraints, device fallback, and network resizing.

    Usage:
        model = quad.load("model.onnx", device=Device("npu"))
        output = model(input_tensor)

        # Network resizing — change input dimensions at load time
        model = quad.load(
            "model.dlc",
            device=Device("npu"),
            input_dimensions={"data": (3, 1080, 1440, 3)}
        )
    """

    def __init__(
        self,
        path: str,
        device: Device,
        power_budget_mw: float | None = None,
        compiled: bool = False,
        input_dimensions: InputShapeMap | None = None,
        enable_init_cache: bool = False,
        pd_type: str = "unsigned",
        enable_cpu_fxp: bool = False,
    ):
        self._path = Path(path)
        self._device = device
        self._power_budget_mw = power_budget_mw
        self._compiled = compiled
        self._loaded = True
        self._load_time_ms: float = 0.0
        self._enable_init_cache = enable_init_cache
        self._pd_type = pd_type
        self._enable_cpu_fxp = enable_cpu_fxp

        # Network resizing — applied at build time via SNPEBuilder.setInputDimensions()
        self._input_dimensions: InputShapeMap = input_dimensions or {}

        # Model metadata (populated during load)
        self._input_shapes: list[tuple[int, ...]] = [(1, 3, 224, 224)]
        self._output_shapes: list[tuple[int, ...]] = [(1, 1000)]
        self._num_params: int = 25_000_000  # Default estimate
        self._format = self._path.suffix.lstrip(".")

        # Apply input dimension overrides to input shapes
        if self._input_dimensions:
            self._input_shapes = list(self._input_dimensions.values())

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def device(self) -> Device:
        return self._device

    @property
    def power_budget_mw(self) -> float | None:
        return self._power_budget_mw

    @property
    def format(self) -> str:
        return self._format

    @property
    def input_shapes(self) -> list[tuple[int, ...]]:
        return self._input_shapes

    @property
    def output_shapes(self) -> list[tuple[int, ...]]:
        return self._output_shapes

    @property
    def num_params(self) -> int:
        return self._num_params

    @property
    def enable_init_cache(self) -> bool:
        """Whether init caching is enabled for this model (DSP/AIP only)."""
        return self._enable_init_cache

    @property
    def pd_type(self) -> str:
        """Protection Domain type: 'unsigned' (default) or 'signed'.

        Controls snpe-net-run --platform_options:
          unsigned → unsignedPD:ON  (default, no signing required)
          signed   → unsignedPD:OFF (requires customer-signed skel libs)
        """
        return self._pd_type

    @property
    def enable_cpu_fxp(self) -> bool:
        """Whether CPU Fixed Point mode is enabled.

        CPU FXP executes quantized models directly on CPU without dequantization.
        Requirements:
          - Must use a quantized DLC
          - Must select CPU runtime
          - Not all ops support FXP; unsupported ops fall back to CPU float
          - DSP→CPU fallback also uses FXP (no dequant step = better perf)

        Default: False (CPU dequantizes to float32 for backward compat).
        Enable with: load("model.dlc", enable_cpu_fxp=True)
        CLI: snpe-net-run --enable_cpu_fxp
        """
        return self._enable_cpu_fxp

    @property
    def platform_options(self) -> str:
        """Return the platform options string for this model's PD type."""
        from quad.adapters.dsp_env import get_platform_option
        return get_platform_option(self._pd_type)

    @property
    def input_dimensions(self) -> InputShapeMap:
        """Dynamic input dimension overrides (empty = use model defaults)."""
        return self._input_dimensions

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def resize_input(self, input_dimensions: InputShapeMap) -> None:
        """Resize the network inputs at runtime.

        Equivalent to SNPEBuilder.setInputDimensions() — allows changing
        input shapes without re-converting the model. Useful for:
        - Processing different image resolutions (e.g. 720p vs 1080p)
        - Changing batch size
        - Handling variable-length sequences

        Args:
            input_dimensions: Map of input_name → new shape tuple
                e.g. {"data": (3, 1080, 1440, 3)} for 1080p batch of 3

        Example:
            # Original model: data = (1, 3, 224, 224)
            model.resize_input({"data": (1, 3, 1080, 1920)})  # 1080p
        """
        self._input_dimensions = input_dimensions
        self._input_shapes = list(input_dimensions.values())

    def set_power_mode(self, mode: str) -> None:
        """Set power mode: 'performance', 'balanced', or 'efficiency'."""
        power_map = {
            "performance": None,  # No constraint
            "balanced": 5000.0,
            "efficiency": 3000.0,
        }
        self._power_budget_mw = power_map.get(mode)

    def __call__(self, *inputs: Tensor | np.ndarray, **kwargs) -> Tensor:
        """Run inference.

        Args:
            *inputs: Input tensors or numpy arrays
            **kwargs: Optional power_budget_mw override

        Returns:
            Output tensor on the model's device
        """
        power_budget = kwargs.get("power_budget_mw", self._power_budget_mw)

        # Convert numpy inputs to tensors
        tensor_inputs = []
        for inp in inputs:
            if isinstance(inp, np.ndarray):
                tensor_inputs.append(Tensor.from_numpy(inp, device=self._device))
            else:
                tensor_inputs.append(inp)

        # Mock inference: generate output based on model metadata
        output_shape = self._output_shapes[0] if self._output_shapes else (1, 1000)
        output_data = np.random.randn(*output_shape).astype("float32")

        # Simulate power-aware execution
        # (In real mode, this would select runtime based on power budget)

        return Tensor.from_numpy(output_data, device=self._device)

    def infer_async(self, *inputs: Tensor | np.ndarray, stream=None) -> "InferenceFuture":
        """Run inference asynchronously.

        Returns a future that can be awaited for the result.
        """
        # In mock mode, compute immediately but wrap in future
        result = self(*inputs)
        return InferenceFuture(result)

    def unload(self) -> None:
        """Unload model and free device memory."""
        self._loaded = False

    def __repr__(self) -> str:
        return (
            f"Model(path='{self._path.name}', device='{self._device.type}', "
            f"format='{self._format}', params={self._num_params:,})"
        )


class InferenceFuture:
    """Future representing an async inference result."""

    def __init__(self, result: Tensor):
        self._result = result
        self._done = True

    def result(self, timeout_ms: float = 30000) -> Tensor:
        """Get the inference result (blocks until complete)."""
        return self._result

    @property
    def done(self) -> bool:
        return self._done


def load(
    path: str,
    device: Device | str = "auto",
    power_budget_mw: float | None = None,
    input_dimensions: InputShapeMap | None = None,
    enable_init_cache: bool = False,
    pd_type: str = "unsigned",
    enable_cpu_fxp: bool = False,
) -> Model:
    """Load a model for inference.

    Supports ONNX, PyTorch, QNN (.qbin), and SNPE (.dlc) formats.

    Args:
        path: Path to model file (.onnx, .pt, .qbin, .dlc)
        device: Target device (Device object or string like "npu", "auto")
        power_budget_mw: Optional power constraint in milliwatts
        input_dimensions: Optional dynamic input resizing.
            Map of input_name → shape tuple applied at build time via
            SNPEBuilder.setInputDimensions() / TensorShapeMap.
        enable_init_cache: If True, use init caching to speed up subsequent
            loads (DSP and AIP runtimes only). On first run, initialization
            structures are stored in the DLC file. On subsequent loads they
            are read from the file, skipping re-computation.
            NOTE: In real mode the DLC is saved after build. In mock mode
            this flag is noted but has no effect on performance.

    Returns:
        Loaded Model ready for inference.

    Examples:
        # Standard load
        model = quad.load("model.dlc", device="npu")

        # With network resizing — load for 1080p instead of original 224x224
        model = quad.load(
            "model.dlc",
            device="npu",
            input_dimensions={"data": (3, 1080, 1440, 3)},
        )

        # Batch of 4 inputs
        model = quad.load(
            "model.dlc",
            device="npu",
            input_dimensions={"input": (4, 3, 224, 224)},
        )
    """
    start = time.perf_counter()

    if isinstance(device, str):
        device = Device(device)

    model = Model(
        path=path,
        device=device,
        power_budget_mw=power_budget_mw,
        compiled=path.endswith((".qbin", ".dlc", ".bin")),
        input_dimensions=input_dimensions,
        enable_init_cache=enable_init_cache,
        pd_type=pd_type,
        enable_cpu_fxp=enable_cpu_fxp,
    )
    model._load_time_ms = (time.perf_counter() - start) * 1000

    return model
