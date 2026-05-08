"""Model input introspection and dummy-input generation.

Closes GAP_ANALYSIS T2.8: previously every adapter call that needed
inference inputs (profile, execute_inference, calibration) used a
hardcoded ``np.random.randn(1, 3, 224, 224)`` regardless of the
model's actual input shape — so models with different input shapes
silently broke, and quantization calibration was pinned to garbage.

This module:

* introspects a model file (.dlc, .bin, .onnx) to discover its input
  tensors (name, shape, dtype) using the appropriate SDK tool when
  available — ``snpe-dlc-info`` for DLC, ``onnx`` Python module for
  ONNX, ``snpe-network-info`` for QNN context binaries
* falls back to a configurable default shape when no introspection is
  possible (e.g. mock mode without the SDK installed)
* generates raw input files of the right shape and dtype, optionally
  loaded from a calibration dataset directory rather than synthesised
  randomly
* writes the canonical SNPE/QNN ``input_list.txt`` format that
  ``snpe-net-run`` and ``qairt-quantizer`` consume

Used by:
- ``QAIRTAdapter._create_dummy_input_list`` (replacement)
- ``QAIRTAdapter.execute_inference`` (real input marshalling)
- AIMET adapter (calibration data — Phase C)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class TensorSpec:
    """Description of a single input or output tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # numpy dtype string, e.g. 'float32', 'int8'

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= max(d, 1)
        return n

    @property
    def num_bytes(self) -> int:
        return self.numel * self.numpy_dtype.itemsize

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass
class ModelIO:
    """Aggregate of a model's inputs and outputs."""

    inputs: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    source: str = "unknown"  # how introspection happened — for debugging

    @property
    def is_empty(self) -> bool:
        return not self.inputs

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "source": self.source,
        }


# Default fallback when nothing else works (matches the historic mock value
# but is now explicit rather than silently embedded in the adapter)
DEFAULT_FALLBACK_INPUT = TensorSpec(
    name="input",
    shape=(1, 3, 224, 224),
    dtype="float32",
)


# ─── ONNX introspection (no SDK needed) ──────────────────────────────────────


def _onnx_dtype_to_numpy(onnx_dtype: int) -> str:
    """Map ONNX TensorProto.DataType enum → numpy dtype string."""
    # See https://github.com/onnx/onnx/blob/main/onnx/onnx-ml.proto
    mapping = {
        1: "float32",   # FLOAT
        2: "uint8",     # UINT8
        3: "int8",      # INT8
        4: "uint16",    # UINT16
        5: "int16",     # INT16
        6: "int32",     # INT32
        7: "int64",     # INT64
        9: "bool",      # BOOL
        10: "float16",  # FLOAT16
        11: "float64",  # DOUBLE
        12: "uint32",   # UINT32
        13: "uint64",   # UINT64
        16: "bfloat16", # BFLOAT16
    }
    return mapping.get(onnx_dtype, "float32")


def introspect_onnx(model_path: Path) -> ModelIO | None:
    """Read an ONNX model's input/output tensor specs.

    Returns None if the ``onnx`` package isn't installed (the package
    is in the [real] extras, not core deps, so we don't assume it).
    """
    try:
        import onnx
    except ImportError:
        logger.debug("onnx package not installed; skipping ONNX introspection")
        return None

    try:
        model = onnx.load(str(model_path))
    except Exception as e:
        logger.debug("onnx.load failed for %s: %s", model_path, e)
        return None

    def _spec_from_value_info(vi: Any) -> TensorSpec:
        ttype = vi.type.tensor_type
        dtype = _onnx_dtype_to_numpy(ttype.elem_type)
        shape: list[int] = []
        for dim in ttype.shape.dim:
            # Either dim_value (concrete int) or dim_param (symbolic, e.g. 'batch_size')
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                # Symbolic dimension — assume 1 (batch) for now
                shape.append(1)
        return TensorSpec(name=vi.name, shape=tuple(shape), dtype=dtype)

    inputs = [_spec_from_value_info(vi) for vi in model.graph.input]
    outputs = [_spec_from_value_info(vi) for vi in model.graph.output]
    # Filter out initializers (which are listed as inputs in older ONNX models)
    init_names = {init.name for init in model.graph.initializer}
    inputs = [t for t in inputs if t.name not in init_names]
    return ModelIO(inputs=inputs, outputs=outputs, source="onnx-py")


# ─── DLC introspection (uses snpe-dlc-info if available) ─────────────────────


def introspect_dlc(model_path: Path, sdk_root: Path | str | None = None) -> ModelIO | None:
    """Run ``snpe-dlc-info`` and parse the input/output dimensions.

    Returns None if the tool isn't on PATH or if parsing fails. The
    parser is permissive — it looks for the ``Inputs:`` / ``Outputs:``
    sections in the tool's text output.
    """
    tool_paths = [
        shutil.which("snpe-dlc-info"),
        shutil.which("qairt-dlc-info"),
    ]
    if sdk_root:
        sdk = Path(sdk_root)
        for arch in ("x86_64-linux-clang", "x86_64-windows-msvc", "aarch64-windows-msvc"):
            for tool in ("snpe-dlc-info", "qairt-dlc-info"):
                cand = sdk / "bin" / arch / tool
                if cand.exists():
                    tool_paths.append(str(cand))

    tool = next((t for t in tool_paths if t), None)
    if tool is None:
        logger.debug("snpe-dlc-info not found; skipping DLC introspection")
        return None

    try:
        result = subprocess.run(
            [tool, "-i", str(model_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug("snpe-dlc-info failed: %s", e)
        return None

    if result.returncode != 0:
        logger.debug("snpe-dlc-info returncode=%d stderr=%s", result.returncode, result.stderr)
        return None

    return _parse_dlc_info_output(result.stdout)


def _parse_dlc_info_output(stdout: str) -> ModelIO | None:
    """Parse snpe-dlc-info text output. Permissive — best-effort."""
    import re

    inputs: list[TensorSpec] = []
    outputs: list[TensorSpec] = []
    section: str | None = None

    # Typical format:
    #   Input Name        | Dimensions    | Type
    #   ----------------- | ------------- | -----
    #   input             | 1,3,224,224   | float32
    line_re = re.compile(r"^\s*([\w/.-]+)\s*\|\s*([\d, ]+)\s*\|\s*(\w+)")

    for raw in stdout.splitlines():
        low = raw.lower()
        if "input name" in low or low.strip().startswith("inputs:"):
            section = "in"
            continue
        if "output name" in low or low.strip().startswith("outputs:"):
            section = "out"
            continue
        if section is None:
            continue
        m = line_re.match(raw)
        if not m:
            # End of section if we hit a blank line after parsing started
            if not raw.strip() and (inputs or outputs):
                section = None
            continue
        name, dims, dtype = m.group(1), m.group(2), m.group(3)
        try:
            shape = tuple(int(x.strip()) for x in dims.split(",") if x.strip())
        except ValueError:
            continue
        spec = TensorSpec(name=name, shape=shape, dtype=dtype.lower().replace(" ", ""))
        if section == "in":
            inputs.append(spec)
        else:
            outputs.append(spec)

    if not inputs:
        return None
    return ModelIO(inputs=inputs, outputs=outputs, source="snpe-dlc-info")


# ─── Public introspect() entry point ─────────────────────────────────────────


def introspect_model(
    model_path: str | Path,
    sdk_root: Path | str | None = None,
) -> ModelIO:
    """Introspect a model and return its tensor specs.

    Tries each known introspection method in order. Always returns a
    ModelIO — the source field tells you which method succeeded; if
    everything fails, returns ``ModelIO(inputs=[DEFAULT_FALLBACK_INPUT], source='fallback')``
    so callers don't need to do their own None-handling.
    """
    p = Path(model_path)
    suffix = p.suffix.lower()

    if suffix == ".onnx":
        result = introspect_onnx(p)
        if result and not result.is_empty:
            return result

    if suffix in (".dlc", ".bin"):
        result = introspect_dlc(p, sdk_root=sdk_root)
        if result and not result.is_empty:
            return result

    # Fallback — best-effort default
    logger.info(
        "model_introspection_fallback",
        extra={
            "model_path": str(p),
            "suffix": suffix,
            "reason": "no introspection tool succeeded; using DEFAULT_FALLBACK_INPUT",
        },
    )
    return ModelIO(
        inputs=[DEFAULT_FALLBACK_INPUT],
        outputs=[],
        source="fallback",
    )


# ─── Input-list generation ───────────────────────────────────────────────────


def generate_random_input(spec: TensorSpec, *, seed: int | None = None) -> np.ndarray:
    """Generate a random tensor matching ``spec``.

    Distribution depends on dtype:
    - float types → standard normal (μ=0, σ=1)
    - int/uint types → uniform in the dtype's representable range,
      clipped to ±127 for int8 / ±32768 for int16 etc. so we don't
      hit pathological edge cases
    - bool → Bernoulli(0.5)
    """
    rng = np.random.default_rng(seed)
    dt = spec.numpy_dtype

    if np.issubdtype(dt, np.floating):
        return rng.standard_normal(spec.shape, dtype=np.float32).astype(dt)
    if np.issubdtype(dt, np.unsignedinteger):
        info = np.iinfo(dt)
        return rng.integers(low=info.min, high=info.max, size=spec.shape, dtype=dt)
    if np.issubdtype(dt, np.signedinteger):
        info = np.iinfo(dt)
        return rng.integers(low=info.min, high=info.max, size=spec.shape, dtype=dt)
    if dt == np.bool_:
        return rng.integers(low=0, high=2, size=spec.shape).astype(np.bool_)
    # Fallback — treat as float32
    return rng.standard_normal(spec.shape).astype(np.float32)


def write_input_list(
    model_io: ModelIO,
    output_dir: str | Path | None = None,
    *,
    num_samples: int = 1,
    calibration_data: dict[str, np.ndarray] | None = None,
    seed: int | None = 42,
) -> str:
    """Write per-tensor .raw files + an input_list.txt for SDK tools.

    Args:
        model_io: introspected model inputs (and optionally outputs)
        output_dir: where to write files. None → tempdir.
        num_samples: how many calibration / inference samples to write
            (one line per sample in the input_list.txt).
        calibration_data: if provided, dict of input_name → ndarray of
            shape ``(num_samples, *input_shape)`` to use instead of
            random data. Useful for AIMET PTQ calibration.
        seed: RNG seed for reproducible random inputs (when
            calibration_data is None).

    Returns:
        Absolute POSIX path to the input_list.txt file. The directory
        also contains ``inputN_<tensor>.raw`` files referenced by it.
    """
    if model_io.is_empty:
        raise ValueError("ModelIO has no inputs; cannot write input list.")

    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="quad_input_"))
    out.mkdir(parents=True, exist_ok=True)

    list_lines: list[str] = []
    for sample_idx in range(num_samples):
        # Each sample row in input_list.txt is a colon-separated list of
        # "tensor_name:=path/to/file.raw" entries — but for single-input
        # models the simpler form "path/to/file.raw" also works.
        entries: list[str] = []
        for tensor in model_io.inputs:
            file_name = (
                f"input{sample_idx}_{tensor.name}.raw"
                if num_samples > 1 or len(model_io.inputs) > 1
                else "input.raw"
            )
            file_path = out / file_name
            if calibration_data and tensor.name in calibration_data:
                arr = calibration_data[tensor.name]
                if arr.ndim > len(tensor.shape):
                    # Caller passed a batched array; pick this sample
                    arr = arr[sample_idx]
                arr = np.ascontiguousarray(arr.astype(tensor.numpy_dtype))
            else:
                # Each sample uses a different seed slice for variety
                sample_seed = None if seed is None else seed + sample_idx
                arr = generate_random_input(tensor, seed=sample_seed)
            arr.tofile(str(file_path))
            if len(model_io.inputs) > 1:
                entries.append(f"{tensor.name}:={file_path.as_posix()}")
            else:
                entries.append(file_path.as_posix())
        list_lines.append(" ".join(entries))

    list_path = out / "input_list.txt"
    list_path.write_text("\n".join(list_lines) + "\n")
    logger.debug(
        "input_list_written",
        extra={
            "path": list_path.as_posix(),
            "num_samples": num_samples,
            "num_inputs": len(model_io.inputs),
            "source": model_io.source,
        },
    )
    return list_path.as_posix()


def create_input_list_for_model(
    model_path: str | Path,
    *,
    sdk_root: Path | str | None = None,
    output_dir: str | Path | None = None,
    num_samples: int = 1,
    calibration_data: dict[str, np.ndarray] | None = None,
    seed: int | None = 42,
) -> tuple[str, ModelIO]:
    """One-stop helper: introspect + write input list.

    Returns ``(input_list_path, ModelIO)`` so the caller can also
    inspect what was discovered (useful for logging the introspection
    source — 'onnx-py' vs 'snpe-dlc-info' vs 'fallback').
    """
    model_io = introspect_model(model_path, sdk_root=sdk_root)
    list_path = write_input_list(
        model_io,
        output_dir=output_dir,
        num_samples=num_samples,
        calibration_data=calibration_data,
        seed=seed,
    )
    return list_path, model_io
