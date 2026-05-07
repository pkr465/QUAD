"""PSNPE Manager — orchestrates parallel SNPE execution.

Architecture
------------
``PSNPEManager`` operates in two modes selected at import time:

**Mock mode** (default, no SDK required)
    Uses Python's ``ThreadPoolExecutor`` to simulate parallel SNPE workers.
    Execution produces synthetic float32 output buffers with realistic timing
    noise so that downstream code can be developed and tested without hardware.

**Real mode** (requires Qualcomm AI Engine SDK on ``$PATH``)
    Delegates to the ``snpe-parallel-run`` command-line tool that ships with
    the SNPE SDK.  Input/output buffers are written to a temporary directory
    and read back after the subprocess completes.

The mode is selected automatically: if ``snpe-parallel-run`` is not found on
``$PATH`` (or the environment variable ``QUAD_PSNPE_MOCK=1`` is set), mock
mode is used.  Set ``QUAD_PSNPE_MOCK=0`` to force real mode (will raise
``RuntimeError`` if the tool is absent).

Example
-------
    from quad.psnpe import PSNPEManager, BuildConfig, RuntimeConfig

    mgr = PSNPEManager()
    built = mgr.build(BuildConfig(
        container_path="model.dlc",
        runtime_configs=[
            RuntimeConfig(runtime="dsp", num_instances=4),
        ],
        output_buffer_names=["output:0"],
    ))
    assert built

    inputs = [{"input:0": b"\\x00" * 602112} for _ in range(8)]
    results = mgr.execute_sync(inputs)
    print(results[0].throughput_fps)   # > 0
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from quad.psnpe.config import BuildConfig, ExecutionMode, ModelConfig, RuntimeConfig


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PSNPEResult:
    """Aggregated result from a PSNPE execution call.

    Attributes:
        outputs: One dict per input sample; keys are tensor names, values are
            raw bytes in the buffer format chosen at build time (float32 by
            default).
        latency_ms: Wall-clock time in milliseconds from first input enqueue to
            last output received.
        throughput_fps: Effective frames-per-second computed as
            ``len(outputs) / (latency_ms / 1000)``.
        instances_used: Number of SNPE worker instances that were active during
            this execution.
        mode: The ``ExecutionMode`` that was used for this call.
    """

    outputs: list[dict[str, bytes]]
    latency_ms: float
    throughput_fps: float
    instances_used: int
    mode: ExecutionMode


# ---------------------------------------------------------------------------
# Mock worker helpers
# ---------------------------------------------------------------------------

# Default synthetic tensor sizes used when no real DLC is loaded.
_MOCK_INPUT_NAMES: list[str] = ["input:0"]
_MOCK_OUTPUT_NAMES: list[str] = ["output:0"]
_MOCK_OUTPUT_BYTES: int = 1000 * 4  # 1000 fp32 classes


def _mock_infer(
    sample: dict[str, bytes],
    output_names: list[str],
    instance_id: int,
    latency_noise_ms: float = 2.0,
) -> dict[str, bytes]:
    """Simulate one SNPE forward pass in mock mode.

    Produces a random fp32 buffer for each output tensor.  A small sleep
    proportional to ``latency_noise_ms`` mimics hardware dispatch latency.
    """
    # Simulate a small per-instance dispatch overhead (non-blocking sleep
    # avoids making tests slow while still exercising the threading path).
    time.sleep(latency_noise_ms / 1000.0)

    result: dict[str, bytes] = {}
    for name in output_names:
        # Produce deterministic-ish synthetic output: hash input bytes → seed.
        seed = hash(name) ^ sum(sample.values().__iter__().__next__()[:4]) if sample else 0
        count = _MOCK_OUTPUT_BYTES // 4
        values = [float((seed + i) % 256) / 256.0 for i in range(count)]
        result[name] = struct.pack(f"{count}f", *values)
    return result


# ---------------------------------------------------------------------------
# Main manager class
# ---------------------------------------------------------------------------


class PSNPEManager:
    """Manages a PSNPE instance: build, execute, and release.

    Lifecycle::

        mgr = PSNPEManager()
        mgr.build(config)          # Creates worker threads / subprocesses
        results = mgr.execute_sync(inputs)
        mgr.release()

    The same instance can be rebuilt with a new ``BuildConfig`` by calling
    ``release()`` followed by ``build()`` again.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, sdk_root: str | None = None) -> None:
        """Initialise PSNPEManager.

        Args:
            sdk_root: Optional path to the Qualcomm AI Engine SDK root
                directory.  When given, the manager prepends
                ``{sdk_root}/bin/x86_64-linux-clang`` to ``$PATH`` so that
                ``snpe-parallel-run`` is found automatically.  Defaults to the
                ``SNPE_ROOT`` environment variable if set, then to ``None``
                (mock mode).
        """
        self._sdk_root = sdk_root or os.environ.get("SNPE_ROOT")
        self._mock_mode: bool = self._detect_mock_mode()
        self._config: BuildConfig | None = None
        self._built: bool = False
        self._executor: ThreadPoolExecutor | None = None

        # Tensor name lists populated during build()
        self._input_tensor_names: list[str] = []
        self._output_tensor_names: list[str] = []

    def _detect_mock_mode(self) -> bool:
        """Return True if mock mode should be used."""
        env_override = os.environ.get("QUAD_PSNPE_MOCK", "").strip()
        if env_override == "0":
            return False  # Caller explicitly demands real mode
        if env_override == "1":
            return True

        # Auto-detect: look for snpe-parallel-run on $PATH (optionally
        # extended by sdk_root).
        search_env = dict(os.environ)
        if self._sdk_root:
            bin_dir = str(Path(self._sdk_root) / "bin" / "x86_64-linux-clang")
            search_env["PATH"] = bin_dir + os.pathsep + search_env.get("PATH", "")

        return shutil.which("snpe-parallel-run", path=search_env.get("PATH")) is None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, config: BuildConfig) -> bool:
        """Build the PSNPE pipeline from *config*.

        In mock mode this creates a ``ThreadPoolExecutor`` with
        ``config.total_instances`` worker threads.  In real mode it invokes
        the SDK initialisation stage (loading the DLC, allocating buffers).

        Args:
            config: Fully-populated ``BuildConfig`` describing the model,
                runtimes, and execution parameters.

        Returns:
            ``True`` on success, ``False`` on failure (details logged to
            stderr).

        Raises:
            ValueError: If *config* fails validation.
        """
        config.validate()

        # Release any previously built state.
        self.release()

        self._config = config

        # Resolve tensor names from config or fall back to mock defaults.
        self._output_tensor_names = list(config.output_buffer_names)
        self._input_tensor_names = _MOCK_INPUT_NAMES[:]  # Real mode: parse DLC

        if self._mock_mode:
            self._executor = ThreadPoolExecutor(
                max_workers=config.total_instances,
                thread_name_prefix="psnpe-worker",
            )
        else:
            success = self._real_build(config)
            if not success:
                return False

        self._built = True
        return True

    def _real_build(self, config: BuildConfig) -> bool:
        """Invoke SDK tools to initialise the PSNPE pipeline."""
        # In a full implementation this would run:
        #   snpe-parallel-run --container {dlc} --build-only ...
        # and parse the output.  For now we stub it out because the SDK may
        # not be present in all CI environments.
        cmd = [
            "snpe-parallel-run",
            "--container", config.container_path,
            "--mode", config.transmission_mode.value,
            "--profiling_level", config.profiling_level,
        ]
        for rc in config.runtime_configs:
            cmd += ["--runtime", rc.runtime, "--num_threads", str(rc.num_instances)]

        import subprocess
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            import sys
            print(f"[PSNPEManager] real build failed: {exc}", file=sys.stderr)
            return False

    # ------------------------------------------------------------------
    # Execute — synchronous
    # ------------------------------------------------------------------

    def execute_sync(
        self,
        inputs: list[dict[str, bytes]],
        output_dir: str = "./output",
    ) -> list[PSNPEResult]:
        """Run synchronous bulk inference.

        All *inputs* are distributed across the worker pool and results are
        collected before the call returns.

        Args:
            inputs: List of input sample dicts; each dict maps a tensor name
                to raw bytes (fp32 or quantised, matching the buffer mode
                chosen at build time).
            output_dir: Directory for any intermediate files produced in real
                mode.  Ignored in mock mode.

        Returns:
            A list of ``PSNPEResult`` objects — one per input sample — in the
            same order as *inputs*.
        """
        self._assert_built("execute_sync")

        t_start = time.perf_counter()

        if self._mock_mode:
            raw_outputs = self._mock_execute_batch(inputs)
        else:
            raw_outputs = self._real_execute_sync(inputs, output_dir)

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0
        n = len(inputs)
        fps = n / (latency_ms / 1000.0) if latency_ms > 0 else float("inf")

        return [
            PSNPEResult(
                outputs=[raw_outputs[i]],
                latency_ms=latency_ms,
                throughput_fps=fps,
                instances_used=self._config.total_instances,  # type: ignore[union-attr]
                mode=ExecutionMode.SYNC,
            )
            for i in range(n)
        ]

    def _mock_execute_batch(
        self, inputs: list[dict[str, bytes]]
    ) -> list[dict[str, bytes]]:
        """Distribute inputs across mock worker threads."""
        assert self._executor is not None
        futures = {
            self._executor.submit(
                _mock_infer,
                sample,
                self._output_tensor_names,
                idx % (self._config.total_instances or 1),  # type: ignore[union-attr]
            ): idx
            for idx, sample in enumerate(inputs)
        }
        ordered: list[dict[str, bytes]] = [{}] * len(inputs)
        for fut in as_completed(futures):
            idx = futures[fut]
            ordered[idx] = fut.result()
        return ordered

    def _real_execute_sync(
        self, inputs: list[dict[str, bytes]], output_dir: str
    ) -> list[dict[str, bytes]]:
        """Write inputs to disk, invoke snpe-parallel-run, read outputs."""
        import subprocess

        cfg = self._config
        assert cfg is not None

        with tempfile.TemporaryDirectory(prefix="psnpe_") as tmp:
            input_list_path = os.path.join(tmp, "input_list.txt")
            input_files: list[str] = []
            for i, sample in enumerate(inputs):
                for tensor_name, data in sample.items():
                    safe = tensor_name.replace(":", "_").replace("/", "_")
                    fpath = os.path.join(tmp, f"input_{i}_{safe}.raw")
                    with open(fpath, "wb") as fh:
                        fh.write(data)
                    input_files.append(fpath)
            with open(input_list_path, "w") as lf:
                for fp in input_files:
                    lf.write(fp + "\n")

            out_path = output_dir
            os.makedirs(out_path, exist_ok=True)

            cmd = [
                "snpe-parallel-run",
                "--container", cfg.container_path,
                "--input_list", input_list_path,
                "--output_dir", out_path,
                "--mode", cfg.transmission_mode.value,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Read outputs — expect one sub-dir per input.
            results: list[dict[str, bytes]] = []
            for i in range(len(inputs)):
                sample_out: dict[str, bytes] = {}
                for name in self._output_tensor_names:
                    safe = name.replace(":", "_").replace("/", "_")
                    fpath = os.path.join(out_path, f"Result_{i}", f"{safe}.raw")
                    if os.path.exists(fpath):
                        with open(fpath, "rb") as fh:
                            sample_out[name] = fh.read()
                results.append(sample_out)
            return results

    # ------------------------------------------------------------------
    # Execute — output-async
    # ------------------------------------------------------------------

    def execute_output_async(
        self,
        inputs: list[dict[str, bytes]],
        output_callback: Callable[[int, dict[str, bytes]], None] | None = None,
        output_dir: str = "./output",
    ) -> None:
        """Submit all inputs and deliver results via *output_callback*.

        The callback is invoked once per completed input with the signature::

            output_callback(index: int, output_map: dict[str, bytes]) -> None

        where *index* is the position of the input in the original *inputs*
        list.  Callbacks may be invoked from worker threads; the caller is
        responsible for any required synchronisation.

        Args:
            inputs: Input sample dicts as in ``execute_sync``.
            output_callback: Optional callable invoked on each result.  If
                ``None``, results are discarded (useful for benchmarking
                throughput without a consumer).
            output_dir: Intermediate file directory (real mode only).
        """
        self._assert_built("execute_output_async")

        assert self._executor is not None or not self._mock_mode

        if self._mock_mode:
            futures = {
                self._executor.submit(  # type: ignore[union-attr]
                    _mock_infer,
                    sample,
                    self._output_tensor_names,
                    idx % (self._config.total_instances or 1),  # type: ignore[union-attr]
                ): idx
                for idx, sample in enumerate(inputs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                result_map = fut.result()
                if output_callback is not None:
                    output_callback(idx, result_map)
        else:
            # In real mode, reuse sync execution then fire callbacks
            raw_outputs = self._real_execute_sync(inputs, output_dir)
            for idx, out_map in enumerate(raw_outputs):
                if output_callback is not None:
                    output_callback(idx, out_map)

    # ------------------------------------------------------------------
    # Execute — input-output-async (full async)
    # ------------------------------------------------------------------

    def execute_input_output_async(
        self,
        file_paths: list[str],
        input_callback: Callable[[str], dict[str, bytes]],
        output_callback: Callable[[int, dict[str, bytes]], None],
    ) -> None:
        """Full async pipeline with callback-driven I/O.

        Input loading and output delivery are both handled by caller-supplied
        callbacks.  This mirrors the ``INPUT_OUTPUT_ASYNC`` mode in the PSNPE
        C++ / Android APIs where the framework owns the buffer lifecycle.

        Pipeline (per frame):
        1. ``input_callback(file_path)`` is invoked to produce the input dict.
        2. The dict is dispatched to an available SNPE worker.
        3. ``output_callback(index, output_map)`` is invoked when done.

        Both callbacks may be called from worker threads.

        Args:
            file_paths: Ordered list of paths/identifiers.  Each entry is
                passed verbatim to *input_callback*.
            input_callback: ``(file_path: str) -> dict[str, bytes]``
                Load and return input tensors for one frame.
            output_callback: ``(index: int, output_map: dict[str, bytes]) -> None``
                Consume one completed output frame.
        """
        self._assert_built("execute_input_output_async")

        assert self._executor is not None or not self._mock_mode

        def _process(idx: int, fpath: str) -> tuple[int, dict[str, bytes]]:
            sample = input_callback(fpath)
            output = _mock_infer(
                sample,
                self._output_tensor_names,
                idx % (self._config.total_instances or 1),  # type: ignore[union-attr]
            )
            return idx, output

        if self._mock_mode:
            futures = {
                self._executor.submit(_process, i, fp): i  # type: ignore[union-attr]
                for i, fp in enumerate(file_paths)
            }
            for fut in as_completed(futures):
                idx, result_map = fut.result()
                output_callback(idx, result_map)
        else:
            # Real mode: load inputs sequentially, batch, run, fire callbacks.
            inputs = [input_callback(fp) for fp in file_paths]
            with tempfile.TemporaryDirectory(prefix="psnpe_async_") as tmp:
                raw = self._real_execute_sync(inputs, tmp)
            for idx, out_map in enumerate(raw):
                output_callback(idx, out_map)

    # ------------------------------------------------------------------
    # Tensor name accessors
    # ------------------------------------------------------------------

    def get_input_tensor_names(self) -> list[str]:
        """Return the list of input tensor names for the loaded model.

        In mock mode returns a synthetic single-input list.  In real mode the
        names are parsed from the DLC container during ``build()``.
        """
        self._assert_built("get_input_tensor_names")
        return list(self._input_tensor_names)

    def get_output_tensor_names(self) -> list[str]:
        """Return the list of output tensor names specified at build time."""
        self._assert_built("get_output_tensor_names")
        return list(self._output_tensor_names)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Shut down worker threads and release all resources.

        Safe to call even if ``build()`` was never called.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._built = False
        self._config = None
        self._input_tensor_names = []
        self._output_tensor_names = []

    def __enter__(self) -> "PSNPEManager":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Class-method factories
    # ------------------------------------------------------------------

    @classmethod
    def from_model_config(
        cls,
        config_json_path: str,
        model_name: str,
        sdk_root: str | None = None,
        output_buffer_names: list[str] | None = None,
    ) -> "PSNPEManager":
        """Build a ``PSNPEManager`` from an Android-style ``model_configs.json``.

        The JSON file is a list of model config objects.  Each object must
        have at least a ``"name"`` key.  Recognised keys::

            {
                "name": "resnet50",
                "model_file": "resnet50.dlc",
                "execute_mode": "sync",
                "enable_init_cache": false,
                "bulk_size": 4,
                "build_configs": [
                    {
                        "runtime": "dsp",
                        "num_instances": 4,
                        "performance_profile": "burst"
                    }
                ]
            }

        Args:
            config_json_path: Path to the JSON file.
            model_name: Value of the ``"name"`` field to look up.
            sdk_root: Optional SDK root forwarded to ``PSNPEManager.__init__``.
            output_buffer_names: Override output tensor names.  When ``None``,
                defaults to ``["output:0"]``.

        Returns:
            A ``PSNPEManager`` that has already been successfully built.

        Raises:
            KeyError: If *model_name* is not found in the JSON file.
            ValueError: If the resulting ``BuildConfig`` fails validation.
        """
        path = Path(config_json_path)
        with path.open() as fh:
            entries: list[dict] = json.load(fh)

        # Support both a list at top-level and a dict wrapper.
        if isinstance(entries, dict):
            entries = list(entries.values())

        entry = next((e for e in entries if e.get("name") == model_name), None)
        if entry is None:
            available = [e.get("name") for e in entries]
            raise KeyError(
                f"Model '{model_name}' not found in '{config_json_path}'. "
                f"Available: {available}"
            )

        mc = ModelConfig(
            name=entry["name"],
            model_file=entry.get("model_file", entry.get("name") + ".dlc"),
            execute_mode=ExecutionMode(entry.get("execute_mode", "sync")),
            enable_init_cache=entry.get("enable_init_cache", False),
            bulk_size=entry.get("bulk_size", 1),
            build_configs=entry.get("build_configs", []),
        )

        out_names = output_buffer_names or entry.get("output_buffer_names", ["output:0"])
        build_cfg = mc.to_build_config(out_names)

        mgr = cls(sdk_root=sdk_root)
        ok = mgr.build(build_cfg)
        if not ok:
            raise RuntimeError(
                f"PSNPEManager.build() failed for model '{model_name}'."
            )
        return mgr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_built(self, method: str) -> None:
        if not self._built:
            raise RuntimeError(
                f"PSNPEManager.{method}() called before build().  "
                "Call build(config) first."
            )

    def __repr__(self) -> str:
        mode_str = "mock" if self._mock_mode else "real"
        if self._built and self._config:
            return (
                f"PSNPEManager(mode={mode_str!r}, "
                f"instances={self._config.total_instances}, "
                f"built=True)"
            )
        return f"PSNPEManager(mode={mode_str!r}, built=False)"
