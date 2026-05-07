"""PSNPE configuration data-classes and enumerations.

These mirror the structures used in the official PSNPE C / C++ / Android APIs:

* ``ExecutionMode``  — maps to ``SNPE::ExecutionMode`` in the C++ header and to
  the ``execute_mode`` field in the Android ``model_configs.json`` schema.
* ``RuntimeConfig`` — one runtime target with a pool of N identical SNPE
  instances; corresponds to ``RuntimeConfig`` in the C tutorial.
* ``BuildConfig``   — full set of parameters passed to ``PSNPE::Builder`` in
  C++, and to ``PsnpeBuilder`` in the Android tutorial.
* ``ModelConfig``   — matches the JSON schema used by the Android sample app
  (``model_configs.json``).
* ``PSNPEConfig``   — thin wrapper kept for backward-compatibility, delegates
  to ``BuildConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ExecutionMode(str, Enum):
    """Execution mode for PSNPE pipeline.

    Corresponds to ``zdl::runtime_lib::ExecutionMode`` in the C++ API.

    SYNC
        All inputs are enqueued and ``execute()`` blocks until all outputs are
        ready.  Simplest mode; best for offline batch jobs.

    OUTPUT_ASYNC
        ``execute()`` returns immediately; results are delivered via a callback
        as each worker thread completes.  Good for producer/consumer pipelines
        where input loading is fast.

    INPUT_OUTPUT_ASYNC
        Both input loading *and* output delivery are driven by callbacks.
        Maximises overlap between I/O and compute — recommended for real-time
        camera / sensor pipelines.
    """

    SYNC = "sync"
    OUTPUT_ASYNC = "outputAsync"
    INPUT_OUTPUT_ASYNC = "inputOutputAsync"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    """Configuration for a single SNPE runtime target.

    One ``RuntimeConfig`` entry maps to one *runtime pool* — i.e. a set of
    ``num_instances`` SNPE instances all targeting the same hardware runtime.

    Attributes:
        runtime: Target runtime string.  Common values: ``"cpu"``, ``"gpu"``,
            ``"dsp"`` (HVX), ``"aic"`` (HMX/AIC).
        num_instances: Number of parallel SNPE instances to create for this
            runtime.  Increasing this value raises throughput at the cost of
            more memory.
        performance_profile: SNPE performance profile hint.  One of
            ``"burst"``, ``"balanced"``, ``"sustained_high_performance"``,
            ``"high_performance"``, ``"power_saver"``, ``"low_power_saver"``,
            ``"high_power_saver"``, ``"low_balanced"``, ``"default"``.
        enable_cpu_fallback: Fall back to CPU for unsupported ops when True.
        user_buffer_mode: Buffer format for user-provided tensors.  One of
            ``"float"`` (fp32), ``"tf8"`` (8-bit quantised), ``"uint8"``.
    """

    runtime: str
    num_instances: int = 1
    performance_profile: str = "burst"
    enable_cpu_fallback: bool = True
    user_buffer_mode: str = "float"


@dataclass
class BuildConfig:
    """Full build configuration passed to the PSNPE builder.

    Mirrors ``PSNPE::Builder`` in the C++ tutorial and ``PsnpeBuilder`` in the
    Android tutorial.

    Attributes:
        container_path: Path to the compiled ``.dlc`` model container.
        runtime_configs: One or more runtime pools.  The list is processed in
            order; the first runtime that can execute a given layer wins.
        output_buffer_names: Names of the output tensors to collect.  Must
            match the tensor names embedded in the DLC.
        transmission_mode: Pipeline execution mode (sync / output-async /
            input-output-async).
        enable_init_cache: Persist the initialisation cache to disk so that
            subsequent ``build()`` calls are faster.
        profiling_level: SNPE profiling verbosity: ``"off"``, ``"basic"``,
            ``"moderate"``, ``"detailed"``.
        output_thread_numbers: Number of threads that drain the output queue in
            OUTPUT_ASYNC and INPUT_OUTPUT_ASYNC modes.
        input_thread_numbers: Number of threads that feed the input queue in
            INPUT_OUTPUT_ASYNC mode.
        platform_options: Opaque key-value string forwarded to the underlying
            SNPE platform interface (e.g. ``"unsignedPD:ON"``).
        bulk_size: Number of input frames grouped into one *bulk* submitted to
            the worker pool in a single ``execute()`` call.  Increasing this
            amortises dispatch overhead at the cost of latency.
    """

    container_path: str
    runtime_configs: list[RuntimeConfig]
    output_buffer_names: list[str]
    transmission_mode: ExecutionMode = ExecutionMode.SYNC
    enable_init_cache: bool = False
    profiling_level: str = "off"
    output_thread_numbers: int = 1
    input_thread_numbers: int = 1
    platform_options: str = ""
    bulk_size: int = 1

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #

    @property
    def total_instances(self) -> int:
        """Total number of SNPE worker instances across all runtime pools."""
        return sum(rc.num_instances for rc in self.runtime_configs)

    def validate(self) -> None:
        """Raise ``ValueError`` if the config is internally inconsistent."""
        if not self.container_path:
            raise ValueError("container_path must not be empty.")
        if not self.runtime_configs:
            raise ValueError("At least one RuntimeConfig is required.")
        if not self.output_buffer_names:
            raise ValueError("output_buffer_names must not be empty.")
        if self.bulk_size < 1:
            raise ValueError(f"bulk_size must be >= 1, got {self.bulk_size}.")
        if self.output_thread_numbers < 1:
            raise ValueError("output_thread_numbers must be >= 1.")
        if self.input_thread_numbers < 1:
            raise ValueError("input_thread_numbers must be >= 1.")
        valid_profiles = {
            "off", "basic", "moderate", "detailed",
        }
        if self.profiling_level not in valid_profiles:
            raise ValueError(
                f"profiling_level must be one of {valid_profiles}, "
                f"got '{self.profiling_level}'."
            )


@dataclass
class ModelConfig:
    """Model configuration entry matching the Android ``model_configs.json`` schema.

    The Android PSNPE sample application reads a JSON file where each object
    corresponds to one ``ModelConfig``.  This dataclass mirrors that schema so
    that ``PSNPEManager.from_model_config()`` can deserialise such files
    directly.

    Attributes:
        name: Logical model name used as a lookup key.
        model_file: Path to the ``.dlc`` container (relative or absolute).
        execute_mode: Execution mode for this model's PSNPE pipeline.
        enable_init_cache: Whether to use the SNPE init cache for this model.
        bulk_size: Number of inputs processed together per ``execute()`` call.
        build_configs: List of raw runtime-config dicts, each with at least a
            ``"runtime"`` key.  Deserialised into ``RuntimeConfig`` objects by
            ``PSNPEManager.from_model_config()``.
    """

    name: str
    model_file: str
    execute_mode: ExecutionMode = ExecutionMode.SYNC
    enable_init_cache: bool = False
    bulk_size: int = 1
    build_configs: list[dict] = field(default_factory=list)

    def to_build_config(self, output_buffer_names: list[str]) -> BuildConfig:
        """Convert to a ``BuildConfig`` suitable for ``PSNPEManager.build()``."""
        runtime_configs = [
            RuntimeConfig(
                runtime=bc.get("runtime", "cpu"),
                num_instances=bc.get("num_instances", 1),
                performance_profile=bc.get("performance_profile", "burst"),
                enable_cpu_fallback=bc.get("enable_cpu_fallback", True),
                user_buffer_mode=bc.get("user_buffer_mode", "float"),
            )
            for bc in self.build_configs
        ]
        return BuildConfig(
            container_path=self.model_file,
            runtime_configs=runtime_configs or [RuntimeConfig(runtime="cpu")],
            output_buffer_names=output_buffer_names,
            transmission_mode=self.execute_mode,
            enable_init_cache=self.enable_init_cache,
            bulk_size=self.bulk_size,
        )


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------


@dataclass
class PSNPEConfig:
    """Thin wrapper around ``BuildConfig`` kept for backward compatibility.

    New code should use ``BuildConfig`` directly.
    """

    build_config: BuildConfig

    @classmethod
    def from_build_config(cls, cfg: BuildConfig) -> "PSNPEConfig":
        return cls(build_config=cfg)
