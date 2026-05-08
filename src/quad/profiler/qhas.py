"""QHAS Profiling — QNN HTP Analysis Summary for SNPE workflow.

Based on SNPE "QHAS Profiling" documentation (80-63442-10 Rev AH, Apr 13 2026).

QHAS (QNN HTP Analysis Summary) is a two-step profiling workflow for HTP runtime:

Step 1 — Graph prepare with schematic generation:
  snpe-dlc-graph-prepare --input_dlc model.dlc --output_dlc model_cache.dlc
      --htp_socs sm8750 --profiling_level qhas
  Output: [MODEL]_schematic.bin  (required for chrometrace generation)

Step 2 — Runtime artifact collection:
  snpe-net-run --container model_cache.dlc --input_list input_list.txt
      --profiling_level qhas --use_dsp
  Output: ./profilelogs/qnn-profiling-data.log

Step 3 — Chrometrace generation (on-device):
  qnn-profile-viewer --config config.json
      --reader [SDK]/lib/[TARGET]/libQnnHtpOptraceProfilingReader.so
      --input_log ./profilelogs/qnn-profiling-data.log
      --schematic ./[MODEL]_schematic.bin
      --output ./chrometrace.json

Config options (config.json "features" block) control additional outputs:
  - enable_input_output_flow_events  (default off) — I/O dependency flow events
  - enable_sequencer_flow_events     (default off) — sequencer ordering flow events
  - htp_json                         (default on)  — topology + op-by-op HTP graph info
  - runtrace                         (default on)  — execution + preemption events per core
  - memory_info                      (default on)  — memory bandwidth + allocation graphs
  - traceback                        (default on)  — trace back to source framework
  - qhas_schema                      (default off) — qhas_schema.json for QHAS JSON validation
  - qhas_json                        (default off) — [model]_qnn_htp_analysis_summary.json

Note: enable_input_output_flow_events and enable_sequencer_flow_events require
the legacy Chrome tracing UI to render correctly.

See also: HTP Optrace Profiling in QNN → Backend → HTP documentation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quad.profiler.levels import ProfilingLevel


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

QHAS_PROFILING_LEVEL = ProfilingLevel.QHAS.value
QHAS_PROFILELOGS_DIR = "profilelogs"
QHAS_LOG_FILENAME = "qnn-profiling-data.log"
QHAS_READER_LIB = "libQnnHtpOptraceProfilingReader.so"

# snpe-dlc-graph-prepare output naming: [MODEL]_schematic.bin
# e.g. inception_v3_quantized.dlc → inception_v3_quantized_schematic.bin
SCHEMATIC_SUFFIX = "_schematic.bin"


# ══════════════════════════════════════════════════════════════════════════════
# Config Model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QHASConfig:
    """Configuration for qnn-profile-viewer chrometrace generation.

    Written to config.json and passed via --config flag.

    Feature defaults match documentation:
      - on  by default: htp_json, runtrace, memory_info, traceback
      - off by default: enable_input_output_flow_events,
                        enable_sequencer_flow_events, qhas_schema, qhas_json

    Note: flow event features require the legacy Chrome tracing UI.
    """
    # Default-OFF features
    enable_input_output_flow_events: bool = False
    enable_sequencer_flow_events: bool = False
    qhas_schema: bool = False
    qhas_json: bool = False

    # Default-ON features
    htp_json: bool = True
    runtrace: bool = True
    memory_info: bool = True
    traceback: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the config.json format expected by qnn-profile-viewer."""
        return {
            "features": {
                "enable_input_output_flow_events": self.enable_input_output_flow_events,
                "enable_sequencer_flow_events": self.enable_sequencer_flow_events,
                "htp_json": self.htp_json,
                "runtrace": self.runtrace,
                "memory_info": self.memory_info,
                "traceback": self.traceback,
                "qhas_schema": self.qhas_schema,
                "qhas_json": self.qhas_json,
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string suitable for writing to config.json."""
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: str) -> None:
        """Write config to a JSON file at the given path."""
        Path(path).write_text(self.to_json())

    @classmethod
    def full(cls) -> "QHASConfig":
        """Create a config with ALL features enabled (maximum output)."""
        return cls(
            enable_input_output_flow_events=True,
            enable_sequencer_flow_events=True,
            htp_json=True,
            runtrace=True,
            memory_info=True,
            traceback=True,
            qhas_schema=True,
            qhas_json=True,
        )

    @classmethod
    def minimal(cls) -> "QHASConfig":
        """Create a config with only the default-on features enabled."""
        return cls()  # All defaults

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QHASConfig":
        """Parse a config.json dict into a QHASConfig instance."""
        features = data.get("features", data)
        return cls(
            enable_input_output_flow_events=features.get(
                "enable_input_output_flow_events", False),
            enable_sequencer_flow_events=features.get(
                "enable_sequencer_flow_events", False),
            htp_json=features.get("htp_json", True),
            runtrace=features.get("runtrace", True),
            memory_info=features.get("memory_info", True),
            traceback=features.get("traceback", True),
            qhas_schema=features.get("qhas_schema", False),
            qhas_json=features.get("qhas_json", False),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "QHASConfig":
        """Parse a JSON string into a QHASConfig instance."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "QHASConfig":
        """Load a QHASConfig from a config.json file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def requires_legacy_ui(self) -> bool:
        """True if any enabled feature requires the legacy Chrome tracing UI."""
        return self.enable_input_output_flow_events or self.enable_sequencer_flow_events

    @property
    def enabled_features(self) -> list[str]:
        """Names of all features that are currently enabled."""
        features = []
        if self.enable_input_output_flow_events:
            features.append("enable_input_output_flow_events")
        if self.enable_sequencer_flow_events:
            features.append("enable_sequencer_flow_events")
        if self.htp_json:
            features.append("htp_json")
        if self.runtrace:
            features.append("runtrace")
        if self.memory_info:
            features.append("memory_info")
        if self.traceback:
            features.append("traceback")
        if self.qhas_schema:
            features.append("qhas_schema")
        if self.qhas_json:
            features.append("qhas_json")
        return features


# ══════════════════════════════════════════════════════════════════════════════
# File Path Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_schematic_path(dlc_path: str) -> str:
    """Derive the schematic .bin path that snpe-dlc-graph-prepare will produce.

    Naming rule: [MODEL]_schematic.bin in the current working directory.
    E.g. inception_v3_quantized.dlc → inception_v3_quantized_schematic.bin

    Args:
        dlc_path: Path to the input .dlc file

    Returns:
        Expected path to the generated schematic .bin file
    """
    stem = Path(dlc_path).stem
    return f"{stem}{SCHEMATIC_SUFFIX}"


def get_profilelogs_dir(working_dir: str = ".") -> str:
    """Return the profilelogs directory path for a given working directory.

    snpe-net-run with --profiling_level qhas writes artifacts to
    ./profilelogs/qnn-profiling-data.log in the current working directory.

    Returns a POSIX-style path (consistent across OS) since the consumer
    is the SDK tool which runs on POSIX targets.
    """
    return (Path(working_dir) / QHAS_PROFILELOGS_DIR).as_posix()


def get_log_path(working_dir: str = ".") -> str:
    """Return the full path to the QHAS profiling log file (POSIX-style)."""
    return (Path(working_dir) / QHAS_PROFILELOGS_DIR / QHAS_LOG_FILENAME).as_posix()


def get_reader_lib_path(sdk_root: str, target: str = "aarch64-android") -> str:
    """Return the path to the QHAS profiling reader shared library.

    Args:
        sdk_root: Root of the QNN/QAIRT SDK installation
        target: Target architecture (e.g. 'aarch64-android', 'aarch64-ubuntu-gcc11.4')

    Returns:
        Path to libQnnHtpOptraceProfilingReader.so (POSIX-style).
    """
    return (Path(sdk_root) / "lib" / target / QHAS_READER_LIB).as_posix()


# ══════════════════════════════════════════════════════════════════════════════
# CLI Builders
# ══════════════════════════════════════════════════════════════════════════════

def build_graph_prepare_qhas_args(
    input_dlc: str,
    output_dlc: str,
    htp_soc: str = "sm8750",
) -> list[str]:
    """Build snpe-dlc-graph-prepare CLI args for QHAS schematic generation.

    Step 1 of QHAS workflow. Produces [MODEL]_schematic.bin alongside output_dlc.

    Args:
        input_dlc: Path to quantized input .dlc
        output_dlc: Path for output cached .dlc
        htp_soc: HTP SoC target (e.g. 'sm8750', 'sm8650', 'sm8550')

    Returns:
        CLI argument list for snpe-dlc-graph-prepare

    Example::
        args = build_graph_prepare_qhas_args(
            "inception_v3_quantized.dlc",
            "inception_v3_quantized_cache.dlc",
        )
        # Produces inception_v3_quantized_schematic.bin
    """
    return [
        "snpe-dlc-graph-prepare",
        "--input_dlc", input_dlc,
        "--output_dlc", output_dlc,
        "--htp_socs", htp_soc,
        "--profiling_level", QHAS_PROFILING_LEVEL,
    ]


def build_net_run_qhas_args(
    container: str,
    input_list: str,
    output_dir: str = "output",
) -> list[str]:
    """Build snpe-net-run CLI args for QHAS artifact collection.

    Step 2 of QHAS workflow. Produces ./profilelogs/qnn-profiling-data.log.

    Args:
        container: Path to cached .dlc from graph-prepare step
        input_list: Path to input_list.txt
        output_dir: Output directory for inference results

    Returns:
        CLI argument list for snpe-net-run

    Example::
        args = build_net_run_qhas_args(
            "inception_v3_quantized_cache.dlc",
            "input_list.txt",
        )
    """
    return [
        "snpe-net-run",
        "--container", container,
        "--input_list", input_list,
        "--output_dir", output_dir,
        "--profiling_level", QHAS_PROFILING_LEVEL,
        "--use_dsp",
    ]


def build_profile_viewer_args(
    config_path: str,
    reader_lib_path: str,
    log_path: str,
    schematic_path: str,
    output_path: str = "chrometrace.json",
) -> list[str]:
    """Build qnn-profile-viewer CLI args for chrometrace generation.

    Step 3 of QHAS workflow. Generates chrometrace.json for chrome://tracing.
    Run on-device where the reader library is available.

    Args:
        config_path: Path to config.json with feature settings
        reader_lib_path: Path to libQnnHtpOptraceProfilingReader.so
        log_path: Path to profilelogs/qnn-profiling-data.log
        schematic_path: Path to [MODEL]_schematic.bin from step 1
        output_path: Path for the output chrometrace JSON file

    Returns:
        CLI argument list for qnn-profile-viewer

    Example::
        args = build_profile_viewer_args(
            "config.json",
            "/sdk/lib/aarch64-android/libQnnHtpOptraceProfilingReader.so",
            "./profilelogs/qnn-profiling-data.log",
            "./inception_v3_quantized_schematic.bin",
        )
    """
    return [
        "qnn-profile-viewer",
        "--config", config_path,
        "--reader", reader_lib_path,
        "--input_log", log_path,
        "--schematic", schematic_path,
        "--output", output_path,
    ]


# ══════════════════════════════════════════════════════════════════════════════
# High-Level Workflow Builder
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QHASWorkflow:
    """Complete QHAS profiling workflow for a model.

    Encapsulates all three steps and their file path relationships.

    Example::
        wf = QHASWorkflow.create(
            model_dlc="inception_v3_quantized.dlc",
            cached_dlc="inception_v3_quantized_cache.dlc",
            input_list="input_list.txt",
            sdk_root="/opt/qairt/2.45.0.260326",
        )
        # Step 1: subprocess.run(wf.graph_prepare_args)
        # Step 2: subprocess.run(wf.net_run_args)
        # wf.config.write(wf.config_path)
        # Step 3: subprocess.run(wf.profile_viewer_args)
    """
    model_dlc: str
    cached_dlc: str
    input_list: str
    sdk_root: str
    htp_soc: str = "sm8750"
    target: str = "aarch64-android"
    working_dir: str = "."
    output_chrometrace: str = "chrometrace.json"
    config: QHASConfig = field(default_factory=QHASConfig)
    config_path: str = "qhas_config.json"

    @classmethod
    def create(
        cls,
        model_dlc: str,
        cached_dlc: str,
        input_list: str,
        sdk_root: str,
        htp_soc: str = "sm8750",
        target: str = "aarch64-android",
        config: QHASConfig | None = None,
    ) -> "QHASWorkflow":
        """Create a QHASWorkflow with sensible defaults."""
        return cls(
            model_dlc=model_dlc,
            cached_dlc=cached_dlc,
            input_list=input_list,
            sdk_root=sdk_root,
            htp_soc=htp_soc,
            target=target,
            config=config or QHASConfig(),
        )

    @property
    def schematic_path(self) -> str:
        """Expected path of the schematic .bin produced by graph-prepare."""
        return get_schematic_path(self.model_dlc)

    @property
    def log_path(self) -> str:
        """Path to the QHAS profiling data log."""
        return get_log_path(self.working_dir)

    @property
    def reader_lib_path(self) -> str:
        """Path to the QHAS reader shared library."""
        return get_reader_lib_path(self.sdk_root, self.target)

    @property
    def graph_prepare_args(self) -> list[str]:
        """Step 1: snpe-dlc-graph-prepare CLI args."""
        return build_graph_prepare_qhas_args(
            self.model_dlc, self.cached_dlc, self.htp_soc
        )

    @property
    def net_run_args(self) -> list[str]:
        """Step 2: snpe-net-run CLI args."""
        return build_net_run_qhas_args(self.cached_dlc, self.input_list)

    @property
    def profile_viewer_args(self) -> list[str]:
        """Step 3: qnn-profile-viewer CLI args."""
        return build_profile_viewer_args(
            config_path=self.config_path,
            reader_lib_path=self.reader_lib_path,
            log_path=self.log_path,
            schematic_path=self.schematic_path,
            output_path=self.output_chrometrace,
        )

    def describe(self) -> str:
        """Return a human-readable description of the full workflow."""
        lines = [
            "QHAS Profiling Workflow",
            "=" * 40,
            "",
            "Step 1 — Graph prepare (generates schematic.bin):",
            "  " + " ".join(self.graph_prepare_args),
            f"  → Produces: {self.schematic_path}",
            "",
            "Step 2 — Runtime artifact collection:",
            "  " + " ".join(self.net_run_args),
            f"  → Produces: {self.log_path}",
            "",
            f"Step 3 — Write config: {self.config_path}",
            f"  Enabled features: {', '.join(self.config.enabled_features)}",
        ]
        if self.config.requires_legacy_ui():
            lines.append("  NOTE: Flow events require legacy Chrome tracing UI")
        lines += [
            "",
            "Step 4 — Chrometrace generation (run on-device):",
            "  " + " ".join(self.profile_viewer_args),
            f"  → Produces: {self.output_chrometrace}",
            "  Open in: chrome://tracing",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

QHAS_PROFILE_NOTES: dict[str, Any] = {
    "description": (
        "QHAS (QNN HTP Analysis Summary) profiling for SNPE HTP runtime. "
        "Three-step workflow: graph-prepare (schematic) → net-run (log) → "
        "profile-viewer (chrometrace)."
    ),
    "profiling_level": QHAS_PROFILING_LEVEL,
    "steps": {
        "1_graph_prepare": {
            "command": "snpe-dlc-graph-prepare",
            "flag": "--profiling_level qhas",
            "output": "[MODEL]_schematic.bin (in current working directory)",
            "required_for": "chrometrace generation — must be passed to qnn-profile-viewer",
        },
        "2_net_run": {
            "command": "snpe-net-run",
            "flags": ["--profiling_level qhas", "--use_dsp"],
            "output": "./profilelogs/qnn-profiling-data.log",
        },
        "3_chrometrace": {
            "command": "qnn-profile-viewer",
            "reader_lib": QHAS_READER_LIB,
            "inputs": ["config.json", "qnn-profiling-data.log", "[MODEL]_schematic.bin"],
            "output": "chrometrace.json",
            "note": "Run on-device where reader library is available",
            "viewer": "chrome://tracing",
        },
    },
    "config_features": {
        "enable_input_output_flow_events": {
            "default": False,
            "description": "I/O dependency flow events in chrometrace",
            "caveat": "Requires legacy Chrome tracing UI",
        },
        "enable_sequencer_flow_events": {
            "default": False,
            "description": "Sequencer ordering dependency flow events",
            "caveat": "Requires legacy Chrome tracing UI",
        },
        "htp_json": {
            "default": True,
            "description": "[NAME]_htp.json — topology and op-by-op HTP graph info",
        },
        "runtrace": {
            "default": True,
            "description": "Runtrace execution and preemption events per core",
        },
        "memory_info": {
            "default": True,
            "description": "Memory bandwidth and allocation graphs per core",
        },
        "traceback": {
            "default": True,
            "description": "Traceback to source framework ops",
        },
        "qhas_schema": {
            "default": False,
            "description": "qhas_schema.json for validating QHAS JSON",
        },
        "qhas_json": {
            "default": False,
            "description": "[model]_qnn_htp_analysis_summary.json",
        },
    },
    "output_files": {
        "schematic": "[MODEL]_schematic.bin — topology schematic from graph-prepare",
        "log": "profilelogs/qnn-profiling-data.log — runtime profiling data",
        "chrometrace": "chrometrace.json — visualization for chrome://tracing",
        "htp_json": "[NAME]_htp.json — op-by-op HTP graph info (when htp_json=true)",
        "qhas_json": "[model]_qnn_htp_analysis_summary.json (when qhas_json=true)",
    },
    "see_also": [
        "HTP Optrace Profiling in QNN → Backend → HTP",
        "QNN HTP Optrace Profiling section",
    ],
}
