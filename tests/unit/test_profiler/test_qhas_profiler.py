"""Tests for QHAS (QNN HTP Analysis Summary) profiling workflow."""

from __future__ import annotations

import json

import pytest

from quad.profiler.qhas import (
    QHAS_LOG_FILENAME,
    QHAS_PROFILING_LEVEL,
    QHAS_PROFILE_NOTES,
    QHAS_READER_LIB,
    SCHEMATIC_SUFFIX,
    QHASConfig,
    QHASWorkflow,
    build_graph_prepare_qhas_args,
    build_net_run_qhas_args,
    build_profile_viewer_args,
    get_log_path,
    get_profilelogs_dir,
    get_reader_lib_path,
    get_schematic_path,
)


# ══════════════════════════════════════════════════════════════════════════════
# QHASConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestQHASConfig:
    def test_defaults_match_documentation(self) -> None:
        cfg = QHASConfig()
        # Default OFF
        assert cfg.enable_input_output_flow_events is False
        assert cfg.enable_sequencer_flow_events is False
        assert cfg.qhas_schema is False
        assert cfg.qhas_json is False
        # Default ON
        assert cfg.htp_json is True
        assert cfg.runtrace is True
        assert cfg.memory_info is True
        assert cfg.traceback is True

    def test_to_dict_structure(self) -> None:
        cfg = QHASConfig()
        d = cfg.to_dict()
        assert "features" in d
        features = d["features"]
        assert "enable_input_output_flow_events" in features
        assert "htp_json" in features
        assert "runtrace" in features
        assert "memory_info" in features
        assert "traceback" in features
        assert "qhas_schema" in features
        assert "qhas_json" in features

    def test_to_json_valid(self) -> None:
        cfg = QHASConfig()
        json_str = cfg.to_json()
        parsed = json.loads(json_str)
        assert "features" in parsed

    def test_to_json_sample_matches_documentation(self) -> None:
        """The sample config.json from the documentation has all features true."""
        cfg = QHASConfig.full()
        d = cfg.to_dict()["features"]
        assert d["enable_input_output_flow_events"] is True
        assert d["enable_sequencer_flow_events"] is True
        assert d["htp_json"] is True
        assert d["runtrace"] is True
        assert d["memory_info"] is True
        assert d["traceback"] is True
        assert d["qhas_schema"] is True
        assert d["qhas_json"] is True

    def test_full_factory(self) -> None:
        cfg = QHASConfig.full()
        assert all([
            cfg.enable_input_output_flow_events,
            cfg.enable_sequencer_flow_events,
            cfg.htp_json,
            cfg.runtrace,
            cfg.memory_info,
            cfg.traceback,
            cfg.qhas_schema,
            cfg.qhas_json,
        ])

    def test_minimal_factory(self) -> None:
        cfg = QHASConfig.minimal()
        assert cfg.enable_input_output_flow_events is False
        assert cfg.qhas_json is False
        assert cfg.htp_json is True  # default-on preserved

    def test_from_dict_roundtrip(self) -> None:
        original = QHASConfig.full()
        restored = QHASConfig.from_dict(original.to_dict())
        assert restored.enable_input_output_flow_events == original.enable_input_output_flow_events
        assert restored.htp_json == original.htp_json
        assert restored.qhas_json == original.qhas_json

    def test_from_json_roundtrip(self) -> None:
        original = QHASConfig(qhas_json=True, runtrace=False)
        restored = QHASConfig.from_json(original.to_json())
        assert restored.qhas_json is True
        assert restored.runtrace is False

    def test_from_file(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]
        cfg = QHASConfig.full()
        path = str(tmp_path / "config.json")
        cfg.write(path)
        loaded = QHASConfig.from_file(path)
        assert loaded.qhas_json is True
        assert loaded.enable_input_output_flow_events is True

    def test_requires_legacy_ui_false_by_default(self) -> None:
        assert QHASConfig().requires_legacy_ui() is False

    def test_requires_legacy_ui_input_output_flow(self) -> None:
        cfg = QHASConfig(enable_input_output_flow_events=True)
        assert cfg.requires_legacy_ui() is True

    def test_requires_legacy_ui_sequencer_flow(self) -> None:
        cfg = QHASConfig(enable_sequencer_flow_events=True)
        assert cfg.requires_legacy_ui() is True

    def test_enabled_features_default(self) -> None:
        cfg = QHASConfig()
        features = cfg.enabled_features
        assert "htp_json" in features
        assert "runtrace" in features
        assert "memory_info" in features
        assert "traceback" in features
        assert "qhas_json" not in features
        assert "enable_input_output_flow_events" not in features

    def test_enabled_features_full(self) -> None:
        cfg = QHASConfig.full()
        features = cfg.enabled_features
        assert len(features) == 8

    def test_from_dict_flat_features(self) -> None:
        """from_dict should also accept a flat dict (features without wrapper)."""
        flat = {"htp_json": False, "runtrace": True}
        cfg = QHASConfig.from_dict(flat)
        assert cfg.htp_json is False
        assert cfg.runtrace is True

    def test_write_creates_file(self, tmp_path: "Path") -> None:  # type: ignore[name-defined]
        path = str(tmp_path / "out_config.json")
        QHASConfig().write(path)
        content = json.loads(open(path).read())
        assert "features" in content


# ══════════════════════════════════════════════════════════════════════════════
# File Path Helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestPathHelpers:
    def test_get_schematic_path_basic(self) -> None:
        result = get_schematic_path("inception_v3_quantized.dlc")
        assert result == "inception_v3_quantized_schematic.bin"

    def test_get_schematic_path_preserves_stem(self) -> None:
        result = get_schematic_path("/some/path/my_model.dlc")
        assert result == "my_model_schematic.bin"

    def test_get_schematic_path_suffix(self) -> None:
        result = get_schematic_path("model.dlc")
        assert result.endswith(SCHEMATIC_SUFFIX)

    def test_get_profilelogs_dir_default(self) -> None:
        result = get_profilelogs_dir()
        assert result.endswith("profilelogs")

    def test_get_profilelogs_dir_custom(self) -> None:
        result = get_profilelogs_dir("/tmp/myrun")
        assert "profilelogs" in result
        assert "/tmp/myrun" in result

    def test_get_log_path_default(self) -> None:
        result = get_log_path()
        assert QHAS_LOG_FILENAME in result
        assert "profilelogs" in result

    def test_get_log_path_custom_dir(self) -> None:
        result = get_log_path("/tmp/run")
        assert result.endswith(QHAS_LOG_FILENAME)

    def test_get_reader_lib_path(self) -> None:
        result = get_reader_lib_path("/opt/qairt/2.45.0", "aarch64-android")
        assert QHAS_READER_LIB in result
        assert "aarch64-android" in result
        assert "/opt/qairt/2.45.0" in result

    def test_get_reader_lib_path_default_target(self) -> None:
        result = get_reader_lib_path("/sdk")
        assert "aarch64-android" in result


# ══════════════════════════════════════════════════════════════════════════════
# CLI Builders
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildGraphPrepareQhasArgs:
    def test_command_name(self) -> None:
        args = build_graph_prepare_qhas_args("m.dlc", "m_cache.dlc")
        assert args[0] == "snpe-dlc-graph-prepare"

    def test_profiling_level_qhas(self) -> None:
        args = build_graph_prepare_qhas_args("m.dlc", "m_cache.dlc")
        idx = args.index("--profiling_level")
        assert args[idx + 1] == QHAS_PROFILING_LEVEL

    def test_input_dlc(self) -> None:
        args = build_graph_prepare_qhas_args("model.dlc", "out.dlc")
        idx = args.index("--input_dlc")
        assert args[idx + 1] == "model.dlc"

    def test_output_dlc(self) -> None:
        args = build_graph_prepare_qhas_args("model.dlc", "out.dlc")
        idx = args.index("--output_dlc")
        assert args[idx + 1] == "out.dlc"

    def test_htp_socs_default(self) -> None:
        args = build_graph_prepare_qhas_args("m.dlc", "m_cache.dlc")
        idx = args.index("--htp_socs")
        assert args[idx + 1] == "sm8750"

    def test_htp_socs_custom(self) -> None:
        args = build_graph_prepare_qhas_args("m.dlc", "m_cache.dlc", htp_soc="sm8650")
        idx = args.index("--htp_socs")
        assert args[idx + 1] == "sm8650"

    def test_documentation_example(self) -> None:
        """Reproduce the exact example from documentation."""
        args = build_graph_prepare_qhas_args(
            "inception_v3_quantized.dlc",
            "inception_v3_quantized_cache.dlc",
            htp_soc="sm8750",
        )
        assert "--profiling_level" in args
        assert "qhas" in args
        assert "inception_v3_quantized.dlc" in args
        assert "inception_v3_quantized_cache.dlc" in args
        assert "sm8750" in args


class TestBuildNetRunQhasArgs:
    def test_command_name(self) -> None:
        args = build_net_run_qhas_args("model.dlc", "inputs.txt")
        assert args[0] == "snpe-net-run"

    def test_profiling_level_qhas(self) -> None:
        args = build_net_run_qhas_args("model.dlc", "inputs.txt")
        idx = args.index("--profiling_level")
        assert args[idx + 1] == QHAS_PROFILING_LEVEL

    def test_use_dsp_flag(self) -> None:
        args = build_net_run_qhas_args("model.dlc", "inputs.txt")
        assert "--use_dsp" in args

    def test_container_arg(self) -> None:
        args = build_net_run_qhas_args("my_model.dlc", "inputs.txt")
        idx = args.index("--container")
        assert args[idx + 1] == "my_model.dlc"

    def test_input_list_arg(self) -> None:
        args = build_net_run_qhas_args("model.dlc", "my_inputs.txt")
        idx = args.index("--input_list")
        assert args[idx + 1] == "my_inputs.txt"

    def test_documentation_example(self) -> None:
        """Reproduce the exact example from documentation."""
        args = build_net_run_qhas_args(
            "inception_v3_quantized_cache.dlc",
            "input_list.txt",
        )
        assert "inception_v3_quantized_cache.dlc" in args
        assert "input_list.txt" in args
        assert "--use_dsp" in args
        assert "qhas" in args


class TestBuildProfileViewerArgs:
    def test_command_name(self) -> None:
        args = build_profile_viewer_args("cfg.json", "reader.so", "log.log", "schematic.bin")
        assert args[0] == "qnn-profile-viewer"

    def test_config_arg(self) -> None:
        args = build_profile_viewer_args("my_config.json", "r.so", "l.log", "s.bin")
        idx = args.index("--config")
        assert args[idx + 1] == "my_config.json"

    def test_reader_arg(self) -> None:
        args = build_profile_viewer_args("c.json", "libQnn.so", "l.log", "s.bin")
        idx = args.index("--reader")
        assert args[idx + 1] == "libQnn.so"

    def test_input_log_arg(self) -> None:
        args = build_profile_viewer_args("c.json", "r.so", "my_log.log", "s.bin")
        idx = args.index("--input_log")
        assert args[idx + 1] == "my_log.log"

    def test_schematic_arg(self) -> None:
        args = build_profile_viewer_args("c.json", "r.so", "l.log", "my_schematic.bin")
        idx = args.index("--schematic")
        assert args[idx + 1] == "my_schematic.bin"

    def test_output_default(self) -> None:
        args = build_profile_viewer_args("c.json", "r.so", "l.log", "s.bin")
        idx = args.index("--output")
        assert args[idx + 1] == "chrometrace.json"

    def test_output_custom(self) -> None:
        args = build_profile_viewer_args("c.json", "r.so", "l.log", "s.bin",
                                         output_path="my_trace.json")
        idx = args.index("--output")
        assert args[idx + 1] == "my_trace.json"

    def test_documentation_example(self) -> None:
        """Reproduce the exact documentation example structure."""
        sdk = "/opt/qairt/2.45.0"
        args = build_profile_viewer_args(
            config_path="config.json",
            reader_lib_path=f"{sdk}/lib/aarch64-android/{QHAS_READER_LIB}",
            log_path="./profilelogs/qnn-profiling-data.log",
            schematic_path="./inception_v3_quantized_schematic.bin",
            output_path="./chrometrace.json",
        )
        assert "qnn-profile-viewer" in args
        assert QHAS_READER_LIB in " ".join(args)
        assert "qnn-profiling-data.log" in " ".join(args)
        assert "schematic.bin" in " ".join(args)


# ══════════════════════════════════════════════════════════════════════════════
# QHASWorkflow
# ══════════════════════════════════════════════════════════════════════════════

class TestQHASWorkflow:
    def _make_workflow(self) -> QHASWorkflow:
        return QHASWorkflow.create(
            model_dlc="inception_v3_quantized.dlc",
            cached_dlc="inception_v3_quantized_cache.dlc",
            input_list="input_list.txt",
            sdk_root="/opt/qairt/2.45.0",
        )

    def test_schematic_path_derived_from_model_dlc(self) -> None:
        wf = self._make_workflow()
        assert wf.schematic_path == "inception_v3_quantized_schematic.bin"

    def test_log_path_contains_profilelogs(self) -> None:
        wf = self._make_workflow()
        assert "profilelogs" in wf.log_path
        assert QHAS_LOG_FILENAME in wf.log_path

    def test_reader_lib_contains_sdk_root(self) -> None:
        wf = self._make_workflow()
        assert "/opt/qairt/2.45.0" in wf.reader_lib_path
        assert QHAS_READER_LIB in wf.reader_lib_path

    def test_graph_prepare_args_step_1(self) -> None:
        wf = self._make_workflow()
        args = wf.graph_prepare_args
        assert args[0] == "snpe-dlc-graph-prepare"
        assert "qhas" in args

    def test_net_run_args_step_2(self) -> None:
        wf = self._make_workflow()
        args = wf.net_run_args
        assert args[0] == "snpe-net-run"
        assert "--use_dsp" in args
        assert "qhas" in args

    def test_profile_viewer_args_step_3(self) -> None:
        wf = self._make_workflow()
        args = wf.profile_viewer_args
        assert args[0] == "qnn-profile-viewer"
        assert wf.schematic_path in args
        assert wf.log_path in args
        assert wf.reader_lib_path in args

    def test_describe_contains_all_steps(self) -> None:
        wf = self._make_workflow()
        desc = wf.describe()
        assert "Step 1" in desc
        assert "Step 2" in desc
        assert "Step 3" in desc or "Step 4" in desc
        assert "schematic" in desc.lower()
        assert "profilelogs" in desc
        assert "chrometrace" in desc.lower()

    def test_describe_legacy_ui_warning_when_flow_events(self) -> None:
        wf = QHASWorkflow.create(
            model_dlc="m.dlc",
            cached_dlc="m_cache.dlc",
            input_list="inputs.txt",
            sdk_root="/sdk",
            config=QHASConfig(enable_input_output_flow_events=True),
        )
        desc = wf.describe()
        assert "legacy" in desc.lower() or "Legacy" in desc

    def test_describe_no_legacy_warning_by_default(self) -> None:
        wf = self._make_workflow()
        desc = wf.describe()
        assert "legacy" not in desc.lower()

    def test_htp_soc_propagates_to_graph_prepare(self) -> None:
        wf = QHASWorkflow.create(
            "m.dlc", "m_cache.dlc", "inputs.txt", "/sdk", htp_soc="sm8650"
        )
        assert "sm8650" in wf.graph_prepare_args

    def test_custom_target_in_reader_lib(self) -> None:
        wf = QHASWorkflow.create(
            "m.dlc", "m_cache.dlc", "inputs.txt", "/sdk",
            target="aarch64-ubuntu-gcc11.4",
        )
        assert "aarch64-ubuntu-gcc11.4" in wf.reader_lib_path


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

class TestQHASProfileNotes:
    def test_profiling_level_value(self) -> None:
        assert QHAS_PROFILING_LEVEL == "qhas"

    def test_notes_steps_present(self) -> None:
        steps = QHAS_PROFILE_NOTES["steps"]
        assert "1_graph_prepare" in steps
        assert "2_net_run" in steps
        assert "3_chrometrace" in steps

    def test_graph_prepare_step_output(self) -> None:
        step = QHAS_PROFILE_NOTES["steps"]["1_graph_prepare"]
        assert "schematic.bin" in step["output"]

    def test_net_run_step_flags(self) -> None:
        step = QHAS_PROFILE_NOTES["steps"]["2_net_run"]
        flags = step["flags"]
        assert any("qhas" in f for f in flags)
        assert any("use_dsp" in f for f in flags)

    def test_chrometrace_reader_lib(self) -> None:
        step = QHAS_PROFILE_NOTES["steps"]["3_chrometrace"]
        assert QHAS_READER_LIB in step["reader_lib"]

    def test_config_feature_defaults(self) -> None:
        features = QHAS_PROFILE_NOTES["config_features"]
        # Default-ON
        for name in ("htp_json", "runtrace", "memory_info", "traceback"):
            assert features[name]["default"] is True
        # Default-OFF
        for name in ("enable_input_output_flow_events", "enable_sequencer_flow_events",
                     "qhas_schema", "qhas_json"):
            assert features[name]["default"] is False

    def test_flow_event_features_have_caveat(self) -> None:
        features = QHAS_PROFILE_NOTES["config_features"]
        for key in ("enable_input_output_flow_events", "enable_sequencer_flow_events"):
            assert "caveat" in features[key]
            assert "legacy" in features[key]["caveat"].lower()

    def test_output_files_documented(self) -> None:
        outputs = QHAS_PROFILE_NOTES["output_files"]
        assert "schematic" in outputs
        assert "log" in outputs
        assert "chrometrace" in outputs
        assert "htp_json" in outputs
        assert "qhas_json" in outputs

    def test_log_filename_constant(self) -> None:
        assert QHAS_LOG_FILENAME == "qnn-profiling-data.log"

    def test_reader_lib_constant(self) -> None:
        assert QHAS_READER_LIB == "libQnnHtpOptraceProfilingReader.so"

    def test_schematic_suffix_constant(self) -> None:
        assert SCHEMATIC_SUFFIX == "_schematic.bin"
