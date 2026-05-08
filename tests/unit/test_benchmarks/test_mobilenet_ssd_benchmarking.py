"""Tests for MobilenetSSD benchmarking module."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quad.benchmarks.config import (
    BenchmarkModelConfig,
    BenchmarkResults,
    BenchmarkTimingRow,
    SNPEBenchmarkConfig,
    VALID_BUFFER_TYPES,
    VALID_RUNTIMES,
)
from quad.benchmarks.mobilenet_ssd import (
    MOBILENET_SSD_BENCHMARK_NOTES,
    MOBILENET_SSD_DEFAULTS,
    MOBILENET_SSD_GPU_DSP_SPEEDUP_RANGE,
    MOBILENET_SSD_OUTPUT_LAYERS,
    TIMING_NOTE,
    build_mobilenet_ssd_benchmark_config,
    build_mobilenet_ssd_input_list,
    build_snpe_bench_cmd,
    find_snpe_bench,
    get_latest_results_dir,
    parse_benchmark_csv,
    parse_benchmark_json,
    parse_input_list,
)


# ══════════════════════════════════════════════════════════════════════════════
# SNPEBenchmarkConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestSNPEBenchmarkConfig:
    def _model(self) -> BenchmarkModelConfig:
        return BenchmarkModelConfig(
            name="mobilenet_ssd",
            dlc="/tmp/mobilenet_ssd.dlc",
            input_list="/tmp/imagelist.txt",
            data=["/tmp/images"],
        )

    def test_to_dict_contains_all_keys(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="mobilenet_ssd",
            host_root_path="mobilenet_ssd",
            host_results_dir="mobilenet_ssd/results",
            device_path="/data/local/tmp/snpebm",
            devices=["454d40f3"],
            model=self._model(),
        )
        d = cfg.to_dict()
        assert "Name" in d
        assert "HostRootPath" in d
        assert "HostResultsDir" in d
        assert "DevicePath" in d
        assert "Devices" in d
        assert "Runs" in d
        assert "Model" in d
        assert "Runtimes" in d
        assert "Measurements" in d
        assert "ProfilingLevel" in d

    def test_cpu_fallback_only_when_true(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="n", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=[], model=self._model(),
            cpu_fallback=False,
        )
        assert "CpuFallback" not in cfg.to_dict()

        cfg.cpu_fallback = True
        assert cfg.to_dict()["CpuFallback"] is True

    def test_buffer_types_only_when_set(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="n", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=[], model=self._model(),
            buffer_types=[],
        )
        assert "BufferTypes" not in cfg.to_dict()

        cfg.buffer_types = ["ub_float", "ub_tf8"]
        assert cfg.to_dict()["BufferTypes"] == ["ub_float", "ub_tf8"]

    def test_to_json_valid(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="test", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=["abc123"], model=self._model(),
        )
        parsed = json.loads(cfg.to_json())
        assert parsed["Name"] == "test"
        assert "Model" in parsed

    def test_from_dict_roundtrip(self) -> None:
        original = SNPEBenchmarkConfig(
            name="mobilenet_ssd",
            host_root_path="mobilenet_ssd",
            host_results_dir="mobilenet_ssd/results",
            device_path="/data/local/tmp/snpebm",
            devices=["454d40f3"],
            model=self._model(),
            runs=2,
            runtimes=["GPU"],
            measurements=["timing"],
            profiling_level="detailed",
            buffer_types=["ub_float", "ub_tf8"],
            cpu_fallback=True,
        )
        restored = SNPEBenchmarkConfig.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.cpu_fallback is True
        assert restored.buffer_types == ["ub_float", "ub_tf8"]
        assert restored.profiling_level == "detailed"
        assert restored.model.dlc == "/tmp/mobilenet_ssd.dlc"

    def test_from_json_roundtrip(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="test", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=["x"],
            model=self._model(),
            cpu_fallback=True,
        )
        restored = SNPEBenchmarkConfig.from_json(cfg.to_json())
        assert restored.cpu_fallback is True

    def test_write_creates_file(self, tmp_path: Path) -> None:
        cfg = SNPEBenchmarkConfig(
            name="test", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=[], model=self._model(),
        )
        path = str(tmp_path / "bench.json")
        cfg.write(path)
        loaded = json.loads(open(path).read())
        assert loaded["Name"] == "test"

    def test_profiling_level_case_preserved(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="n", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=[], model=self._model(),
            profiling_level="detailed",
        )
        assert cfg.to_dict()["ProfilingLevel"] == "detailed"

    def test_validation_invalid_runtime(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="n", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=[], model=self._model(),
            runtimes=["INVALID_RT"],
        )
        errors = cfg.validate()
        assert any("INVALID_RT" in e for e in errors)

    def test_validation_invalid_buffer_type(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="n", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=[], model=self._model(),
            buffer_types=["bad_type"],
        )
        errors = cfg.validate()
        assert any("bad_type" in e for e in errors)

    def test_validation_passes_for_valid_config(self) -> None:
        cfg = SNPEBenchmarkConfig(
            name="n", host_root_path="p", host_results_dir="r",
            device_path="/d", devices=[], model=self._model(),
            runtimes=["GPU", "CPU"],
            buffer_types=["ub_float", "ub_tf8"],
            profiling_level="detailed",
            measurements=["timing"],
        )
        assert cfg.validate() == []

    def test_documentation_example_structure(self) -> None:
        """Reproduce the exact JSON from the documentation."""
        cfg = SNPEBenchmarkConfig(
            name="mobilenet_ssd",
            host_root_path="mobilenet_ssd",
            host_results_dir="mobilenet_ssd/results",
            device_path="/data/local/tmp/snpebm",
            devices=["454d40f3"],
            model=BenchmarkModelConfig(
                name="mobilenet_ssd",
                dlc="/tmp/mobilenet_ssd.dlc",
                input_list="/tmp/imagelist.txt",
                data=["/tmp/images"],
            ),
            runs=2,
            runtimes=["GPU"],
            measurements=["timing"],
            profiling_level="detailed",
            buffer_types=["ub_float", "ub_tf8"],
            cpu_fallback=True,
        )
        d = cfg.to_dict()
        assert d["Devices"] == ["454d40f3"]
        assert d["CpuFallback"] is True
        assert "ub_tf8" in d["BufferTypes"]
        assert d["Runs"] == 2
        assert d["ProfilingLevel"] == "detailed"


# ══════════════════════════════════════════════════════════════════════════════
# build_mobilenet_ssd_benchmark_config
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildMobilenetSsdBenchmarkConfig:
    def test_cpu_fallback_always_true(self) -> None:
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/m.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=["abc123"],
        )
        assert cfg.cpu_fallback is True

    def test_buffer_types_include_ub_tf8(self) -> None:
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/m.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=["abc123"],
        )
        assert "ub_tf8" in cfg.buffer_types
        assert "ub_float" in cfg.buffer_types

    def test_default_runtime_is_gpu(self) -> None:
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/m.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=[],
        )
        assert "GPU" in cfg.runtimes

    def test_default_profiling_level_detailed(self) -> None:
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/m.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=[],
        )
        assert cfg.profiling_level == "detailed"

    def test_custom_runtimes(self) -> None:
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/m.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=[],
            runtimes=["CPU", "GPU", "DSP"],
        )
        assert "DSP" in cfg.runtimes

    def test_model_dlc_path_set(self) -> None:
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/mobilenet_ssd.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=[],
        )
        assert cfg.model.dlc == "/tmp/mobilenet_ssd.dlc"

    def test_validation_passes(self) -> None:
        cfg = build_mobilenet_ssd_benchmark_config(
            dlc_path="/tmp/m.dlc",
            input_list_path="/tmp/imagelist.txt",
            image_data_dirs=["/tmp/images"],
            device_serials=["abc"],
        )
        errors = cfg.validate()
        assert errors == [], f"Unexpected errors: {errors}"


# ══════════════════════════════════════════════════════════════════════════════
# Input List Builder / Parser
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildMobilenetSsdInputList:
    def test_first_line_has_output_layers(self, tmp_path: Path) -> None:
        out = str(tmp_path / "imagelist.txt")
        build_mobilenet_ssd_input_list(
            image_raw_paths=["/tmp/0#.rawtensor", "/tmp/1#.rawtensor"],
            output_path=out,
        )
        lines = Path(out).read_text().splitlines()
        assert lines[0].startswith("#")
        assert "Postprocessor/BatchMultiClassNonMaxSuppression" in lines[0]
        assert "add_6" in lines[0]

    def test_image_paths_follow_header(self, tmp_path: Path) -> None:
        out = str(tmp_path / "imagelist.txt")
        build_mobilenet_ssd_input_list(
            image_raw_paths=["img0.rawtensor", "img1.rawtensor"],
            output_path=out,
        )
        lines = [l for l in Path(out).read_text().splitlines() if l]
        assert lines[1] == "img0.rawtensor"
        assert lines[2] == "img1.rawtensor"

    def test_custom_output_layers(self, tmp_path: Path) -> None:
        out = str(tmp_path / "imagelist.txt")
        build_mobilenet_ssd_input_list(
            image_raw_paths=["img.raw"],
            output_path=out,
            output_layers=["custom_output_1", "custom_output_2"],
        )
        first_line = Path(out).read_text().splitlines()[0]
        assert "custom_output_1" in first_line
        assert "custom_output_2" in first_line

    def test_returns_output_path(self, tmp_path: Path) -> None:
        out = str(tmp_path / "imagelist.txt")
        result = build_mobilenet_ssd_input_list(["img.raw"], out)
        assert result == out

    def test_documentation_format(self, tmp_path: Path) -> None:
        """Reproduce exact format from documentation."""
        out = str(tmp_path / "imagelist.txt")
        build_mobilenet_ssd_input_list(
            image_raw_paths=["tmp/0#.rawtensor", "tmp/1#.rawtensor"],
            output_path=out,
        )
        content = Path(out).read_text()
        assert content.startswith("#Postprocessor/BatchMultiClassNonMaxSuppression add_6")
        assert "tmp/0#.rawtensor" in content


class TestParseInputList:
    def test_parses_output_layers_from_header(self, tmp_path: Path) -> None:
        path = tmp_path / "imagelist.txt"
        path.write_text(
            "#Postprocessor/BatchMultiClassNonMaxSuppression add_6\n"
            "/tmp/0.rawtensor\n"
            "/tmp/1.rawtensor\n"
        )
        layers, images = parse_input_list(str(path))
        assert "Postprocessor/BatchMultiClassNonMaxSuppression" in layers
        assert "add_6" in layers
        assert "/tmp/0.rawtensor" in images
        assert "/tmp/1.rawtensor" in images

    def test_no_header_returns_empty_layers(self, tmp_path: Path) -> None:
        path = tmp_path / "imagelist.txt"
        path.write_text("/tmp/0.rawtensor\n/tmp/1.rawtensor\n")
        layers, images = parse_input_list(str(path))
        assert layers == []
        assert len(images) == 2

    def test_roundtrip_with_builder(self, tmp_path: Path) -> None:
        out = str(tmp_path / "imagelist.txt")
        imgs = ["/tmp/0.rawtensor", "/tmp/1.rawtensor"]
        build_mobilenet_ssd_input_list(imgs, out)
        layers, parsed_imgs = parse_input_list(out)
        assert layers == MOBILENET_SSD_OUTPUT_LAYERS
        assert parsed_imgs == imgs

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "imagelist.txt"
        path.write_text("#layer1\n\n/tmp/img.raw\n\n")
        layers, images = parse_input_list(str(path))
        assert images == ["/tmp/img.raw"]


# ══════════════════════════════════════════════════════════════════════════════
# snpe_bench.py CLI builder
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildSnpeBenchCmd:
    def test_raises_when_not_found(self) -> None:
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(FileNotFoundError, match="snpe_bench.py"):
                    build_snpe_bench_cmd("/tmp/cfg.json")

    def test_basic_command_structure(self) -> None:
        with patch("shutil.which", return_value="/sdk/benchmarks/snpe_bench.py"):
            cmd = build_snpe_bench_cmd("/tmp/mobilenetssd.json")
        assert "python3" in cmd
        assert "/tmp/mobilenetssd.json" in cmd
        assert "-c" in cmd
        assert "-a" in cmd

    def test_generate_json_flag(self) -> None:
        with patch("shutil.which", return_value="/sdk/snpe_bench.py"):
            cmd = build_snpe_bench_cmd("/tmp/cfg.json", generate_json=True)
        assert "--generate_json" in cmd

    def test_no_generate_json_by_default(self) -> None:
        with patch("shutil.which", return_value="/sdk/snpe_bench.py"):
            cmd = build_snpe_bench_cmd("/tmp/cfg.json")
        assert "--generate_json" not in cmd

    def test_finds_bench_in_sdk_root(self, tmp_path: Path) -> None:
        bench_dir = tmp_path / "benchmarks" / "SNPE"
        bench_dir.mkdir(parents=True)
        bench_file = bench_dir / "snpe_bench.py"
        bench_file.touch()
        with patch("shutil.which", return_value=None):
            with patch.dict("os.environ", {"SNPE_ROOT": str(tmp_path)}):
                cmd = build_snpe_bench_cmd("/tmp/cfg.json", sdk_root=str(tmp_path))
        assert str(bench_file) in cmd


# ══════════════════════════════════════════════════════════════════════════════
# Results parsing
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkTimingRow:
    def test_ms_conversion(self) -> None:
        row = BenchmarkTimingRow(
            runtime="GPU",
            total_inference_us=5000.0,
            forward_propagate_us=4500.0,
        )
        assert row.total_inference_ms == pytest.approx(5.0)
        assert row.forward_propagate_ms == pytest.approx(4.5)

    def test_speedup_vs(self) -> None:
        cpu = BenchmarkTimingRow("CPU", total_inference_us=100000.0, forward_propagate_us=90000.0)
        gpu = BenchmarkTimingRow("GPU", total_inference_us=5000.0, forward_propagate_us=4000.0)
        speedup = gpu.speedup_vs(cpu)  # GPU is faster → speedup > 1
        assert speedup == pytest.approx(20.0)

    def test_speedup_vs_zero_total(self) -> None:
        a = BenchmarkTimingRow("A", 0.0, 0.0)
        b = BenchmarkTimingRow("B", 1000.0, 900.0)
        assert a.speedup_vs(b) == 0.0


class TestBenchmarkResults:
    def _results(self) -> BenchmarkResults:
        return BenchmarkResults(
            model_name="mobilenet_ssd",
            run_dir="/results/20250101_120000",
            rows=[
                BenchmarkTimingRow("CPU", 100000.0, 90000.0),
                BenchmarkTimingRow("GPU", 5000.0, 4500.0),
                BenchmarkTimingRow("DSP", 4000.0, 3500.0),
            ],
        )

    def test_cpu_row_found(self) -> None:
        results = self._results()
        assert results.cpu_row is not None
        assert results.cpu_row.runtime == "CPU"

    def test_gpu_row_found(self) -> None:
        assert self._results().gpu_row is not None

    def test_dsp_row_found(self) -> None:
        assert self._results().dsp_row is not None

    def test_gpu_vs_cpu_speedup(self) -> None:
        speedup = self._results().gpu_vs_cpu_speedup()
        assert speedup is not None
        assert speedup == pytest.approx(20.0)

    def test_dsp_vs_cpu_speedup(self) -> None:
        speedup = self._results().dsp_vs_cpu_speedup()
        assert speedup is not None
        assert speedup == pytest.approx(25.0)

    def test_speedup_in_documented_range(self) -> None:
        """Validate the documented 17-39x speedup range is plausible."""
        lo, hi = MOBILENET_SSD_GPU_DSP_SPEEDUP_RANGE
        results = self._results()
        gpu_speedup = results.gpu_vs_cpu_speedup()
        dsp_speedup = results.dsp_vs_cpu_speedup()
        # Our mock results (20x/25x) fall in the documented range
        assert lo <= gpu_speedup <= hi * 2  # Allow wide range for mock data
        assert lo <= dsp_speedup <= hi * 2


class TestParseBenchmarkCsv:
    def test_parses_csv_with_timing_rows(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "Runtime,Total Inference Time,Forward Propagate\n"
            "CPU,100000,90000\n"
            "GPU,5000,4500\n"
        )
        results = parse_benchmark_csv(str(csv_path))
        assert len(results.rows) == 2
        assert results.cpu_row is not None
        assert results.cpu_row.total_inference_us == pytest.approx(100000.0)
        assert results.gpu_row is not None
        assert results.gpu_row.total_inference_us == pytest.approx(5000.0)

    def test_gpu_speedup_from_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "Runtime,Total Inference Time,Forward Propagate\n"
            "CPU,391000,350000\n"
            "GPU,21000,19000\n"
        )
        results = parse_benchmark_csv(str(csv_path))
        speedup = results.gpu_vs_cpu_speedup()
        assert speedup is not None
        assert speedup == pytest.approx(391000 / 21000, rel=0.01)

    def test_model_name_from_filename(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "mobilenet_ssd.csv"
        csv_path.write_text("Runtime,Total Inference Time,Forward Propagate\n")
        results = parse_benchmark_csv(str(csv_path))
        assert results.model_name == "mobilenet_ssd"


class TestParseBenchmarkJson:
    def test_parses_json_results(self, tmp_path: Path) -> None:
        json_path = tmp_path / "results.json"
        data = {
            "results": [
                {"Runtime": "CPU", "Total Inference Time": 100000, "Forward Propagate": 90000},
                {"Runtime": "GPU", "Total Inference Time": 5000, "Forward Propagate": 4500},
            ]
        }
        json_path.write_text(json.dumps(data))
        results = parse_benchmark_json(str(json_path))
        assert len(results.rows) == 2
        assert results.gpu_row is not None


# ══════════════════════════════════════════════════════════════════════════════
# get_latest_results_dir
# ══════════════════════════════════════════════════════════════════════════════

class TestGetLatestResultsDir:
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="symlink_to() requires SeCreateSymbolicLinkPrivilege on Windows; "
        "default user accounts don't have it. The fallback path "
        "(test_falls_back_to_most_recent_dir) covers the same logical branch.",
    )
    def test_returns_latest_results_symlink_if_exists(self, tmp_path: Path) -> None:
        actual = tmp_path / "20250101_120000"
        actual.mkdir()
        latest = tmp_path / "latest_results"
        latest.symlink_to(actual)
        result = get_latest_results_dir(str(tmp_path))
        assert result == str(actual.resolve())

    def test_falls_back_to_most_recent_dir(self, tmp_path: Path) -> None:
        import time
        d1 = tmp_path / "20250101_110000"
        d1.mkdir()
        time.sleep(0.01)
        d2 = tmp_path / "20250101_120000"
        d2.mkdir()
        result = get_latest_results_dir(str(tmp_path))
        assert result == str(d2)

    def test_returns_none_when_empty(self, tmp_path: Path) -> None:
        result = get_latest_results_dir(str(tmp_path))
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

class TestMobilenetSsdBenchmarkNotes:
    def test_output_layers_match_documentation(self) -> None:
        assert "Postprocessor/BatchMultiClassNonMaxSuppression" in MOBILENET_SSD_OUTPUT_LAYERS
        assert "add_6" in MOBILENET_SSD_OUTPUT_LAYERS

    def test_notes_cpu_fallback_required(self) -> None:
        cpu_fb = MOBILENET_SSD_BENCHMARK_NOTES["required_config_fields"]["CpuFallback"]
        assert cpu_fb["value"] is True
        assert "DetectionOutput" in cpu_fb["reason"] or "CPU" in cpu_fb["reason"]

    def test_notes_profiling_level_recommendation(self) -> None:
        pl = MOBILENET_SSD_BENCHMARK_NOTES["required_config_fields"]["ProfilingLevel"]
        assert pl["recommended"] == "detailed"
        assert "detailed" in pl["note"].lower()

    def test_notes_performance_speedup(self) -> None:
        perf = MOBILENET_SSD_BENCHMARK_NOTES["performance"]
        assert "17" in perf["gpu_dsp_vs_cpu_speedup"]
        assert "39" in perf["gpu_dsp_vs_cpu_speedup"]

    def test_notes_results_location(self) -> None:
        results = MOBILENET_SSD_BENCHMARK_NOTES["results"]
        assert "latest_results" in results["latest_link"]
        assert results["timing_unit"] == "microseconds"
        assert "--generate_json" in results["json_run_command"]

    def test_notes_input_list_format(self) -> None:
        fmt = MOBILENET_SSD_BENCHMARK_NOTES["output_layers"]["input_list_format"]
        assert "#" in fmt
        assert "Postprocessor" in fmt

    def test_notes_output_update_warning(self) -> None:
        note = MOBILENET_SSD_BENCHMARK_NOTES["output_layers"]["update_note"]
        assert "retrained" in note.lower() or "change" in note.lower()

    def test_timing_note_mentions_detailed(self) -> None:
        assert "detailed" in TIMING_NOTE.lower()

    def test_defaults_cpu_fallback_true(self) -> None:
        assert MOBILENET_SSD_DEFAULTS["cpu_fallback"] is True

    def test_defaults_runtimes_gpu(self) -> None:
        assert "GPU" in MOBILENET_SSD_DEFAULTS["runtimes"]

    def test_buffer_types_valid(self) -> None:
        for bt in MOBILENET_SSD_DEFAULTS["buffer_types"]:
            assert bt in VALID_BUFFER_TYPES

    def test_runtimes_valid(self) -> None:
        for rt in MOBILENET_SSD_DEFAULTS["runtimes"]:
            assert rt in VALID_RUNTIMES

    def test_exported_from_benchmarks_package(self) -> None:
        from quad.benchmarks import (  # noqa: F401
            MOBILENET_SSD_BENCHMARK_NOTES,
            MOBILENET_SSD_OUTPUT_LAYERS,
            SNPEBenchmarkConfig,
            build_mobilenet_ssd_benchmark_config,
            build_mobilenet_ssd_input_list,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Templates
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkTemplates:
    def _templates_dir(self) -> Path:
        # tests/unit/test_benchmarks/ → project root is 3 levels up
        return Path(__file__).parent.parent.parent.parent / "templates"

    def test_benchmark_config_template_exists(self) -> None:
        tmpl = self._templates_dir() / "snpe" / "benchmarking" / "benchmark_config.json.j2"
        assert tmpl.exists()

    def test_imagelist_template_exists(self) -> None:
        tmpl = self._templates_dir() / "snpe" / "benchmarking" / "imagelist.txt.j2"
        assert tmpl.exists()

    def test_benchmark_config_template_renders(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(
            loader=FileSystemLoader(str(self._templates_dir())),
            trim_blocks=True, lstrip_blocks=True,
        )
        tmpl = env.get_template("snpe/benchmarking/benchmark_config.json.j2")
        rendered = tmpl.render(
            name="mobilenet_ssd",
            host_root_path="mobilenet_ssd",
            host_results_dir="mobilenet_ssd/results",
            device_path="/data/local/tmp/snpebm",
            devices=["454d40f3"],
            runs=2,
            model_name="mobilenet_ssd",
            dlc_path="/tmp/mobilenet_ssd.dlc",
            input_list_path="/tmp/imagelist.txt",
            data_dirs=["/tmp/images"],
            runtimes=["GPU"],
            measurements=["timing"],
            cpu_fallback=True,
            buffer_types=["ub_float", "ub_tf8"],
            profiling_level="detailed",
        )
        # Should be valid JSON
        parsed = json.loads(rendered)
        assert parsed["Name"] == "mobilenet_ssd"
        assert parsed["CpuFallback"] is True
        assert "ub_tf8" in parsed["BufferTypes"]

    def test_imagelist_template_renders(self) -> None:
        from jinja2 import Environment, FileSystemLoader
        env = Environment(
            loader=FileSystemLoader(str(self._templates_dir())),
            trim_blocks=True, lstrip_blocks=True,
        )
        tmpl = env.get_template("snpe/benchmarking/imagelist.txt.j2")
        rendered = tmpl.render(
            output_layers=[
                "Postprocessor/BatchMultiClassNonMaxSuppression",
                "add_6",
            ],
            image_paths=["tmp/0#.rawtensor", "tmp/1#.rawtensor"],
        )
        lines = [l for l in rendered.splitlines() if l.strip()]
        assert lines[0].startswith("#")
        assert "Postprocessor" in lines[0]
        assert "tmp/0#.rawtensor" in rendered
