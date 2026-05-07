"""Unit tests for the QUAD PSNPE module.

Coverage:
    - ExecutionMode enum values
    - RuntimeConfig defaults and field assignment
    - BuildConfig creation, derived properties, and validation
    - ModelConfig.to_build_config()
    - PSNPEManager lifecycle (build / release)
    - execute_sync() — correct number of results
    - execute_output_async() — callback invoked for each input
    - execute_input_output_async() — both callbacks invoked
    - get_input_tensor_names() / get_output_tensor_names()
    - from_model_config() — parses JSON and returns a built manager
    - Multiple RuntimeConfig entries (two DSP pools)
    - Mock mode throughput > 0 FPS
    - Guard: calling execute before build raises RuntimeError
    - PSNPEResult fields are correctly populated
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from quad.psnpe import (
    BuildConfig,
    ExecutionMode,
    ModelConfig,
    PSNPEConfig,
    PSNPEManager,
    RuntimeConfig,
)
from quad.psnpe.manager import PSNPEResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_runtime() -> RuntimeConfig:
    return RuntimeConfig(runtime="dsp", num_instances=2)


def _make_build_config(**overrides: Any) -> BuildConfig:
    defaults: dict[str, Any] = dict(
        container_path="model.dlc",
        runtime_configs=[_make_runtime()],
        output_buffer_names=["output:0"],
    )
    defaults.update(overrides)
    return BuildConfig(**defaults)


def _tiny_input() -> dict[str, bytes]:
    """Single-sample input dict with a 4-byte float32 payload."""
    import struct
    return {"input:0": struct.pack("f", 1.0)}


@pytest.fixture()
def built_manager() -> PSNPEManager:
    """A PSNPEManager that has been built in mock mode."""
    os.environ.setdefault("QUAD_PSNPE_MOCK", "1")
    mgr = PSNPEManager()
    mgr.build(_make_build_config())
    yield mgr
    mgr.release()


# ---------------------------------------------------------------------------
# ExecutionMode
# ---------------------------------------------------------------------------


class TestExecutionMode:
    def test_sync_value(self) -> None:
        assert ExecutionMode.SYNC == "sync"
        assert ExecutionMode.SYNC.value == "sync"

    def test_output_async_value(self) -> None:
        assert ExecutionMode.OUTPUT_ASYNC == "outputAsync"

    def test_input_output_async_value(self) -> None:
        assert ExecutionMode.INPUT_OUTPUT_ASYNC == "inputOutputAsync"

    def test_is_str_subclass(self) -> None:
        assert isinstance(ExecutionMode.SYNC, str)

    def test_all_three_members(self) -> None:
        members = {m.value for m in ExecutionMode}
        assert members == {"sync", "outputAsync", "inputOutputAsync"}

    def test_from_string_roundtrip(self) -> None:
        for mode in ExecutionMode:
            assert ExecutionMode(mode.value) is mode


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class TestRuntimeConfig:
    def test_required_field(self) -> None:
        rc = RuntimeConfig(runtime="cpu")
        assert rc.runtime == "cpu"

    def test_defaults(self) -> None:
        rc = RuntimeConfig(runtime="dsp")
        assert rc.num_instances == 1
        assert rc.performance_profile == "burst"
        assert rc.enable_cpu_fallback is True
        assert rc.user_buffer_mode == "float"

    def test_override_all_fields(self) -> None:
        rc = RuntimeConfig(
            runtime="gpu",
            num_instances=8,
            performance_profile="power_saver",
            enable_cpu_fallback=False,
            user_buffer_mode="tf8",
        )
        assert rc.runtime == "gpu"
        assert rc.num_instances == 8
        assert rc.performance_profile == "power_saver"
        assert rc.enable_cpu_fallback is False
        assert rc.user_buffer_mode == "tf8"

    def test_multiple_runtimes_are_independent(self) -> None:
        r1 = RuntimeConfig(runtime="dsp", num_instances=4)
        r2 = RuntimeConfig(runtime="cpu", num_instances=1)
        assert r1.runtime != r2.runtime
        assert r1.num_instances != r2.num_instances


# ---------------------------------------------------------------------------
# BuildConfig
# ---------------------------------------------------------------------------


class TestBuildConfig:
    def test_basic_creation(self) -> None:
        cfg = _make_build_config()
        assert cfg.container_path == "model.dlc"
        assert len(cfg.runtime_configs) == 1
        assert cfg.output_buffer_names == ["output:0"]

    def test_defaults(self) -> None:
        cfg = _make_build_config()
        assert cfg.transmission_mode is ExecutionMode.SYNC
        assert cfg.enable_init_cache is False
        assert cfg.profiling_level == "off"
        assert cfg.output_thread_numbers == 1
        assert cfg.input_thread_numbers == 1
        assert cfg.platform_options == ""
        assert cfg.bulk_size == 1

    def test_total_instances_single_runtime(self) -> None:
        cfg = _make_build_config(
            runtime_configs=[RuntimeConfig(runtime="dsp", num_instances=4)]
        )
        assert cfg.total_instances == 4

    def test_total_instances_multiple_runtimes(self) -> None:
        cfg = _make_build_config(
            runtime_configs=[
                RuntimeConfig(runtime="dsp", num_instances=4),
                RuntimeConfig(runtime="cpu", num_instances=2),
            ]
        )
        assert cfg.total_instances == 6

    def test_validate_passes_for_good_config(self) -> None:
        _make_build_config().validate()  # Must not raise

    def test_validate_rejects_empty_container_path(self) -> None:
        with pytest.raises(ValueError, match="container_path"):
            _make_build_config(container_path="").validate()

    def test_validate_rejects_empty_runtime_configs(self) -> None:
        with pytest.raises(ValueError, match="RuntimeConfig"):
            _make_build_config(runtime_configs=[]).validate()

    def test_validate_rejects_empty_output_names(self) -> None:
        with pytest.raises(ValueError, match="output_buffer_names"):
            _make_build_config(output_buffer_names=[]).validate()

    def test_validate_rejects_bad_profiling_level(self) -> None:
        with pytest.raises(ValueError, match="profiling_level"):
            _make_build_config(profiling_level="verbose").validate()

    def test_validate_rejects_bulk_size_zero(self) -> None:
        with pytest.raises(ValueError, match="bulk_size"):
            _make_build_config(bulk_size=0).validate()

    def test_non_default_transmission_mode(self) -> None:
        cfg = _make_build_config(transmission_mode=ExecutionMode.OUTPUT_ASYNC)
        assert cfg.transmission_mode is ExecutionMode.OUTPUT_ASYNC

    def test_multiple_output_buffers(self) -> None:
        cfg = _make_build_config(
            output_buffer_names=["logits:0", "embeddings:0"]
        )
        assert len(cfg.output_buffer_names) == 2


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_to_build_config_basic(self) -> None:
        mc = ModelConfig(
            name="resnet50",
            model_file="resnet50.dlc",
            build_configs=[{"runtime": "dsp", "num_instances": 2}],
        )
        bc = mc.to_build_config(["output:0"])
        assert bc.container_path == "resnet50.dlc"
        assert len(bc.runtime_configs) == 1
        assert bc.runtime_configs[0].runtime == "dsp"
        assert bc.runtime_configs[0].num_instances == 2

    def test_to_build_config_inherits_execute_mode(self) -> None:
        mc = ModelConfig(
            name="m",
            model_file="m.dlc",
            execute_mode=ExecutionMode.OUTPUT_ASYNC,
            build_configs=[{"runtime": "cpu"}],
        )
        bc = mc.to_build_config(["out:0"])
        assert bc.transmission_mode is ExecutionMode.OUTPUT_ASYNC

    def test_to_build_config_defaults_runtime_when_empty(self) -> None:
        mc = ModelConfig(name="m", model_file="m.dlc")
        bc = mc.to_build_config(["out:0"])
        assert len(bc.runtime_configs) == 1
        assert bc.runtime_configs[0].runtime == "cpu"


# ---------------------------------------------------------------------------
# PSNPEConfig (backward-compatibility alias)
# ---------------------------------------------------------------------------


class TestPSNPEConfig:
    def test_wraps_build_config(self) -> None:
        bc = _make_build_config()
        pc = PSNPEConfig.from_build_config(bc)
        assert pc.build_config is bc


# ---------------------------------------------------------------------------
# PSNPEManager — build / release
# ---------------------------------------------------------------------------


class TestPSNPEManagerBuild:
    def setup_method(self) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"

    def test_build_returns_true(self) -> None:
        mgr = PSNPEManager()
        assert mgr.build(_make_build_config()) is True
        mgr.release()

    def test_build_marks_manager_as_built(self) -> None:
        mgr = PSNPEManager()
        mgr.build(_make_build_config())
        assert mgr._built is True
        mgr.release()

    def test_release_clears_built_flag(self) -> None:
        mgr = PSNPEManager()
        mgr.build(_make_build_config())
        mgr.release()
        assert not mgr._built

    def test_double_build_is_safe(self) -> None:
        mgr = PSNPEManager()
        mgr.build(_make_build_config())
        assert mgr.build(_make_build_config()) is True
        mgr.release()

    def test_build_invalid_config_raises(self) -> None:
        mgr = PSNPEManager()
        with pytest.raises(ValueError):
            mgr.build(_make_build_config(container_path=""))

    def test_context_manager(self) -> None:
        with PSNPEManager() as mgr:
            built = mgr.build(_make_build_config())
            assert built
        assert not mgr._built


# ---------------------------------------------------------------------------
# PSNPEManager — tensor name accessors
# ---------------------------------------------------------------------------


class TestTensorNames:
    def test_get_output_tensor_names(self, built_manager: PSNPEManager) -> None:
        names = built_manager.get_output_tensor_names()
        assert isinstance(names, list)
        assert len(names) >= 1

    def test_get_input_tensor_names(self, built_manager: PSNPEManager) -> None:
        names = built_manager.get_input_tensor_names()
        assert isinstance(names, list)
        assert len(names) >= 1

    def test_output_names_match_build_config(self, built_manager: PSNPEManager) -> None:
        assert "output:0" in built_manager.get_output_tensor_names()

    def test_raises_before_build(self) -> None:
        mgr = PSNPEManager()
        with pytest.raises(RuntimeError, match="build"):
            mgr.get_output_tensor_names()


# ---------------------------------------------------------------------------
# PSNPEManager — execute_sync
# ---------------------------------------------------------------------------


class TestExecuteSync:
    def test_returns_list_of_results(self, built_manager: PSNPEManager) -> None:
        inputs = [_tiny_input() for _ in range(3)]
        results = built_manager.execute_sync(inputs)
        assert len(results) == 3

    def test_each_result_has_one_output_dict(self, built_manager: PSNPEManager) -> None:
        results = built_manager.execute_sync([_tiny_input()])
        assert len(results[0].outputs) == 1

    def test_output_keys_match_config(self, built_manager: PSNPEManager) -> None:
        results = built_manager.execute_sync([_tiny_input()])
        assert "output:0" in results[0].outputs[0]

    def test_result_is_psnpe_result(self, built_manager: PSNPEManager) -> None:
        results = built_manager.execute_sync([_tiny_input()])
        assert isinstance(results[0], PSNPEResult)

    def test_latency_is_positive(self, built_manager: PSNPEManager) -> None:
        results = built_manager.execute_sync([_tiny_input()])
        assert results[0].latency_ms > 0

    def test_throughput_fps_positive(self, built_manager: PSNPEManager) -> None:
        inputs = [_tiny_input() for _ in range(4)]
        results = built_manager.execute_sync(inputs)
        assert results[0].throughput_fps > 0

    def test_instances_used_matches_config(self, built_manager: PSNPEManager) -> None:
        results = built_manager.execute_sync([_tiny_input()])
        assert results[0].instances_used == 2  # _make_runtime() -> num_instances=2

    def test_mode_is_sync(self, built_manager: PSNPEManager) -> None:
        results = built_manager.execute_sync([_tiny_input()])
        assert results[0].mode is ExecutionMode.SYNC

    def test_raises_before_build(self) -> None:
        mgr = PSNPEManager()
        with pytest.raises(RuntimeError, match="build"):
            mgr.execute_sync([_tiny_input()])

    def test_multiple_inputs_produce_correct_count(
        self, built_manager: PSNPEManager
    ) -> None:
        n = 10
        results = built_manager.execute_sync([_tiny_input() for _ in range(n)])
        assert len(results) == n


# ---------------------------------------------------------------------------
# PSNPEManager — execute_output_async
# ---------------------------------------------------------------------------


class TestExecuteOutputAsync:
    def test_callback_called_for_each_input(self, built_manager: PSNPEManager) -> None:
        received: list[int] = []

        def cb(idx: int, out: dict) -> None:
            received.append(idx)

        inputs = [_tiny_input() for _ in range(5)]
        built_manager.execute_output_async(inputs, output_callback=cb)
        assert sorted(received) == list(range(5))

    def test_callback_receives_output_dict(self, built_manager: PSNPEManager) -> None:
        results: list[dict] = []

        def cb(idx: int, out: dict) -> None:
            results.append(out)

        built_manager.execute_output_async([_tiny_input()], output_callback=cb)
        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert "output:0" in results[0]

    def test_no_callback_does_not_raise(self, built_manager: PSNPEManager) -> None:
        built_manager.execute_output_async([_tiny_input()], output_callback=None)

    def test_raises_before_build(self) -> None:
        mgr = PSNPEManager()
        with pytest.raises(RuntimeError, match="build"):
            mgr.execute_output_async([_tiny_input()])


# ---------------------------------------------------------------------------
# PSNPEManager — execute_input_output_async
# ---------------------------------------------------------------------------


class TestExecuteInputOutputAsync:
    def test_input_callback_called_for_each_path(
        self, built_manager: PSNPEManager
    ) -> None:
        loaded: list[str] = []

        def in_cb(path: str) -> dict:
            loaded.append(path)
            return _tiny_input()

        outputs: list[int] = []

        def out_cb(idx: int, out: dict) -> None:
            outputs.append(idx)

        paths = [f"/fake/frame_{i}.raw" for i in range(4)]
        built_manager.execute_input_output_async(
            file_paths=paths,
            input_callback=in_cb,
            output_callback=out_cb,
        )
        assert sorted(loaded) == sorted(paths)

    def test_output_callback_called_for_each_input(
        self, built_manager: PSNPEManager
    ) -> None:
        received: list[int] = []

        def out_cb(idx: int, out: dict) -> None:
            received.append(idx)

        paths = [f"/fake/frame_{i}.raw" for i in range(6)]
        built_manager.execute_input_output_async(
            file_paths=paths,
            input_callback=lambda p: _tiny_input(),
            output_callback=out_cb,
        )
        assert sorted(received) == list(range(6))

    def test_output_callback_receives_expected_tensor(
        self, built_manager: PSNPEManager
    ) -> None:
        out_maps: list[dict] = []

        built_manager.execute_input_output_async(
            file_paths=["/fake/frame_0.raw"],
            input_callback=lambda p: _tiny_input(),
            output_callback=lambda i, m: out_maps.append(m),
        )
        assert "output:0" in out_maps[0]

    def test_raises_before_build(self) -> None:
        mgr = PSNPEManager()
        with pytest.raises(RuntimeError, match="build"):
            mgr.execute_input_output_async(
                file_paths=["/fake/frame.raw"],
                input_callback=lambda p: _tiny_input(),
                output_callback=lambda i, m: None,
            )


# ---------------------------------------------------------------------------
# Multiple runtime configs (two DSP instances)
# ---------------------------------------------------------------------------


class TestMultipleRuntimeConfigs:
    def setup_method(self) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"

    def test_build_with_two_dsp_instances(self) -> None:
        cfg = _make_build_config(
            runtime_configs=[
                RuntimeConfig(runtime="dsp", num_instances=2),
                RuntimeConfig(runtime="dsp", num_instances=2),
            ]
        )
        assert cfg.total_instances == 4
        mgr = PSNPEManager()
        assert mgr.build(cfg) is True
        mgr.release()

    def test_execute_sync_with_two_runtimes(self) -> None:
        cfg = _make_build_config(
            runtime_configs=[
                RuntimeConfig(runtime="dsp", num_instances=2),
                RuntimeConfig(runtime="cpu", num_instances=1),
            ]
        )
        mgr = PSNPEManager()
        mgr.build(cfg)
        results = mgr.execute_sync([_tiny_input(), _tiny_input()])
        assert len(results) == 2
        mgr.release()

    def test_instances_used_reflects_total(self) -> None:
        cfg = _make_build_config(
            runtime_configs=[
                RuntimeConfig(runtime="dsp", num_instances=3),
                RuntimeConfig(runtime="cpu", num_instances=1),
            ]
        )
        mgr = PSNPEManager()
        mgr.build(cfg)
        results = mgr.execute_sync([_tiny_input()])
        assert results[0].instances_used == 4
        mgr.release()


# ---------------------------------------------------------------------------
# from_model_config factory
# ---------------------------------------------------------------------------


class TestFromModelConfig:
    _JSON_CONTENT = [
        {
            "name": "resnet50",
            "model_file": "resnet50.dlc",
            "execute_mode": "sync",
            "enable_init_cache": False,
            "bulk_size": 2,
            "build_configs": [
                {"runtime": "dsp", "num_instances": 4, "performance_profile": "burst"}
            ],
        },
        {
            "name": "mobilenet",
            "model_file": "mobilenet.dlc",
            "execute_mode": "outputAsync",
            "bulk_size": 1,
            "build_configs": [{"runtime": "cpu", "num_instances": 2}],
        },
    ]

    @pytest.fixture()
    def config_json(self, tmp_path) -> str:
        p = tmp_path / "model_configs.json"
        p.write_text(json.dumps(self._JSON_CONTENT))
        return str(p)

    def test_returns_built_manager(self, config_json: str) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"
        mgr = PSNPEManager.from_model_config(config_json, "resnet50")
        assert mgr._built
        mgr.release()

    def test_parses_bulk_size(self, config_json: str) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"
        mgr = PSNPEManager.from_model_config(config_json, "resnet50")
        assert mgr._config is not None
        assert mgr._config.bulk_size == 2
        mgr.release()

    def test_parses_runtime(self, config_json: str) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"
        mgr = PSNPEManager.from_model_config(config_json, "resnet50")
        assert mgr._config.runtime_configs[0].runtime == "dsp"
        assert mgr._config.runtime_configs[0].num_instances == 4
        mgr.release()

    def test_parses_execute_mode(self, config_json: str) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"
        mgr = PSNPEManager.from_model_config(config_json, "mobilenet")
        assert mgr._config.transmission_mode is ExecutionMode.OUTPUT_ASYNC
        mgr.release()

    def test_raises_for_missing_model(self, config_json: str) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"
        with pytest.raises(KeyError, match="nonexistent"):
            PSNPEManager.from_model_config(config_json, "nonexistent")

    def test_execute_after_from_model_config(self, config_json: str) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"
        mgr = PSNPEManager.from_model_config(config_json, "resnet50")
        results = mgr.execute_sync([_tiny_input()])
        assert len(results) == 1
        mgr.release()


# ---------------------------------------------------------------------------
# Mock mode throughput
# ---------------------------------------------------------------------------


class TestMockModeThroughput:
    def setup_method(self) -> None:
        os.environ["QUAD_PSNPE_MOCK"] = "1"

    def test_throughput_fps_greater_than_zero(self) -> None:
        mgr = PSNPEManager()
        mgr.build(_make_build_config(
            runtime_configs=[RuntimeConfig(runtime="dsp", num_instances=4)]
        ))
        inputs = [_tiny_input() for _ in range(8)]
        results = mgr.execute_sync(inputs)
        assert all(r.throughput_fps > 0 for r in results)
        mgr.release()

    def test_more_instances_does_not_crash(self) -> None:
        mgr = PSNPEManager()
        mgr.build(_make_build_config(
            runtime_configs=[RuntimeConfig(runtime="dsp", num_instances=8)]
        ))
        results = mgr.execute_sync([_tiny_input() for _ in range(16)])
        assert len(results) == 16
        mgr.release()

    def test_latency_ms_is_finite(self) -> None:
        import math
        mgr = PSNPEManager()
        mgr.build(_make_build_config())
        results = mgr.execute_sync([_tiny_input()])
        assert math.isfinite(results[0].latency_ms)
        mgr.release()
