"""Tests for SNPE Low-Level Performance APIs (SNPE 2.22+)."""

from __future__ import annotations

import pytest

from quad.utils.perf_profile import (
    DCVSVoltageCorner,
    SNPEPerfConfig,
    SNPEPerfProfile,
    SnpePerfPreset,
    create_multi_instance_sync_profile,
    create_sustained_high_performance_config,
)


class TestDCVSVoltageCorner:
    def test_all_corners_exist(self) -> None:
        assert DCVSVoltageCorner.TURBO
        assert DCVSVoltageCorner.NOM
        assert DCVSVoltageCorner.SVS
        assert DCVSVoltageCorner.DISABLE

    def test_performance_levels_ordered(self) -> None:
        """Turbo > NOM > SVS > MIN > DISABLE."""
        assert DCVSVoltageCorner.TURBO.performance_level > DCVSVoltageCorner.NOM.performance_level
        assert DCVSVoltageCorner.NOM.performance_level > DCVSVoltageCorner.SVS.performance_level
        assert DCVSVoltageCorner.SVS.performance_level > DCVSVoltageCorner.MIN.performance_level
        assert DCVSVoltageCorner.MIN.performance_level > DCVSVoltageCorner.DISABLE.performance_level

    def test_turbo_plus_higher_than_turbo(self) -> None:
        assert DCVSVoltageCorner.TURBO_PLUS.performance_level > DCVSVoltageCorner.TURBO.performance_level


class TestSnpePerfPreset:
    def test_all_10_presets_exist(self) -> None:
        presets = list(SnpePerfPreset)
        assert len(presets) == 10

    def test_rpc_polling_presets(self) -> None:
        """BURST, SHP, HP have RPC polling."""
        assert SnpePerfPreset.BURST.has_rpc_polling is True
        assert SnpePerfPreset.SUSTAINED_HIGH_PERFORMANCE.has_rpc_polling is True
        assert SnpePerfPreset.HIGH_PERFORMANCE.has_rpc_polling is True
        assert SnpePerfPreset.BALANCED.has_rpc_polling is False

    def test_hysteresis_only_burst_shp(self) -> None:
        """300ms hysteresis only for BURST and SHP."""
        assert SnpePerfPreset.BURST.has_hysteresis_300ms is True
        assert SnpePerfPreset.SUSTAINED_HIGH_PERFORMANCE.has_hysteresis_300ms is True
        assert SnpePerfPreset.HIGH_PERFORMANCE.has_hysteresis_300ms is False

    def test_no_async_voting_for_burst_shp(self) -> None:
        """Async voting is DISABLED for BURST and SHP."""
        assert SnpePerfPreset.BURST.has_async_voting is False
        assert SnpePerfPreset.SUSTAINED_HIGH_PERFORMANCE.has_async_voting is False

    def test_async_voting_for_others(self) -> None:
        assert SnpePerfPreset.BALANCED.has_async_voting is True
        assert SnpePerfPreset.HIGH_PERFORMANCE.has_async_voting is True

    def test_system_settings_no_async_voting(self) -> None:
        """SYSTEM_SETTINGS: client votes directly — no async voting."""
        assert SnpePerfPreset.SYSTEM_SETTINGS.has_async_voting is False


class TestSNPEPerfProfile:
    def test_create_empty_profile(self) -> None:
        p = SNPEPerfProfile()
        assert p.preset is None

    def test_create_with_preset(self) -> None:
        p = SNPEPerfProfile(preset=SnpePerfPreset.BURST)
        assert p.preset == SnpePerfPreset.BURST

    def test_c_api_create_call_no_preset(self) -> None:
        p = SNPEPerfProfile()
        assert p.c_api_create_call == "Snpe_SNPEPerfProfile_Create();"

    def test_c_api_create_call_with_preset(self) -> None:
        p = SNPEPerfProfile(preset=SnpePerfPreset.BURST)
        assert "BURST" in p.c_api_create_call
        assert "Snpe_SNPEPerfProfile_CreatePreset" in p.c_api_create_call

    def test_cpp_api_create_call(self) -> None:
        p = SNPEPerfProfile(preset=SnpePerfPreset.HIGH_PERFORMANCE)
        assert "SNPEPerfProfile" in p.cpp_api_create_call
        assert "HIGH_PERFORMANCE" in p.cpp_api_create_call

    def test_disable_done_votes(self) -> None:
        """Multi-instance sync: disable all 'done' votes."""
        p = SNPEPerfProfile(preset=SnpePerfPreset.BURST)
        p.disable_done_votes()
        assert p.bus_vcorner_min_done == DCVSVoltageCorner.DISABLE
        assert p.bus_vcorner_target_done == DCVSVoltageCorner.DISABLE
        assert p.bus_vcorner_max_done == DCVSVoltageCorner.DISABLE
        assert p.core_vcorner_min_done == DCVSVoltageCorner.DISABLE
        assert p.core_vcorner_target_done == DCVSVoltageCorner.DISABLE
        assert p.core_vcorner_max_done == DCVSVoltageCorner.DISABLE

    def test_disable_done_votes_returns_self(self) -> None:
        p = SNPEPerfProfile()
        result = p.disable_done_votes()
        assert result is p  # Fluent interface

    def test_set_hysteresis(self) -> None:
        p = SNPEPerfProfile()
        p.set_hysteresis(500.0)
        assert p.dsp_hysteresis_time_ms == 500.0

    def test_set_hysteresis_zero_disables(self) -> None:
        p = SNPEPerfProfile()
        p.set_hysteresis(0.0)
        assert p.dsp_hysteresis_time_ms == 0.0

    def test_to_yaml_section_has_required_keys(self) -> None:
        p = SNPEPerfProfile(high_performance_mode=True)
        y = p.to_yaml_section("execute")
        assert "HIGH_PERFORMANCE_MODE" in y
        assert "DSP_ENABLE_DCVS_START" in y
        assert "DSP_ENABLE_DCVS_DONE" in y


class TestSNPEPerfConfig:
    def test_to_yaml_dict_has_sections(self) -> None:
        cfg = SNPEPerfConfig()
        d = cfg.to_yaml_dict()
        assert "general" in d
        assert "init" in d
        assert "execute" in d
        assert "deinit" in d

    def test_general_has_hysteresis(self) -> None:
        cfg = SNPEPerfConfig()
        assert "DSP_HYSTERESIS_TIME_US" in cfg.to_yaml_dict()["general"]

    def test_to_yaml_string_is_non_empty(self) -> None:
        cfg = SNPEPerfConfig()
        yaml_str = cfg.to_yaml_string()
        assert len(yaml_str) > 0
        assert "general:" in yaml_str


class TestFactoryFunctions:
    def test_shp_config_has_turbo_start(self) -> None:
        cfg = create_sustained_high_performance_config()
        init_yaml = cfg.init.to_yaml_section("init")
        assert "TURBO" in init_yaml.get("BUS_VOLTAGE_CORNER_MIN_START", "")

    def test_shp_deinit_drops_to_min(self) -> None:
        cfg = create_sustained_high_performance_config()
        deinit_yaml = cfg.deinit.to_yaml_section("deinit")
        assert "MIN" in deinit_yaml.get("BUS_VOLTAGE_CORNER_MIN_DONE", "")

    def test_shp_general_has_rpc_polling(self) -> None:
        cfg = create_sustained_high_performance_config()
        assert cfg.general_rpc_polling_us == 9999  # SHP default from docs

    def test_shp_async_voting_disabled(self) -> None:
        cfg = create_sustained_high_performance_config()
        assert cfg.general_async_voting_enable is False  # SHP: no async voting

    def test_multi_instance_profile_disables_done(self) -> None:
        p = create_multi_instance_sync_profile(SnpePerfPreset.BURST)
        assert p.preset == SnpePerfPreset.BURST
        assert p.bus_vcorner_min_done == DCVSVoltageCorner.DISABLE
        assert p.core_vcorner_max_done == DCVSVoltageCorner.DISABLE
