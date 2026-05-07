"""Tests for burst mode configuration and platform options builder."""

from __future__ import annotations

import pytest

from quad.utils.perf import (
    BURST_CPU_SLEEP_THRESHOLD_MS,
    BURST_DEFAULT_INACTIVITY_TIMEOUT_MS,
    BURST_DEFAULT_INACTIVITY_TIMEOUT_US,
    BurstModeConfig,
    build_platform_options,
)


class TestBurstModeConstants:
    def test_cpu_sleep_threshold(self) -> None:
        """CPUs can enter deep sleep if inference > 100ms (from docs)."""
        assert BURST_CPU_SLEEP_THRESHOLD_MS == 100.0

    def test_default_inactivity_timeout(self) -> None:
        """Default DSP vote hold = 300ms (from docs)."""
        assert BURST_DEFAULT_INACTIVITY_TIMEOUT_MS == 300.0
        assert BURST_DEFAULT_INACTIVITY_TIMEOUT_US == 300_000

    def test_microsecond_conversion(self) -> None:
        assert BURST_DEFAULT_INACTIVITY_TIMEOUT_US == int(
            BURST_DEFAULT_INACTIVITY_TIMEOUT_MS * 1000
        )


class TestBurstModeConfig:
    def test_default_is_300ms(self) -> None:
        cfg = BurstModeConfig()
        assert cfg.inactivity_timeout_ms == 300.0
        assert cfg.inactivity_timeout_us == 300_000

    def test_10ms_timeout(self) -> None:
        """Example from docs: user sets 10ms → inactivityTimeout:10000."""
        cfg = BurstModeConfig(inactivity_timeout_ms=10.0)
        assert cfg.inactivity_timeout_us == 10_000
        assert cfg.platform_option_string == "inactivityTimeout:10000"

    def test_platform_option_format(self) -> None:
        """Platform option must be 'inactivityTimeout:VALUE_IN_US'."""
        cfg = BurstModeConfig(inactivity_timeout_ms=300.0)
        option = cfg.platform_option_string
        assert option.startswith("inactivityTimeout:")
        assert ":" in option
        # Value must be in microseconds
        value = int(option.split(":")[1])
        assert value == 300_000

    def test_combined_with_unsigned_pd(self) -> None:
        cfg = BurstModeConfig(inactivity_timeout_ms=10.0)
        combined = cfg.combined_with_pd("unsigned")
        assert "unsignedPD:ON" in combined
        assert "inactivityTimeout:10000" in combined
        assert ";" in combined  # Semicolon separator

    def test_combined_with_signed_pd(self) -> None:
        cfg = BurstModeConfig(inactivity_timeout_ms=10.0)
        combined = cfg.combined_with_pd("signed")
        assert "unsignedPD:OFF" in combined
        assert "inactivityTimeout:10000" in combined

    def test_is_non_default_timeout(self) -> None:
        assert BurstModeConfig(300.0).is_non_default_timeout() is False
        assert BurstModeConfig(10.0).is_non_default_timeout() is True

    def test_repr_contains_timeout_and_option(self) -> None:
        cfg = BurstModeConfig(10.0)
        r = repr(cfg)
        assert "10" in r
        assert "inactivityTimeout" in r

    def test_various_timeout_values(self) -> None:
        for ms, expected_us in [(10, 10_000), (100, 100_000), (500, 500_000)]:
            cfg = BurstModeConfig(ms)
            assert cfg.inactivity_timeout_us == expected_us


class TestBuildPlatformOptions:
    def test_default_unsigned_pd_only(self) -> None:
        result = build_platform_options()
        assert result == "unsignedPD:ON"

    def test_signed_pd(self) -> None:
        result = build_platform_options(pd_type="signed")
        assert result == "unsignedPD:OFF"

    def test_with_burst_config_10ms(self) -> None:
        """Example from docs: --platform_options 'inactivityTimeout:10000'."""
        result = build_platform_options(
            burst_config=BurstModeConfig(inactivity_timeout_ms=10.0)
        )
        assert "unsignedPD:ON" in result
        assert "inactivityTimeout:10000" in result

    def test_with_burst_config_300ms_default(self) -> None:
        result = build_platform_options(
            burst_config=BurstModeConfig()
        )
        assert "inactivityTimeout:300000" in result

    def test_signed_pd_with_burst(self) -> None:
        result = build_platform_options(
            pd_type="signed",
            burst_config=BurstModeConfig(10.0),
        )
        assert result == "unsignedPD:OFF;inactivityTimeout:10000"

    def test_extra_options(self) -> None:
        result = build_platform_options(
            extra_options={"powerHint": "MAX"},
        )
        assert "powerHint:MAX" in result
        assert "unsignedPD:ON" in result

    def test_combined_all_options(self) -> None:
        result = build_platform_options(
            pd_type="signed",
            burst_config=BurstModeConfig(10.0),
            extra_options={"powerHint": "MAX"},
        )
        assert "unsignedPD:OFF" in result
        assert "inactivityTimeout:10000" in result
        assert "powerHint:MAX" in result
        # All separated by semicolons
        assert result.count(";") >= 2

    def test_semicolons_not_colons_for_option_separator(self) -> None:
        """Platform options use semicolons between options, colons within key:value."""
        result = build_platform_options(
            burst_config=BurstModeConfig(10.0),
        )
        # Between options: semicolons
        # Within key:value: colons
        assert "unsignedPD:ON;inactivityTimeout:10000" == result

    def test_no_burst_means_no_inactivity_timeout(self) -> None:
        result = build_platform_options(pd_type="unsigned")
        assert "inactivityTimeout" not in result
