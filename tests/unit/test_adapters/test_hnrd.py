"""Tests for HNRD (Hexagon NPU Runtime Driver) utilities."""

from __future__ import annotations

import pytest

from quad.adapters.hnrd import (
    HNRD_MIN_SDK_VERSION,
    CacheCompatibilityMode,
    HNRDDeploymentConfig,
    SNPE_ERRORCODE_DLCACHING_SUBOPTIMAL_CACHE,
    SNPE_ERRORCODE_QNN_CONTEXT_ERROR_CREATE_FROM_BINARY,
    SNPE_ERRORCODE_QNN_COMMON_ERROR_NOT_SUPPORTED,
    get_cache_error_type,
    get_cache_recovery_strategy,
    is_hnrd_log,
)


class TestHNRDConstants:
    def test_min_version(self) -> None:
        assert HNRD_MIN_SDK_VERSION == "2.22.2"

    def test_error_code_constants(self) -> None:
        assert "SUBOPTIMAL" in SNPE_ERRORCODE_DLCACHING_SUBOPTIMAL_CACHE
        assert "CREATE_FROM_BINARY" in SNPE_ERRORCODE_QNN_CONTEXT_ERROR_CREATE_FROM_BINARY
        assert "NOT_SUPPORTED" in SNPE_ERRORCODE_QNN_COMMON_ERROR_NOT_SUPPORTED


class TestCacheCompatibilityMode:
    def test_all_modes_exist(self) -> None:
        assert CacheCompatibilityMode.STRICT
        assert CacheCompatibilityMode.PERMISSIVE
        assert CacheCompatibilityMode.ALWAYS_GENERATE_NEW_CACHE

    def test_c_api_enum_values(self) -> None:
        assert CacheCompatibilityMode.STRICT.value == "SNPE_CACHE_COMPATIBILITY_STRICT"
        assert CacheCompatibilityMode.PERMISSIVE.value == "SNPE_CACHE_COMPATIBILITY_PERMISSIVE"
        assert CacheCompatibilityMode.ALWAYS_GENERATE_NEW_CACHE.value == "SNPE_CACHE_COMPATIBILITY_ALWAYS_GENERATE_NEW_CACHE"

    def test_all_modes_have_descriptions(self) -> None:
        for mode in CacheCompatibilityMode:
            assert len(mode.description) > 0


class TestHNRDLogDetection:
    def test_traditional_not_available_log(self) -> None:
        log = "[WARNING] QnnDsp <W> Traditional path not available. Switching to user driver path"
        assert is_hnrd_log(log) is True

    def test_driver_loaded_log(self) -> None:
        log = "[WARNING] QnnDsp <W> HTP user driver is loaded. Switched to user driver path"
        assert is_hnrd_log(log) is True

    def test_normal_log_not_hnrd(self) -> None:
        assert is_hnrd_log("SNPE Version: 2.45.0") is False
        assert is_hnrd_log("Inference: 5.2ms") is False


class TestHNRDDeploymentConfig:
    def test_traditional_bundles_all_htp_files(self) -> None:
        cfg = HNRDDeploymentConfig(use_hnrd=False, hexagon_version="v73")
        files = cfg.files_to_bundle
        assert "SNPE.dll" in files
        assert "SnpeHtpPrepare.dll" in files
        assert "SnpeHtpV73Stub.dll" in files
        assert "SnpeHtpV73Skel.so" in files

    def test_hnrd_only_bundles_snpe_dll(self) -> None:
        """HNRD path: only SNPE.dll needed; driver handles platform libs."""
        cfg = HNRDDeploymentConfig(use_hnrd=True, hexagon_version="v73")
        files = cfg.files_to_bundle
        assert files == ["SNPE.dll"]
        assert "SnpeHtpV73Stub.dll" not in files  # Not bundled → HNRD path

    def test_path_name(self) -> None:
        assert "Traditional" in HNRDDeploymentConfig(use_hnrd=False).path_name
        assert "HNRD" in HNRDDeploymentConfig(use_hnrd=True).path_name

    def test_different_hexagon_versions(self) -> None:
        for v in ("v68", "v73", "v75"):
            cfg = HNRDDeploymentConfig(use_hnrd=False, hexagon_version=v)
            files = cfg.files_to_bundle
            ver_num = v.lstrip("v")
            assert f"SnpeHtpV{ver_num}Stub.dll" in files


class TestCacheErrorHandling:
    def test_suboptimal_error_classified(self) -> None:
        assert get_cache_error_type(SNPE_ERRORCODE_DLCACHING_SUBOPTIMAL_CACHE) == "suboptimal"

    def test_create_from_binary_classified_invalid(self) -> None:
        assert get_cache_error_type(SNPE_ERRORCODE_QNN_CONTEXT_ERROR_CREATE_FROM_BINARY) == "invalid"

    def test_not_supported_classified_invalid(self) -> None:
        assert get_cache_error_type(SNPE_ERRORCODE_QNN_COMMON_ERROR_NOT_SUPPORTED) == "invalid"

    def test_unknown_error_classified_other(self) -> None:
        assert get_cache_error_type("SNPE_ERRORCODE_UNKNOWN_XYZ") == "other"


class TestCacheRecoveryStrategies:
    """Verify the background thread recovery pattern from HNRD docs."""

    def test_suboptimal_uses_permissive_then_regenerate(self) -> None:
        strategy = get_cache_recovery_strategy(SNPE_ERRORCODE_DLCACHING_SUBOPTIMAL_CACHE)
        # Immediate: use permissive to keep serving
        assert strategy["immediate_action"] == CacheCompatibilityMode.PERMISSIVE
        # Background: regenerate better cache
        assert strategy["background_action"] == CacheCompatibilityMode.ALWAYS_GENERATE_NEW_CACHE
        assert strategy["enable_init_cache"] is True

    def test_invalid_forces_regeneration(self) -> None:
        strategy = get_cache_recovery_strategy(SNPE_ERRORCODE_QNN_CONTEXT_ERROR_CREATE_FROM_BINARY)
        assert strategy["immediate_action"] == CacheCompatibilityMode.ALWAYS_GENERATE_NEW_CACHE
        assert strategy["background_action"] is None  # No background needed

    def test_all_strategies_have_description(self) -> None:
        for code in [
            SNPE_ERRORCODE_DLCACHING_SUBOPTIMAL_CACHE,
            SNPE_ERRORCODE_QNN_CONTEXT_ERROR_CREATE_FROM_BINARY,
            SNPE_ERRORCODE_QNN_COMMON_ERROR_NOT_SUPPORTED,
            "UNKNOWN",
        ]:
            s = get_cache_recovery_strategy(code)
            assert len(s["description"]) > 0
