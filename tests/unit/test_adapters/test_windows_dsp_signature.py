"""Tests for Windows DSP skel signature verification."""

from __future__ import annotations

import pytest

from quad.adapters.dsp_env import (
    DSP_SIGNATURE_ERROR_CODE,
    DSP_TRANSPORT_STATUS_FAILED,
    get_catalog_filename,
    get_catalog_sdk_path,
    get_skel_sdk_path,
    is_windows_signature_error,
    windows_dsp_deployment_check,
)


class TestCatalogFiles:
    def test_v73_catalog_filename(self) -> None:
        cat = get_catalog_filename("v73")
        assert cat == "libqnnhtpv73.cat"

    def test_v75_catalog_filename(self) -> None:
        cat = get_catalog_filename("v75")
        assert cat == "libqnnhtpv75.cat"

    def test_v79_catalog_filename(self) -> None:
        cat = get_catalog_filename("v79")
        assert cat == "libqnnhtpv79.cat"

    def test_v65_has_no_catalog(self) -> None:
        """v65/v66 DSP variants have no Windows catalog."""
        assert get_catalog_filename("v65") is None

    def test_v66_has_no_catalog(self) -> None:
        assert get_catalog_filename("v66") is None

    def test_catalog_all_lowercase(self) -> None:
        """Catalog name must be all lowercase."""
        cat = get_catalog_filename("v73")
        assert cat == cat.lower(), "Catalog filename must be all lowercase"

    def test_catalog_ends_with_dot_cat(self) -> None:
        for v in ("v68", "v69", "v73", "v75"):
            cat = get_catalog_filename(v)
            assert cat is not None and cat.endswith(".cat")

    def test_catalog_sdk_path(self) -> None:
        path = get_catalog_sdk_path("/opt/qairt", "v73")
        assert path == "/opt/qairt/lib/hexagon-v73/unsigned/libqnnhtpv73.cat"

    def test_catalog_sdk_path_v65_returns_none(self) -> None:
        assert get_catalog_sdk_path("/opt/qairt", "v65") is None


class TestCatalogSameFolder:
    """The .so and .cat MUST be in the same folder."""

    def test_same_folder_passes(self, tmp_path) -> None:
        skel = str(tmp_path / "libSnpeHtpV73Skel.so")
        cat = str(tmp_path / "libqnnhtpv73.cat")
        errors = windows_dsp_deployment_check(skel, cat)
        assert errors == []

    def test_different_folders_fails(self, tmp_path) -> None:
        folder_a = tmp_path / "a"
        folder_b = tmp_path / "b"
        folder_a.mkdir(); folder_b.mkdir()
        skel = str(folder_a / "libSnpeHtpV73Skel.so")
        cat = str(folder_b / "libqnnhtpv73.cat")
        errors = windows_dsp_deployment_check(skel, cat)
        assert len(errors) == 1
        assert "transportStatus: 9" in errors[0]
        assert "SAME folder" in errors[0]

    def test_different_folder_error_mentions_paths(self, tmp_path) -> None:
        skel = str(tmp_path / "skel_dir" / "libSnpeHtpV73Skel.so")
        cat = str(tmp_path / "cat_dir" / "libqnnhtpv73.cat")
        errors = windows_dsp_deployment_check(skel, cat)
        assert any("skel_dir" in e or "cat_dir" in e for e in errors)


class TestSignatureErrorDetection:
    def test_detects_transport_status_9(self) -> None:
        log = "QnnDsp <E> Unable to load Skel Library. transportStatus: 9"
        assert is_windows_signature_error(log) is True

    def test_detects_error_code(self) -> None:
        log = "QnnDsp <E> DspTransport.openSession qnn_open failed, 0x80000406"
        assert is_windows_signature_error(log) is True

    def test_detects_unable_to_load_skel(self) -> None:
        log = "QnnDsp <E> Unable to load Skel Library"
        assert is_windows_signature_error(log) is True

    def test_normal_output_not_flagged(self) -> None:
        log = "SNPE Version: 2.45.0\nRuntime: DSP\nInference: 5.2ms"
        assert is_windows_signature_error(log) is False

    def test_error_code_constant_correct(self) -> None:
        assert DSP_SIGNATURE_ERROR_CODE == "0x80000406"
        assert DSP_TRANSPORT_STATUS_FAILED == 9


class TestSkelAndCatalogCoLocated:
    """Ensure skel and catalog are always from the same SDK directory."""

    def test_skel_and_catalog_same_directory(self) -> None:
        """SDK paths for .so and .cat must share the same unsigned/ folder."""
        import os
        skel_path = get_skel_sdk_path("/opt/qairt", "v73")
        cat_path = get_catalog_sdk_path("/opt/qairt", "v73")
        assert os.path.dirname(skel_path) == os.path.dirname(cat_path), (
            "Skel .so and .cat must be in the same SDK directory"
        )

    def test_v75_skel_and_catalog_same_directory(self) -> None:
        import os
        skel = get_skel_sdk_path("/sdk", "v75")
        cat = get_catalog_sdk_path("/sdk", "v75")
        assert os.path.dirname(skel) == os.path.dirname(cat)
