"""Tests for Windows Error Reporting (WER) utilities."""

from __future__ import annotations

import pytest

from quad.adapters.wer import (
    WER_MIN_SDK_VERSION,
    WERStatus,
    get_wer_library_files,
    get_wer_status,
)


class TestWERMinVersion:
    def test_min_version_is_2_28(self) -> None:
        assert WER_MIN_SDK_VERSION == "2.28.0"


class TestGetWERStatus:
    def test_sdk_2_28_has_wer(self) -> None:
        status = get_wer_status("2.28.0")
        assert status.wer_available is True
        assert status.is_active is True

    def test_sdk_2_45_has_wer(self) -> None:
        status = get_wer_status("2.45.0")
        assert status.wer_available is True

    def test_sdk_before_2_28_no_wer(self) -> None:
        status = get_wer_status("2.22.2")
        assert status.wer_available is False
        assert status.is_active is False

    def test_sdk_1_x_no_wer(self) -> None:
        assert get_wer_status("1.15.0").wer_available is False

    def test_note_mentions_continuation(self) -> None:
        """App continues executing after WER report generation."""
        status = get_wer_status("2.28.0")
        assert "continues" in status.note.lower() or "executing" in status.note.lower()

    def test_note_mentions_privacy_settings(self) -> None:
        """Submission is controlled by Windows OS privacy settings."""
        status = get_wer_status("2.28.0")
        assert "privacy" in status.note.lower() or "submission" in status.note.lower()

    def test_unavailable_note_suggests_upgrade(self) -> None:
        status = get_wer_status("2.22.0")
        assert "2.28.0" in status.note or "upgrade" in status.note.lower()


class TestWERLibraryFiles:
    def test_traditional_path_stub_dll(self) -> None:
        files = get_wer_library_files("v73", use_hnrd=False)
        assert "SnpeHtpV73Stub.dll" in files

    def test_hnrd_path_drv_dll(self) -> None:
        files = get_wer_library_files("v73", use_hnrd=True)
        assert "QnnHtpV73StubDrv.dll" in files

    def test_traditional_not_hnrd(self) -> None:
        trad = get_wer_library_files("v73", use_hnrd=False)
        hnrd = get_wer_library_files("v73", use_hnrd=True)
        # No overlap — different libraries for each path
        assert set(trad).isdisjoint(set(hnrd))

    def test_different_hexagon_versions(self) -> None:
        for v in ("v68", "v73", "v75"):
            files = get_wer_library_files(v, use_hnrd=False)
            ver_num = v.lstrip("v")
            assert f"SnpeHtpV{ver_num}Stub.dll" in files
