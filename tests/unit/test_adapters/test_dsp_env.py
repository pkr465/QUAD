"""Tests for DSP runtime environment utilities."""

from __future__ import annotations

import pytest

from quad.adapters.dsp_env import (
    build_adsp_library_path,
    get_hexagon_version_for_chipset,
    get_skel_info,
    get_skel_sdk_path,
)


class TestGetSkelInfo:
    def test_v65_uses_dsp_prefix(self) -> None:
        hexagon_dir, skel = get_skel_info("v65")
        assert hexagon_dir == "hexagon-v65"
        assert skel == "libSnpeDspV65Skel.so"
        assert "Dsp" in skel

    def test_v66_uses_dsp_prefix(self) -> None:
        hexagon_dir, skel = get_skel_info("v66")
        assert "Dsp" in skel
        assert skel == "libSnpeDspV66Skel.so"

    def test_v68_uses_htp_prefix(self) -> None:
        hexagon_dir, skel = get_skel_info("v68")
        assert hexagon_dir == "hexagon-v68"
        assert skel == "libSnpeHtpV68Skel.so"
        assert "Htp" in skel  # NOT "Dsp"

    def test_v69_uses_htp_prefix(self) -> None:
        _, skel = get_skel_info("v69")
        assert "Htp" in skel
        assert skel == "libSnpeHtpV69Skel.so"

    def test_v73_uses_htp_prefix(self) -> None:
        _, skel = get_skel_info("v73")
        assert skel == "libSnpeHtpV73Skel.so"

    def test_v75_uses_htp_prefix(self) -> None:
        _, skel = get_skel_info("v75")
        assert "Htp" in skel

    def test_accepts_full_string(self) -> None:
        """Accept "hexagon-v68" as well as "v68"."""
        hexagon_dir, skel = get_skel_info("hexagon-v73")
        assert skel == "libSnpeHtpV73Skel.so"

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Hexagon"):
            get_skel_info("v99")

    def test_v65_v66_never_have_htp(self) -> None:
        for v in ("v65", "v66"):
            _, skel = get_skel_info(v)
            assert "Htp" not in skel, f"{v} should not use Htp prefix"

    def test_v68_and_later_never_have_dsp(self) -> None:
        for v in ("v68", "v69", "v73", "v75"):
            _, skel = get_skel_info(v)
            assert "Dsp" not in skel, f"{v} should not use Dsp prefix"


class TestChipsetLookup:
    def test_snapdragon_8_elite(self) -> None:
        v = get_hexagon_version_for_chipset("Snapdragon 8 Elite")
        assert v == "v79"

    def test_snapdragon_x_elite(self) -> None:
        v = get_hexagon_version_for_chipset("Snapdragon X Elite")
        assert v == "v75"

    def test_snapdragon_8_gen_2(self) -> None:
        v = get_hexagon_version_for_chipset("SM8550")
        assert v == "v73"

    def test_qcs2210_arduino_unoq(self) -> None:
        v = get_hexagon_version_for_chipset("QCS2210")
        assert v == "v66"

    def test_unknown_returns_none(self) -> None:
        v = get_hexagon_version_for_chipset("NonExistentChip")
        assert v is None


class TestAdspLibraryPath:
    def test_android_has_three_mandatory_paths(self) -> None:
        path = build_adsp_library_path("/data/local/tmp/snpe/dsp", "android")
        assert "/system/lib/rfsa/adsp" in path
        assert "/system/vendor/lib/rfsa/adsp" in path
        assert "/dsp" in path

    def test_uses_semicolons_not_colons(self) -> None:
        path = build_adsp_library_path("/data/local/tmp/snpe/dsp", "android")
        assert ";" in path
        assert ":" not in path, "ADSP_LIBRARY_PATH must use semicolons, not colons"

    def test_skel_dir_is_first(self) -> None:
        skel_dir = "/data/local/tmp/inception/dsp"
        path = build_adsp_library_path(skel_dir, "android")
        assert path.startswith(skel_dir), "Skel dir must be first in path"

    def test_automotive_uses_different_paths(self) -> None:
        path = build_adsp_library_path("/data/dsp", "automotive")
        assert "/usr/lib/rfsa/adsp" in path
        assert "/system/lib/rfsa/adsp" not in path

    def test_linux_same_as_android(self) -> None:
        android_path = build_adsp_library_path("/dsp", "android")
        linux_path = build_adsp_library_path("/dsp", "linux")
        assert android_path == linux_path


class TestSkelSdkPath:
    def test_v68_path_structure(self) -> None:
        path = get_skel_sdk_path("/opt/qairt", "v68")
        assert path == "/opt/qairt/lib/hexagon-v68/unsigned/libSnpeHtpV68Skel.so"

    def test_v66_path_structure(self) -> None:
        path = get_skel_sdk_path("/opt/qairt", "v66")
        assert path == "/opt/qairt/lib/hexagon-v66/unsigned/libSnpeDspV66Skel.so"

    def test_v73_path_structure(self) -> None:
        path = get_skel_sdk_path("/opt/snpe", "v73")
        assert "hexagon-v73" in path
        assert "libSnpeHtpV73Skel.so" in path
