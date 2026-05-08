"""Tests for the local-host hardware probe (T3.6)."""

from __future__ import annotations

import sys

import pytest

from quad.runtime.host_probe import (
    HostInfo,
    hostinfo_to_device_profiles,
    probe_host,
)


class TestHostInfo:
    def test_is_qualcomm_detection(self) -> None:
        qc = HostInfo(cpu_name="Snapdragon X Elite X1E80100 - Qualcomm Oryon CPU")
        assert qc.is_qualcomm is True

        non_qc = HostInfo(cpu_name="Intel(R) Core(TM) i7-9750H")
        assert non_qc.is_qualcomm is False

    def test_is_qualcomm_when_npu_qualcomm(self) -> None:
        # Some Linux boards report a generic CPU name but Hexagon NPU
        info = HostInfo(cpu_name="ARMv8 Cortex-A78", npu_name="Hexagon DSP/HTP")
        assert info.is_qualcomm is True

    def test_to_dict_serialisable(self) -> None:
        info = HostInfo(cpu_name="x", cpu_cores=8, source="test")
        d = info.to_dict()
        assert d["cpu_name"] == "x"
        assert d["cpu_cores"] == 8


class TestProbeHostNeverRaises:
    """The probe must always return a HostInfo, never raise."""

    def test_returns_hostinfo(self) -> None:
        info = probe_host()
        assert isinstance(info, HostInfo)
        # We're running on *something* — at minimum os_arch should be set
        assert info.os_arch != ""

    def test_source_is_set(self) -> None:
        info = probe_host()
        assert info.source != ""


class TestProbeWindows:
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only probe")
    def test_windows_probe_finds_cpu(self) -> None:
        info = probe_host()
        # On the test laptop we know there's a Snapdragon X Elite
        # but on CI Windows runners it's an x64 server. Either way we
        # should have cpu_name + cpu_cores from the Win32 probe.
        assert info.cpu_name != ""
        assert info.cpu_cores > 0
        assert info.source.startswith("windows")


class TestProbeLinux:
    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only probe")
    def test_linux_probe_finds_cpu_threads(self) -> None:
        info = probe_host()
        assert info.cpu_threads > 0


class TestHostinfoToDeviceProfiles:
    def test_full_qualcomm_setup(self) -> None:
        info = HostInfo(
            cpu_name="Snapdragon X Elite",
            cpu_cores=12,
            cpu_max_mhz=4012,
            ram_gb=32.0,
            gpu_name="Adreno X1-85",
            npu_name="Hexagon NPU",
            npu_present=True,
        )
        profiles = hostinfo_to_device_profiles(info)
        assert "cpu" in profiles
        assert "gpu" in profiles
        assert "npu" in profiles
        assert profiles["cpu"]["name"] == "Snapdragon X Elite"
        assert profiles["cpu"]["cores"] == 12
        assert profiles["gpu"]["name"] == "Adreno X1-85"
        assert profiles["npu"]["name"] == "Hexagon NPU"

    def test_cpu_only_machine(self) -> None:
        info = HostInfo(cpu_name="Intel Core i7", cpu_cores=8)
        profiles = hostinfo_to_device_profiles(info)
        assert "cpu" in profiles
        assert "gpu" not in profiles
        assert "npu" not in profiles

    def test_empty_info_returns_empty(self) -> None:
        # No detected hardware → empty profiles dict; list_devices() then
        # falls back to the legacy hardcoded values.
        profiles = hostinfo_to_device_profiles(HostInfo())
        assert profiles == {}
