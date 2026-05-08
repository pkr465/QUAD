"""Local hardware probing for `quad detect` / `list_devices()`.

Closes GAP_ANALYSIS T3.6: ``quad detect`` was returning hardcoded
device entries on every machine, regardless of what hardware was
actually present. This module provides per-OS probes that read real
hardware data:

* **Windows** — uses PowerShell ``Get-CimInstance`` and ``Get-PnpDevice``
  (same approach as ``examples/sample_app_real_hw.py`` which has been
  validated on a Snapdragon X Elite Dell Latitude 7455)
* **Linux** — reads ``/proc/cpuinfo`` and looks for Adreno / Hexagon
  via ``/sys/class/devfreq/*`` and ``lspci``
* **Android** — uses ``adb getprop`` (only when ``ANDROID_SERIAL`` is
  set; otherwise falls back to host probing since Android is rarely
  the QUAD host machine)
* **macOS** — sysctl + ioreg (limited; useful for dev work without
  Snapdragon)

Each probe returns a partial ``HostInfo`` dataclass; ``probe_host()``
combines all probes and falls back gracefully so the function never
raises — worst case it returns a record with ``source='unknown'`` and
empty fields, and ``list_devices()`` falls back to the legacy hardcoded
profiles.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─── Data class ─────────────────────────────────────────────────────────────


@dataclass
class HostInfo:
    """Best-effort snapshot of the local machine's compute hardware."""

    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_max_mhz: int = 0
    cpu_arch: str = ""

    gpu_name: str = ""
    gpu_driver: str = ""

    npu_name: str = ""
    npu_present: bool = False

    ram_gb: float = 0.0

    os_name: str = ""
    os_arch: str = ""

    source: str = "unknown"  # debug: which probe(s) populated this record

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_qualcomm(self) -> bool:
        """True if the CPU name contains 'Qualcomm' / 'Snapdragon' / 'Oryon'."""
        n = (self.cpu_name + " " + self.gpu_name + " " + self.npu_name).lower()
        return any(k in n for k in ("qualcomm", "snapdragon", "oryon", "kryo", "hexagon"))


# ─── Per-OS probes ───────────────────────────────────────────────────────────


def _run(cmd: list[str], *, timeout: float = 15.0) -> str:
    """Run a command; return stdout or '' on any failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _probe_windows() -> HostInfo | None:
    """PowerShell-based probe (Win32_Processor, Get-PnpDevice, Win32_VideoController)."""
    if os.name != "nt":
        return None
    info = HostInfo(source="windows-powershell")

    cpu_query = (
        "$c = Get-CimInstance Win32_Processor; "
        "Write-Output $c.Name; "
        "Write-Output $c.NumberOfCores; "
        "Write-Output $c.NumberOfLogicalProcessors; "
        "Write-Output $c.MaxClockSpeed"
    )
    out = _run(["powershell.exe", "-NoProfile", "-Command", cpu_query]).splitlines()
    if len(out) >= 4:
        info.cpu_name = out[0].strip()
        try:
            info.cpu_cores = int(out[1].strip())
            info.cpu_threads = int(out[2].strip())
            info.cpu_max_mhz = int(out[3].strip())
        except ValueError:
            pass

    gpu_query = (
        "$g = Get-CimInstance Win32_VideoController | Select-Object -First 1; "
        "Write-Output $g.Name; "
        "Write-Output $g.DriverVersion"
    )
    out = _run(["powershell.exe", "-NoProfile", "-Command", gpu_query]).splitlines()
    if out:
        info.gpu_name = out[0].strip()
        info.gpu_driver = out[1].strip() if len(out) > 1 else ""

    npu_query = (
        "$d = Get-PnpDevice -Status OK | Where-Object { "
        "$_.Class -eq 'ComputeAccelerator' -and "
        "$_.FriendlyName -match 'Hexagon|NPU|Snapdragon' } | Select-Object -First 1; "
        "if ($d) { Write-Output $d.FriendlyName }"
    )
    out = _run(["powershell.exe", "-NoProfile", "-Command", npu_query]).strip()
    if out:
        info.npu_name = out
        info.npu_present = True

    os_query = (
        "Write-Output (Get-CimInstance Win32_OperatingSystem).Caption; "
        "Write-Output ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture)"
    )
    out = _run(["powershell.exe", "-NoProfile", "-Command", os_query]).splitlines()
    if out:
        info.os_name = out[0].strip()
        info.os_arch = out[1].strip() if len(out) > 1 else ""

    info.cpu_arch = info.os_arch or platform.machine()

    # RAM via psutil if available, else WMI
    try:
        import psutil
        info.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        out = _run([
            "powershell.exe", "-NoProfile", "-Command",
            "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)",
        ]).strip()
        try:
            info.ram_gb = float(out)
        except ValueError:
            pass

    return info


def _probe_linux() -> HostInfo | None:
    """/proc/cpuinfo + lspci + sysfs probe."""
    if not sys.platform.startswith("linux"):
        return None
    info = HostInfo(source="linux-procfs")

    # CPU info
    try:
        cpuinfo = open("/proc/cpuinfo").read()
    except OSError:
        cpuinfo = ""
    if cpuinfo:
        # Some Qualcomm boards use 'Hardware', others 'model name'
        m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
        if m:
            info.cpu_name = m.group(1).strip()
        else:
            m = re.search(r"^Hardware\s*:\s*(.+)$", cpuinfo, re.M)
            if m:
                info.cpu_name = m.group(1).strip()
        info.cpu_threads = cpuinfo.count("processor\t:")
        info.cpu_cores = info.cpu_threads  # assume 1 thread per core unless SMT detected

    # GPU info via lspci or /sys
    lspci = _run(["lspci"])
    for line in lspci.splitlines():
        if "VGA" in line or "Display" in line or "3D" in line:
            info.gpu_name = line.split(":", 2)[-1].strip()
            break
    if not info.gpu_name:
        # Adreno on Qualcomm boards lives at /sys/class/devfreq/*adreno*
        try:
            devfreq = os.listdir("/sys/class/devfreq")
            for d in devfreq:
                if "adreno" in d.lower() or "kgsl" in d.lower():
                    info.gpu_name = f"Adreno (via {d})"
                    break
        except OSError:
            pass

    # NPU/DSP — Hexagon kernel module presence
    if os.path.exists("/sys/kernel/debug/cdsprm") or os.path.exists("/dev/cdsprpc-smd"):
        info.npu_name = "Hexagon DSP/HTP"
        info.npu_present = True

    # OS info
    try:
        os_release = open("/etc/os-release").read()
        m = re.search(r'^PRETTY_NAME\s*=\s*"?([^"\n]+)', os_release, re.M)
        info.os_name = m.group(1).strip() if m else "Linux"
    except OSError:
        info.os_name = "Linux"
    info.os_arch = platform.machine()
    info.cpu_arch = info.os_arch

    # RAM
    try:
        import psutil
        info.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            meminfo = open("/proc/meminfo").read()
            m = re.search(r"^MemTotal:\s*(\d+)\s*kB", meminfo, re.M)
            if m:
                info.ram_gb = round(int(m.group(1)) / (1024 * 1024), 1)
        except OSError:
            pass

    # Approx max MHz from /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq (kHz)
    try:
        khz = int(open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read().strip())
        info.cpu_max_mhz = khz // 1000
    except (OSError, ValueError):
        pass

    return info


def _probe_macos() -> HostInfo | None:
    """sysctl-based probe; limited NPU detection (Apple Silicon NE not exposed)."""
    if sys.platform != "darwin":
        return None
    info = HostInfo(source="macos-sysctl")

    info.cpu_name = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    try:
        info.cpu_cores = int(_run(["sysctl", "-n", "hw.physicalcpu"]).strip() or "0")
        info.cpu_threads = int(_run(["sysctl", "-n", "hw.logicalcpu"]).strip() or "0")
        khz = int(_run(["sysctl", "-n", "hw.cpufrequency_max"]).strip() or "0")
        info.cpu_max_mhz = khz // 1_000_000  # hw.cpufrequency is in Hz
    except ValueError:
        pass

    info.os_name = "macOS " + platform.mac_ver()[0]
    info.os_arch = platform.machine()
    info.cpu_arch = info.os_arch

    try:
        import psutil
        info.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            ram_bytes = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or "0")
            info.ram_gb = round(ram_bytes / (1024**3), 1)
        except ValueError:
            pass

    return info


def _probe_adb_target() -> HostInfo | None:
    """ADB-based probe of the connected Android device when ANDROID_SERIAL is set."""
    serial = os.environ.get("ANDROID_SERIAL", "").strip()
    if not serial:
        return None
    if not shutil.which("adb"):
        return None

    def _getprop(key: str) -> str:
        out = _run(["adb", "-s", serial, "shell", "getprop", key])
        return out.strip()

    info = HostInfo(source=f"adb:{serial}")
    info.cpu_name = _getprop("ro.soc.model") or _getprop("ro.hardware.chipname")
    info.cpu_arch = _getprop("ro.product.cpu.abi") or "aarch64"
    info.cpu_threads = sum(1 for _ in _getprop("ro.config.fingerprint").splitlines()) or 8
    info.os_name = "Android " + _getprop("ro.build.version.release")
    info.os_arch = info.cpu_arch
    info.npu_name = _getprop("ro.vendor.qti.cdsprm.npu") or "Hexagon (assumed)"
    info.npu_present = bool(info.cpu_name)
    return info


# ─── Combined probe ──────────────────────────────────────────────────────────


def probe_host(prefer_adb: bool = False) -> HostInfo:
    """Return a best-effort hardware probe of the local machine.

    Args:
        prefer_adb: if True and ``ANDROID_SERIAL`` is set, return the
            ADB target's info instead of the host machine. Otherwise
            ADB is only tried as a fallback.

    Always returns a HostInfo (never raises). The .source field tells
    you which probe(s) populated it; check ``info.cpu_name == ""`` to
    detect a complete probe failure.
    """
    if prefer_adb:
        adb = _probe_adb_target()
        if adb and adb.cpu_name:
            return adb

    for probe in (_probe_windows, _probe_linux, _probe_macos):
        try:
            result = probe()
            if result is not None and (result.cpu_name or result.cpu_cores):
                return result
        except Exception as e:
            logger.debug("host probe failed: %s", e)

    if not prefer_adb:
        adb = _probe_adb_target()
        if adb:
            return adb

    # Total fallback — at least populate platform info
    return HostInfo(
        os_name=platform.platform(),
        os_arch=platform.machine(),
        cpu_arch=platform.machine(),
        source="fallback",
    )


# ─── Bridge to Device profiles ──────────────────────────────────────────────


def hostinfo_to_device_profiles(info: HostInfo) -> dict[str, dict[str, Any]]:
    """Convert a HostInfo into the Device-properties dict format.

    Returns a dict with up to 3 keys ('cpu', 'gpu', 'npu') populated
    only for the compute units that were actually detected. Falls
    back to the legacy hardcoded profiles for missing pieces — that
    way ``list_devices()`` always returns at least a CPU entry.
    """
    profiles: dict[str, dict[str, Any]] = {}

    if info.cpu_name or info.cpu_cores:
        profiles["cpu"] = {
            "name": info.cpu_name or "CPU",
            "type": "cpu",
            "cores": info.cpu_cores or info.cpu_threads or 1,
            "freq_ghz": round(info.cpu_max_mhz / 1000.0, 2) if info.cpu_max_mhz else 0.0,
            "memory_mb": int(info.ram_gb * 1024) if info.ram_gb else 0,
            "power_typical_mw": 5000,  # ARM-class default
        }

    if info.gpu_name:
        profiles["gpu"] = {
            "name": info.gpu_name,
            "type": "gpu",
            "tflops": 0.0,  # not exposed by Win32 / sysfs; would need vendor-specific query
            "cores": 0,
            "memory_mb": int(info.ram_gb * 1024) if info.ram_gb else 0,  # often shared
            "power_typical_mw": 3500,
        }

    if info.npu_present:
        profiles["npu"] = {
            "name": info.npu_name or "Hexagon NPU",
            "type": "npu",
            "tops": 45.0,  # nominal X Elite NPU; would need vendor-specific query for real value
            "cores": 4,
            "memory_mb": min(int(info.ram_gb * 1024) // 4, 8192) if info.ram_gb else 8192,
            "power_typical_mw": 2000,
        }

    return profiles
