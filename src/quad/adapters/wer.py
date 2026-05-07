"""Windows Error Reporting (WER) utilities for SNPE on Windows.

WER is automatic from SNPE SDK 2.28.0+:
  - No application code required
  - Application continues executing after report generation
  - Report SUBMISSION controlled by Windows OS privacy settings
  - WER GENERATION is always enabled in the SDK (cannot be disabled)

Critical errors that generate WER reports:
  - Hexagon DSP hardware accelerator RPC issues
    → SnpeHtpV##Stub.dll    (traditional path)
    → QnnHtpV##StubDrv.dll  (HNRD path)

Developer integration:
  Register with Windows Desktop Application Program to view telemetry.
  https://learn.microsoft.com/en-us/windows/win32/wer/using-wer
"""

from __future__ import annotations

from dataclasses import dataclass


# Minimum SNPE SDK version that generates WER reports
WER_MIN_SDK_VERSION = "2.28.0"

# Libraries involved in WER-reported errors
WER_INVOLVED_LIBRARIES_TRADITIONAL = [
    "SnpeHtpV{XX}Stub.dll",   # Traditional path — XX = hexagon version
]
WER_INVOLVED_LIBRARIES_HNRD = [
    "QnnHtpV{XX}StubDrv.dll", # HNRD path
]

# WER developer program URL
WER_DEVELOPER_PROGRAM_URL = (
    "https://learn.microsoft.com/en-us/windows/win32/wer/using-wer"
)


@dataclass
class WERStatus:
    """WER availability and configuration status for current SDK."""
    sdk_version: str
    wer_available: bool
    min_version: str = WER_MIN_SDK_VERSION
    description: str = ""

    @property
    def is_active(self) -> bool:
        """WER generation is active (always enabled when available)."""
        return self.wer_available

    @property
    def note(self) -> str:
        if self.wer_available:
            return (
                f"WER active (SNPE {self.sdk_version} >= {self.min_version}). "
                "Critical DSP RPC errors automatically reported. "
                "App continues executing after report generation. "
                "Submission controlled by Windows OS privacy settings."
            )
        return (
            f"WER not available (SNPE {self.sdk_version} < {self.min_version}). "
            f"Upgrade to {self.min_version}+ to enable automatic error reporting."
        )


def get_wer_status(sdk_version: str) -> WERStatus:
    """Check whether WER reporting is available for the given SDK version.

    Args:
        sdk_version: SNPE SDK version string, e.g. "2.45.0" or "2.28.0"

    Returns:
        WERStatus describing availability and guidance.
    """
    from quad.adapters.dlc_compat import parse_snpe_version
    v = parse_snpe_version(sdk_version)
    min_v = parse_snpe_version(WER_MIN_SDK_VERSION)
    available = v >= min_v

    return WERStatus(
        sdk_version=sdk_version,
        wer_available=available,
        min_version=WER_MIN_SDK_VERSION,
        description=(
            "Hexagon DSP hardware accelerator RPC issues are automatically "
            "reported via WER. Application continues executing."
        ) if available else "",
    )


def get_wer_library_files(
    hexagon_version: str,
    use_hnrd: bool = False,
) -> list[str]:
    """Return the library filenames that participate in WER error reporting.

    Args:
        hexagon_version: e.g. "v73"
        use_hnrd: True for HNRD path, False for traditional path

    Returns:
        List of DLL filenames involved in WER-reported errors.
    """
    ver_num = hexagon_version.lstrip("v")
    if use_hnrd:
        return [f"QnnHtpV{ver_num}StubDrv.dll"]
    return [f"SnpeHtpV{ver_num}Stub.dll"]
