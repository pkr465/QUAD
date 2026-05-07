"""DSP runtime environment utilities — ADSP_LIBRARY_PATH, skel selection, Windows signatures.

Documents the rules from the SNPE DSP Runtime Environment guides:

General (all platforms):
- ADSP_LIBRARY_PATH uses semicolons (not colons) and must be quoted
- v65/v66: libSnpeDspV{XX}Skel.so  (Dsp prefix)
- v68+:    libSnpeHtpV{XX}Skel.so  (Htp prefix)
- Three mandatory paths: /system/lib/rfsa/adsp, /system/vendor/lib/rfsa/adsp, /dsp
- Automotive Linux: /usr/lib/rfsa/adsp and /dsp

Windows DSP Skel Signature Verification (Snapdragon X Elite+):
- Every skel .so requires a matching .cat (Windows security catalog) file
- The .so and .cat MUST be in the SAME FOLDER or the library will NOT load
- Never modify .so or .cat files — this breaks signature verification
- Error when verification fails: transportStatus: 9 (0x80000406)
- The "unsigned/" folder name refers to the Protection Domain, NOT the
  Windows digital signature status (confusing naming — both signed and
  unsigned PD skel files have the Windows catalog signature)
- Windows device drivers: use HNRD backend, NOT skel libraries directly
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Hexagon version to skel library mapping ──────────────────────────────────
#
# v65, v66 → DSP prefix (libSnpeDspV{XX}Skel.so)
# v68+     → HTP prefix (libSnpeHtpV{XX}Skel.so)

_SKEL_MAP: dict[str, tuple[str, str]] = {
    "v65": ("hexagon-v65", "libSnpeDspV65Skel.so"),
    "v66": ("hexagon-v66", "libSnpeDspV66Skel.so"),
    "v68": ("hexagon-v68", "libSnpeHtpV68Skel.so"),
    "v69": ("hexagon-v69", "libSnpeHtpV69Skel.so"),
    "v73": ("hexagon-v73", "libSnpeHtpV73Skel.so"),
    "v75": ("hexagon-v75", "libSnpeHtpV75Skel.so"),
    "v79": ("hexagon-v79", "libSnpeHtpV79Skel.so"),
    "v81": ("hexagon-v81", "libSnpeHtpV81Skel.so"),
}

# Chipset → Hexagon version mapping (common Snapdragon devices)
_CHIPSET_HEXAGON_MAP: dict[str, str] = {
    # Snapdragon 8 Elite (SM8750) — v79/HTP
    "sm8750": "v79",
    "snapdragon 8 elite": "v79",
    # Snapdragon X Elite
    "x1e-80-100": "v75",
    "snapdragon x elite": "v75",
    # Snapdragon 8 Gen 3 (SM8650)
    "sm8650": "v75",
    "snapdragon 8 gen 3": "v75",
    # Snapdragon 8 Gen 2 (SM8550)
    "sm8550": "v73",
    "snapdragon 8 gen 2": "v73",
    # Snapdragon 8 Gen 1 (SM8450)
    "sm8450": "v69",
    "snapdragon 8 gen 1": "v69",
    # Snapdragon 888 (SM8350)
    "sm8350": "v68",
    "snapdragon 888": "v68",
    # QCS2210 (Arduino UNO Q)
    "qcs2210": "v66",
    # QCS6490
    "qcs6490": "v69",
}

# Mandatory ADSP paths per target type
_ADSP_MANDATORY_ANDROID = [
    "/system/lib/rfsa/adsp",
    "/system/vendor/lib/rfsa/adsp",
    "/dsp",
]

_ADSP_MANDATORY_AUTOMOTIVE_LINUX = [
    "/usr/lib/rfsa/adsp",
    "/dsp",
]


def get_skel_info(hexagon_version: str) -> tuple[str, str]:
    """Return (hexagon_dir, skel_filename) for a Hexagon version string.

    Args:
        hexagon_version: e.g. "v68", "v73" or full string "hexagon-v73"

    Returns:
        Tuple of (hexagon_dir, skel_filename)

    Raises:
        ValueError if version is not recognised.
    """
    version = hexagon_version.replace("hexagon-", "").lower()
    if version not in _SKEL_MAP:
        raise ValueError(
            f"Unknown Hexagon version '{hexagon_version}'. "
            f"Known: {list(_SKEL_MAP.keys())}"
        )
    return _SKEL_MAP[version]


def get_hexagon_version_for_chipset(chipset: str) -> str | None:
    """Look up the Hexagon DSP version for a known chipset name.

    Args:
        chipset: Chipset name (case-insensitive partial match)

    Returns:
        Hexagon version string (e.g. "v73") or None if unknown.
    """
    chipset_lower = chipset.lower()
    for key, version in _CHIPSET_HEXAGON_MAP.items():
        if key in chipset_lower or chipset_lower in key:
            return version
    return None


def build_adsp_library_path(
    skel_dir: str,
    target_type: str = "android",
) -> str:
    """Build the ADSP_LIBRARY_PATH value.

    Rules from SNPE DSP Runtime docs:
    - Use semicolons (NOT colons) as separator
    - Must be quoted when setting with export
    - Mandatory system paths must be included

    Args:
        skel_dir: Directory where skel .so files are deployed
        target_type: "android", "linux", or "automotive"

    Returns:
        Semicolon-separated path string (ready for setenv / export)
    """
    paths = [skel_dir]

    if target_type in ("android", "linux"):
        paths.extend(_ADSP_MANDATORY_ANDROID)
    elif target_type == "automotive":
        paths.extend(_ADSP_MANDATORY_AUTOMOTIVE_LINUX)
    else:
        # Default: include Android paths as safe fallback
        paths.extend(_ADSP_MANDATORY_ANDROID)

    return ";".join(paths)


def get_windows_arch_folder(app_arch: str) -> str:
    """Return the SNPE SDK library folder for a Windows app architecture.

    Implements the ARM64X two-scenario rule:
      ARM64/ARM64EC/ARM64X apps → arm64x-windows-msvc (scenario 1)
      x86_64 apps               → x86_64-windows-msvc (scenario 2)
                                  (uses x86_64 SNPE.lib; ARM64X SNPE.dll at runtime)

    Args:
        app_arch: "arm64x", "arm64ec", "arm64", "aarch64", "x64", "x86_64"

    Returns:
        SDK folder name, e.g. "arm64x-windows-msvc"
    """
    _folder_map = {
        "arm64x":  "arm64x-windows-msvc",    # SC8380XP native
        "arm64ec": "aarch64-windows-msvc",   # ARM64EC uses ARM64 libs
        "arm64":   "aarch64-windows-msvc",
        "aarch64": "aarch64-windows-msvc",
        "x64":     "x86_64-windows-msvc",
        "x86_64":  "x86_64-windows-msvc",
        "amd64":   "x86_64-windows-msvc",
    }
    return _folder_map.get(app_arch.lower(), "x86_64-windows-msvc")


def get_skel_sdk_path(sdk_root: str, hexagon_version: str) -> str:
    """Return the full path to the skel .so in the SDK.

    Args:
        sdk_root: QAIRT_SDK_ROOT / SNPE_ROOT path
        hexagon_version: e.g. "v68", "v73"

    Returns:
        Full path: {sdk_root}/lib/hexagon-v68/unsigned/libSnpeHtpV68Skel.so
    """
    hexagon_dir, skel_name = get_skel_info(hexagon_version)
    return f"{sdk_root}/lib/{hexagon_dir}/unsigned/{skel_name}"


# ══════════════════════════════════════════════════════════════════════════════
# Windows DSP Skel Signature Verification
# ══════════════════════════════════════════════════════════════════════════════
#
# On Windows (Snapdragon X Elite+), each skel .so requires a Windows security
# catalog (.cat) file. The .so and .cat MUST be deployed to the SAME directory.
#
# Catalog filename convention: libqnnhtpvXX.cat  (all lowercase, no "Skel")
#   v73 → libqnnhtpv73.cat
#   v75 → libqnnhtpv75.cat
#
# Error signature: transportStatus: 9 / 0x80000406 = verification failed

# Windows signature error codes
DSP_SIGNATURE_ERROR_CODE = "0x80000406"
DSP_TRANSPORT_STATUS_FAILED = 9


def get_catalog_filename(hexagon_version: str) -> str | None:
    """Return the Windows .cat catalog filename for a given Hexagon version.

    Catalog files are required on Windows (Snapdragon X Elite+) for DSP
    skel signature verification. The .so and .cat MUST be in the SAME folder.

    Only applicable for HTP versions (v68+). v65/v66 DSP variants
    do not have Windows catalog files.

    Args:
        hexagon_version: e.g. "v73", "v75", "hexagon-v73"

    Returns:
        Catalog filename (e.g. "libqnnhtpv73.cat") or None for v65/v66.
    """
    version = hexagon_version.replace("hexagon-", "").lower()
    version_num = version.lstrip("v")
    if version in ("v65", "v66"):
        return None  # No Windows catalog for legacy DSP versions
    # Pattern: libqnnhtpvXX.cat (all lowercase)
    return f"libqnnhtpv{version_num}.cat"


def get_catalog_sdk_path(sdk_root: str, hexagon_version: str) -> str | None:
    """Return the full path to the Windows .cat file in the SDK.

    Returns None for non-Windows HTP versions (v65/v66).

    Args:
        sdk_root: QAIRT_SDK_ROOT / SNPE_ROOT path
        hexagon_version: e.g. "v73"

    Returns:
        Full path: {sdk_root}/lib/hexagon-v73/unsigned/libqnnhtpv73.cat
        or None if not applicable.
    """
    cat_name = get_catalog_filename(hexagon_version)
    if cat_name is None:
        return None
    hexagon_dir, _ = get_skel_info(hexagon_version)
    return f"{sdk_root}/lib/{hexagon_dir}/unsigned/{cat_name}"


def is_windows_signature_error(error_output: str) -> bool:
    """Check if an error string indicates Windows DSP signature failure.

    Args:
        error_output: stderr/log output from QNN/SNPE

    Returns:
        True if the error is a Windows DSP signature verification failure.
    """
    return (
        "transportStatus: 9" in error_output
        or DSP_SIGNATURE_ERROR_CODE in error_output
        or "Unable to load Skel Library" in error_output
    )


def windows_dsp_deployment_check(skel_path: str, catalog_path: str) -> list[str]:
    """Validate that skel .so and .cat are in the same directory.

    Returns a list of error messages (empty = valid).

    Args:
        skel_path: Path where the skel .so was deployed
        catalog_path: Path where the .cat file was deployed
    """
    import os
    errors = []
    skel_dir = os.path.dirname(os.path.abspath(skel_path))
    cat_dir = os.path.dirname(os.path.abspath(catalog_path))
    if skel_dir != cat_dir:
        errors.append(
            f"CRITICAL: skel .so and .cat must be in the SAME folder. "
            f"skel dir: {skel_dir}, cat dir: {cat_dir}. "
            f"This will cause transportStatus: 9 at runtime."
        )
    return errors


# ══════════════════════════════════════════════════════════════════════════════
# Protection Domain (PD) Types
# ══════════════════════════════════════════════════════════════════════════════
#
# SNPE supports two Protection Domain types for DSP execution:
#
# Unsigned PD (default in SNPE2):
#   - Standard execution without signed skel libraries
#   - The "unsigned/" folder in SDK contains unsigned PD skel files
#   - Faster to set up, no signing required
#   - Platform option: unsignedPD:ON (or omit — it is the default)
#
# Signed PD:
#   - Requires customer-signed skel libraries
#   - Higher security: only authorized code runs on Hexagon NPU
#   - Platform option: unsignedPD:OFF
#   - SNPE2 skel files are NOT signed — customer must sign them
#
# Confusing naming: "unsigned/" folder = unsigned Protection Domain
#   This is NOT about the Windows digital signature (.cat files).
#   Both unsigned-PD and signed-PD skel files can have Windows .cat signatures.
#
# RuntimeCheckOption for isRuntimeAvailable():
#   UNSIGNEDPD_CHECK → True for Unsigned PD, False for Signed PD
#   NORMAL_CHECK     → True for Signed PD,   False for Unsigned PD
#   BASIC_CHECK      → same as NORMAL_CHECK


class PDType:
    """Protection Domain type constants."""
    UNSIGNED = "unsigned"   # Default in SNPE2: unsignedPD:ON
    SIGNED = "signed"       # Requires signed skels: unsignedPD:OFF


class RuntimeCheckOption:
    """Runtime availability check options (maps to RuntimeCheckOption_t)."""
    UNSIGNEDPD_CHECK = "UNSIGNEDPD_CHECK"   # Pass = Unsigned PD in use
    NORMAL_CHECK = "NORMAL_CHECK"            # Pass = Signed PD in use
    BASIC_CHECK = "BASIC_CHECK"              # Same as NORMAL_CHECK


# Truth table for isRuntimeAvailable() per the SDK documentation
_RUNTIME_CHECK_MATRIX: dict[str, dict[str, bool]] = {
    #                            UNSIGNEDPD  NORMAL   BASIC
    PDType.UNSIGNED: {
        RuntimeCheckOption.UNSIGNEDPD_CHECK: True,
        RuntimeCheckOption.NORMAL_CHECK: False,
        RuntimeCheckOption.BASIC_CHECK: False,
    },
    PDType.SIGNED: {
        RuntimeCheckOption.UNSIGNEDPD_CHECK: False,
        RuntimeCheckOption.NORMAL_CHECK: True,
        RuntimeCheckOption.BASIC_CHECK: True,
    },
}


def get_platform_option(pd_type: str) -> str:
    """Return the platform options string for the given PD type.

    Returns the string for SNPEBuilder.setPlatformOptions() or
    snpe-net-run --platform_options=...

    Args:
        pd_type: PDType.UNSIGNED (default) or PDType.SIGNED

    Returns:
        "unsignedPD:OFF" for Signed PD, "unsignedPD:ON" for Unsigned PD.
    """
    if pd_type == PDType.SIGNED:
        return "unsignedPD:OFF"
    return "unsignedPD:ON"


def check_runtime_available(
    pd_type: str,
    check_option: str = RuntimeCheckOption.UNSIGNEDPD_CHECK,
) -> bool:
    """Simulate isRuntimeAvailable() result for a given PD type and check option.

    SDK truth table (from Signed PD and Unsigned PD at Runtime docs):
        PD Type    | UNSIGNEDPD_CHECK | NORMAL_CHECK | BASIC_CHECK
        -----------|------------------|--------------|------------
        Unsigned   | Pass (True)      | Fail (False) | Fail (False)
        Signed     | Fail (False)     | Pass (True)  | Pass (True)
    """
    return _RUNTIME_CHECK_MATRIX.get(pd_type, {}).get(check_option, False)
