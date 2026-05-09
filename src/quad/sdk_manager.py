"""QAIRT/SNPE SDK auto-discovery and install helper.

Used by the MCP server's startup hook (and the ``quad sdk`` CLI) to
locate or install a Qualcomm AI Runtime SDK without forcing the user to
fiddle with environment variables.

Resolution order (first match wins):
    1. ``QAIRT_SDK_ROOT`` env var (also ``QNN_SDK_ROOT``, ``SNPE_ROOT``)
    2. ``server.qairt_sdk_root`` from ``quad.toml``
    3. ``./sdks/qairt-*`` (or ``snpe-*``) under the project root
    4. ``~/.quad/sdks/qairt-*``
    5. Vendor default install paths (``C:\\Qualcomm\\AIStack\\QAIRT\\*``,
       ``/opt/qcom/aistack/qairt/*``, ``/opt/qairt/*``)

The Qualcomm download pages are JS-rendered SPAs and the actual archives
sit behind a developer-account login + EULA acceptance — there is no
unauthenticated direct-download URL. Instead of pretending to
auto-download, the manager:

* Auto-discovers an existing install (the common case for repeat runs)
* Provides ``install_archive(zip_path)`` to unpack a user-downloaded
  archive into ``./sdks/`` in one step
* Surfaces a clear missing-SDK message with the exact URLs the user
  should visit:

    https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
    https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai

Setting ``QUAD_SDK_AUTO_DOWNLOAD=1`` enables an experimental download
path (``download_archive``) for environments that have a pre-stored
``QUALCOMM_SOFTWARE_CENTER_TOKEN`` cookie — off by default.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import sys
import tarfile
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

# Re-exposed for tests so callers can verify host detection behaviour
# without poking at internals.
__all__ = (
    "SDKInfo",
    "InstallResult",
    "discover_sdks",
    "resolve_sdk_root",
    "install_archive",
    "apply_to_environment",
    "startup_resolve_and_log",
    "rank_bin_subdir",
    "host_arch_label",
)

logger = logging.getLogger(__name__)

# ── URLs and constants ────────────────────────────────────────────────────────

QAIRT_PRODUCT_URL = (
    "https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk"
)
SNPE_PRODUCT_URL = (
    "https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai"
)

# Recent known-good versions (updated as SDK releases occur). Used only
# for *informational* logging — the actual SDK is whatever the user
# installed.
KNOWN_VERSIONS = (
    "2.45.0.260326",
    "2.43.0.250827",
    "2.41.0.250714",
    "2.39.0.250513",
)

# Locations to scan for an installed SDK
DEFAULT_SCAN_PATHS = (
    # Project-local (preferred — keeps everything self-contained)
    "./sdks",
    # User cache
    "~/.quad/sdks",
    # Windows vendor defaults
    "C:/Qualcomm/AIStack/QAIRT",
    "C:/Qualcomm/AIStack/SNPE",
    # Linux/macOS vendor defaults
    "/opt/qcom/aistack/qairt",
    "/opt/qcom/aistack/snpe",
    "/opt/qairt",
    "/opt/snpe",
)

# Directory pattern. Matches:
#   qairt-2.45.0.260326   (canonical project layout)
#   snpe-2.45.0           (legacy)
#   qairt_2.45            (underscore variant)
#   v2.46.0.260424        (Qualcomm developer-portal naming)
#   2.46.0.260424         (bare version, e.g. inner dir after archive extract)
_SDK_DIR_RE = re.compile(
    r"^(?:(qairt|snpe)[-_ ]?)?v?(\d+\.\d+(?:\.\d+(?:\.\d+)?)?)$",
    re.I,
)


@dataclass
class SDKInfo:
    """Description of a discovered SDK install."""

    root: str
    version: str
    flavor: str  # "qairt" or "snpe"
    source: str  # how it was found (env / config / scan / manual)
    bin_dir: str = ""  # "<root>/bin/<arch>" with the CLI tools
    has_qairt_converter: bool = False
    has_snpe_net_run: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


# ── Discovery ────────────────────────────────────────────────────────────────


def _has_qairt_converter(sub: Path) -> bool:
    return (sub / "qairt-converter").exists() or (sub / "qairt-converter.exe").exists()


def _has_snpe_net_run(sub: Path) -> bool:
    return (sub / "snpe-net-run").exists() or (sub / "snpe-net-run.exe").exists()


def _has_any_tool(sub: Path) -> bool:
    return _has_qairt_converter(sub) or _has_snpe_net_run(sub)


def host_arch_label() -> str:
    """Best-effort host arch tag for picking a per-arch SDK bin subdir.

    On Windows ARM64 a Microsoft Store Python is x86_64 (running under
    Prism), so ``PROCESSOR_ARCHITECTURE`` says AMD64 even though the OS
    itself is ARM64. ``platform.machine()`` reads the OS-reported arch
    via the registry and gives the correct answer in that case, so we
    prefer it.
    """
    machine = (platform.machine() or "").upper()
    if machine in ("ARM64", "AARCH64"):
        return "arm64"
    if machine in ("AMD64", "X86_64"):
        return "x86_64"
    if machine in ("X86", "I386", "I686"):
        return "x86"
    return machine.lower() or "unknown"


# Per-host preference: rank a bin subdir from 0 (avoid) to higher (better).
# Native arch beats emulated arch beats nothing.
_BIN_RANK_BY_NAME = {
    "win32": {
        "arm64": ("aarch64-windows-msvc", "arm64x-windows-msvc", "x86_64-windows-msvc"),
        "x86_64": ("x86_64-windows-msvc", "arm64x-windows-msvc", "aarch64-windows-msvc"),
    },
    "linux": {
        "arm64": (
            "aarch64-ubuntu-gcc9.4",
            "aarch64-oe-linux-gcc11.2",
            "aarch64-oe-linux-gcc9.3",
            "aarch64-oe-linux-gcc8.2",
            "x86_64-linux-clang",
        ),
        "x86_64": ("x86_64-linux-clang",),
    },
    "darwin": {
        "arm64": ("aarch64-ubuntu-gcc9.4", "x86_64-linux-clang"),
        "x86_64": ("x86_64-linux-clang",),
    },
}


def rank_bin_subdir(name: str, *, host_platform: str | None = None,
                    host_arch: str | None = None) -> int:
    """Rank a bin subdir by host preference.

    Higher = better. Returns 0 for unrecognised names.
    """
    host_platform = host_platform or sys.platform
    host_arch = host_arch or host_arch_label()
    table = _BIN_RANK_BY_NAME.get(host_platform, {}).get(host_arch, ())
    if name in table:
        # Prefer earlier entries
        return len(table) - table.index(name)
    return 0


def _looks_like_sdk_root(path: Path) -> str | None:
    """Return SDK flavor ('qairt'|'snpe') if ``path`` resembles an SDK root.

    Searches all bin subdirs before deciding. ``qairt`` wins if either
    flavor's marker is present, since QAIRT supersedes SNPE and a given
    install commonly contains both.
    """
    if not path.is_dir():
        return None
    bin_dir = path / "bin"
    if not bin_dir.is_dir():
        return None
    saw_snpe = False
    for sub in bin_dir.iterdir():
        if not sub.is_dir():
            continue
        if _has_qairt_converter(sub):
            return "qairt"
        if _has_snpe_net_run(sub):
            saw_snpe = True
    return "snpe" if saw_snpe else None


def _version_from_dir_name(name: str) -> tuple[str | None, str | None]:
    """Pull (flavor, version) out of a directory name.

    Recognises ``qairt-2.45.0``, ``snpe-2.45.0``, ``v2.46.0.260424``,
    and bare ``2.46.0.260424``. Returns ``(None, None)`` if no match;
    flavor is ``None`` when only a version was extracted (caller decides).
    """
    m = _SDK_DIR_RE.match(name)
    if not m:
        return (None, None)
    flavor_grp = m.group(1)
    return (flavor_grp.lower() if flavor_grp else None, m.group(2))


def _select_bin_dir(bin_root: Path) -> str:
    """Pick the best per-arch bin subdir for this host.

    Ranking order:
        1. Has at least one of qairt-converter or snpe-net-run
        2. Highest host-arch preference (native arch first, emulated next)
        3. Has qairt-converter (real-mode flagship tool)
        4. Stable alphabetical fallback for determinism

    Returns "" if no bin subdir contains any recognised tool.
    """
    if not bin_root.is_dir():
        return ""
    candidates: list[tuple[int, int, int, str, Path]] = []
    for sub in sorted(bin_root.iterdir()):
        if not sub.is_dir():
            continue
        if not _has_any_tool(sub):
            continue
        candidates.append(
            (
                rank_bin_subdir(sub.name),
                int(_has_qairt_converter(sub)),
                int(_has_snpe_net_run(sub)),
                sub.name,
                sub,
            )
        )
    if not candidates:
        return ""
    # Sort: rank desc, has_qairt_converter desc, has_snpe_net_run desc, name asc
    candidates.sort(key=lambda c: (-c[0], -c[1], -c[2], c[3]))
    return str(candidates[0][4])


def _populate_sdkinfo(root: Path, source: str) -> SDKInfo:
    """Build an SDKInfo by inspecting ``root``."""
    bin_root = root / "bin"

    # Flavor: check the actual install contents, not the dir name guess.
    flavor = _looks_like_sdk_root(root) or "qairt"

    # Version: try the root dir name, then immediate children, then "unknown".
    _, version = _version_from_dir_name(root.name)
    if version is None and root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                _, ver = _version_from_dir_name(child.name)
                if ver:
                    version = ver
                    break
    if version is None:
        version = "unknown"

    # bin_dir: rank-based selection, native arch wins.
    bin_dir = _select_bin_dir(bin_root) if bin_root.is_dir() else ""

    # Capability flags: scan ALL bin subdirs, not just the chosen one,
    # since QAIRT splits converters and runtime tools across arches.
    has_qairt = has_snpe = False
    if bin_root.is_dir():
        for sub in bin_root.iterdir():
            if not sub.is_dir():
                continue
            if _has_qairt_converter(sub):
                has_qairt = True
            if _has_snpe_net_run(sub):
                has_snpe = True

    return SDKInfo(
        root=str(root),
        version=version,
        flavor=flavor,
        source=source,
        bin_dir=bin_dir,
        has_qairt_converter=has_qairt,
        has_snpe_net_run=has_snpe,
    )


def _expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p))).resolve()


def _scan_dir_for_sdks(parent: Path) -> Iterable[Path]:
    """Yield direct children of ``parent`` that look like an SDK root."""
    if not parent.is_dir():
        return
    # Some users unpack directly so parent IS the SDK root
    if _looks_like_sdk_root(parent):
        yield parent
        return
    for child in sorted(parent.iterdir(), reverse=True):
        if child.is_dir() and _looks_like_sdk_root(child):
            yield child


def discover_sdks(
    extra_paths: Iterable[str] = (),
    project_root: Path | None = None,
) -> list[SDKInfo]:
    """Walk all standard locations and return every SDK found.

    Args:
        extra_paths: Additional directories to scan (e.g. from quad.toml).
        project_root: Defaults to ``cwd``. The project-local ``./sdks``
            directory is resolved relative to this.

    Returns:
        List of SDKInfo, ordered by preference (env > config > project > user > vendor).
    """
    found: list[SDKInfo] = []
    seen: set[str] = set()

    def _add(root: Path, source: str) -> None:
        key = str(root.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        info = _populate_sdkinfo(root, source)
        found.append(info)

    # 1. Env vars
    for var in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT"):
        val = os.environ.get(var, "").strip()
        if not val:
            continue
        p = Path(val).resolve()
        if _looks_like_sdk_root(p):
            _add(p, f"env:{var}")
        elif p.is_dir():
            for sdk in _scan_dir_for_sdks(p):
                _add(sdk, f"env:{var}")

    # 2. Caller-supplied paths (e.g. from quad.toml)
    for p in extra_paths:
        ep = _expand(p)
        if _looks_like_sdk_root(ep):
            _add(ep, "config")
        else:
            for sdk in _scan_dir_for_sdks(ep):
                _add(sdk, "config")

    # 3. Project-local
    project_root = (project_root or Path.cwd()).resolve()
    project_sdks = project_root / "sdks"
    for sdk in _scan_dir_for_sdks(project_sdks):
        _add(sdk, "project:./sdks")

    # 4. Default scan paths (~/.quad/sdks, vendor defaults)
    for p in DEFAULT_SCAN_PATHS:
        if p == "./sdks":
            continue  # handled above with explicit project_root
        ep = _expand(p)
        for sdk in _scan_dir_for_sdks(ep):
            _add(sdk, f"scan:{p}")

    return found


def resolve_sdk_root(
    extra_paths: Iterable[str] = (),
    project_root: Path | None = None,
) -> SDKInfo | None:
    """Resolve the single SDK root the server should use.

    Returns ``None`` if no SDK is installed; in that case callers should
    fall back to mock mode and/or print the missing-SDK guidance.
    """
    candidates = discover_sdks(extra_paths=extra_paths, project_root=project_root)
    return candidates[0] if candidates else None


# ── Install (from a user-downloaded archive) ─────────────────────────────────


@dataclass
class InstallResult:
    """Result of unpacking a user-downloaded archive."""

    root: str
    version: str
    flavor: str
    files_extracted: int
    archive_path: str
    target_dir: str
    bytes_extracted: int = 0


def install_archive(
    archive_path: str | Path,
    target_dir: str | Path | None = None,
    project_root: Path | None = None,
    overwrite: bool = False,
) -> InstallResult:
    """Unpack a user-downloaded SDK archive into ``./sdks/`` (default).

    Args:
        archive_path: Path to a ``.zip``, ``.tar.gz``, or ``.tgz`` file
            downloaded from the Qualcomm developer portal.
        target_dir: Override destination. Defaults to
            ``<project_root>/sdks/<archive-stem>``.
        project_root: Defaults to ``cwd``.
        overwrite: If True, remove the target dir before extracting.

    Raises:
        FileNotFoundError: archive does not exist.
        ValueError: archive is not a recognised format or doesn't contain
            an SDK layout (no ``bin/`` directory after extraction).
    """
    archive = Path(archive_path).resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    project_root = (project_root or Path.cwd()).resolve()
    base_target = (project_root / "sdks").resolve()
    base_target.mkdir(parents=True, exist_ok=True)

    # Strip common archive suffixes (.tar.gz before .gz)
    name = archive.name
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    final_dir = Path(target_dir).resolve() if target_dir else (base_target / name)

    if final_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target {final_dir} exists. Pass overwrite=True or remove it first."
            )
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    # Stage to a tmp dir to handle archives that wrap their content in a
    # top-level dir (so we don't end up with sdks/qairt-2.45/qairt-2.45/...)
    staging = base_target / f".{name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    files_extracted = 0
    bytes_extracted = 0

    if archive.suffix.lower() == ".zip" or archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                # Defend against zip-slip (path traversal)
                target_path = (staging / member.filename).resolve()
                try:
                    target_path.relative_to(staging.resolve())
                except ValueError:
                    raise ValueError(f"Archive contains unsafe path: {member.filename}")
                zf.extract(member, staging)
                files_extracted += 1
                bytes_extracted += member.file_size
    elif archive.name.lower().endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            members = tf.getmembers()
            for m in members:
                # tar-slip defence
                target_path = (staging / m.name).resolve()
                try:
                    target_path.relative_to(staging.resolve())
                except ValueError:
                    raise ValueError(f"Archive contains unsafe path: {m.name}")
            tf.extractall(staging)
            files_extracted = len(members)
            bytes_extracted = sum(m.size for m in members)
    else:
        raise ValueError(
            f"Unsupported archive format: {archive.name}. "
            "Use .zip, .tar.gz, .tgz, or .tar."
        )

    # If the archive had a single top-level dir, hoist its contents up
    children = [c for c in staging.iterdir() if c.name not in (".", "..")]
    if len(children) == 1 and children[0].is_dir():
        nested = children[0]
        for entry in nested.iterdir():
            shutil.move(str(entry), str(final_dir / entry.name))
        nested.rmdir()
        staging.rmdir()
    else:
        for entry in children:
            shutil.move(str(entry), str(final_dir / entry.name))
        staging.rmdir()

    flavor = _looks_like_sdk_root(final_dir)
    if flavor is None:
        # Some QAIRT archives place bin/ one level deep — try a deeper match
        for child in final_dir.iterdir():
            if child.is_dir() and _looks_like_sdk_root(child):
                # Hoist that subdirectory's contents up
                for entry in child.iterdir():
                    shutil.move(str(entry), str(final_dir / entry.name))
                child.rmdir()
                flavor = _looks_like_sdk_root(final_dir)
                break

    if flavor is None:
        raise ValueError(
            f"Extracted contents at {final_dir} do not look like a QAIRT/SNPE SDK "
            f"(expected a 'bin/<arch>/' directory containing qairt-converter or "
            f"snpe-net-run). Verify you downloaded the right archive from "
            f"{QAIRT_PRODUCT_URL}"
        )

    info = _populate_sdkinfo(final_dir, source="install")
    return InstallResult(
        root=info.root,
        version=info.version,
        flavor=info.flavor,
        files_extracted=files_extracted,
        bytes_extracted=bytes_extracted,
        archive_path=str(archive),
        target_dir=str(final_dir),
    )


# ── Auto-download (opt-in only — requires pre-stored credentials) ────────────


def auto_download_enabled() -> bool:
    """Whether the experimental auto-download path is enabled."""
    return os.environ.get("QUAD_SDK_AUTO_DOWNLOAD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def missing_sdk_message(reason: str = "") -> str:
    """Human-readable guidance shown when no SDK is installed."""
    suffix = f"\nReason: {reason}" if reason else ""
    return (
        "No Qualcomm AI Runtime SDK detected. The MCP server will run in "
        "mock mode. To enable real hardware mode:\n"
        f"  1. Download QAIRT from {QAIRT_PRODUCT_URL}\n"
        f"     (or SNPE from        {SNPE_PRODUCT_URL})\n"
        "     Both downloads require a Qualcomm developer account + EULA "
        "acceptance.\n"
        "  2. Save the .zip / .tar.gz somewhere on disk.\n"
        "  3. Run: quad sdk install <path-to-archive>\n"
        "     This unpacks into ./sdks/ — gitignored, picked up automatically.\n"
        "  4. Restart the MCP server (or run `quad mode` to confirm READY)."
        f"{suffix}"
    )


# ── Configuration helpers ────────────────────────────────────────────────────


def list_all_bin_dirs(root: str | Path) -> list[str]:
    """Return every per-arch bin subdir of an SDK install, host-arch first.

    QAIRT 2.x splits converters (e.g. `arm64x-windows-msvc/qairt-converter`)
    and runtime tools (e.g. `aarch64-windows-msvc/qnn-platform-validator`)
    across separate per-arch subdirs. To find any tool by name we need
    to look in all of them, ranked by host preference.
    """
    root = Path(root)
    bin_root = root / "bin"
    if not bin_root.is_dir():
        return []
    subs = [s for s in bin_root.iterdir() if s.is_dir() and _has_any_tool(s)]
    subs.sort(key=lambda s: (-rank_bin_subdir(s.name), s.name))
    return [str(s) for s in subs]


def apply_to_environment(info: SDKInfo) -> None:
    """Set the SDK env vars + PATH so child processes inherit the discovery.

    This is what the MCP server startup hook calls after a successful
    discovery so the QAIRTAdapter (and any subprocess it spawns) sees
    the SDK without the user editing their shell init.

    All per-arch bin subdirs of the SDK are prepended to PATH so a tool
    found only in (say) ``aarch64-windows-msvc`` is reachable even when
    ``bin_dir`` points at the converter-bearing arch.
    """
    os.environ.setdefault("QAIRT_SDK_ROOT", info.root)
    os.environ.setdefault("QNN_SDK_ROOT", info.root)
    os.environ.setdefault("SNPE_ROOT", info.root)
    sep = ";" if os.name == "nt" else ":"
    cur_parts = os.environ.get("PATH", "").split(sep)
    additions: list[str] = []
    # Primary bin first (so it stays at the front of PATH for users that
    # rely on a specific arch).
    if info.bin_dir and info.bin_dir not in cur_parts:
        additions.append(info.bin_dir)
    # Then every other live per-arch bin subdir of the install. This is
    # what lets us find tools that live in a different arch from the
    # primary (e.g. converters in arm64x, runtime in aarch64).
    for bd in list_all_bin_dirs(info.root):
        if bd not in cur_parts and bd not in additions:
            additions.append(bd)
    if additions:
        os.environ["PATH"] = sep.join(additions + cur_parts)


def write_state_file(info: SDKInfo | None, project_root: Path | None = None) -> Path:
    """Write a small state file recording the resolved SDK.

    Useful for ``quad sdk status`` and for debugging across sessions.
    """
    project_root = (project_root or Path.cwd()).resolve()
    state_dir = project_root / ".quad"
    state_dir.mkdir(exist_ok=True)
    state_path = state_dir / "sdk.json"
    payload: dict[str, object] = {
        "resolved": info.to_dict() if info else None,
        "qairt_url": QAIRT_PRODUCT_URL,
        "snpe_url": SNPE_PRODUCT_URL,
        "scanned_paths": list(DEFAULT_SCAN_PATHS),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    state_path.write_text(json.dumps(payload, indent=2))
    return state_path


# ── Entry point used by the MCP server startup hook ──────────────────────────


def startup_resolve_and_log(
    extra_paths: Iterable[str] = (),
    project_root: Path | None = None,
) -> SDKInfo | None:
    """Resolve the SDK at server startup and log the result.

    * If found, sets QAIRT_SDK_ROOT/QNN_SDK_ROOT/SNPE_ROOT for child
      processes and returns the SDKInfo.
    * If not found, logs the missing-SDK guidance message at INFO level
      and returns None — the server continues in mock mode.

    Always writes ``.quad/sdk.json`` so downstream tools can inspect.
    """
    info = resolve_sdk_root(extra_paths=extra_paths, project_root=project_root)
    if info is not None:
        apply_to_environment(info)
        logger.info(
            "qairt_sdk_resolved",
            extra={
                "root": info.root,
                "version": info.version,
                "flavor": info.flavor,
                "source": info.source,
                "bin_dir": info.bin_dir,
            },
        )
    else:
        # Use a single info-level message so the SPA-style downloads page
        # URLs are visible without needing debug logs.
        logger.info("qairt_sdk_not_found\n%s", missing_sdk_message())
    write_state_file(info, project_root=project_root)
    return info
