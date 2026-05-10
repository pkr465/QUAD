"""Download + verification logic for the QUAD model registry.

Pure HTTP-stream download with optional SHA-256 check, atomic rename,
re-use of cached files, and a clear error message when an entry is
user-supplied (``path_env_var`` mode) but the env var isn't set.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from quad.model_registry.manifest import ModelEntry, resolve_entry


class ModelFetchError(RuntimeError):
    """Raised when a registry entry can't be fetched or verified."""


def cache_root() -> Path:
    """Per-user model cache: ``~/.quad/models/`` unless QUAD_MODELS_DIR is set."""
    override = os.environ.get("QUAD_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".quad" / "models"


def _cached_path(entry: ModelEntry) -> Path:
    return cache_root() / entry.name / entry.output_filename


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(
    url: str,
    dest: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """HTTPS stream download with progress callbacks. Atomic rename on success.

    Uses httpx if available (already in QUAD's deps), else stdlib urllib.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp returns (fd, path); we must close the fd before opening the
    # path with another handle, otherwise os.replace fails on Windows
    # with WinError 32 ("file in use by another process").
    fd, tmp_str = tempfile.mkstemp(prefix=dest.name + ".", dir=dest.parent)
    os.close(fd)
    tmp = Path(tmp_str)

    try:
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx is in QUAD deps
            httpx = None

        if httpx is not None:
            with httpx.Client(follow_redirects=True, timeout=300.0) as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total = int(resp.headers.get("content-length", 0))
                    written = 0
                    with tmp.open("wb") as out:
                        for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                            if not chunk:
                                continue
                            out.write(chunk)
                            written += len(chunk)
                            if progress:
                                progress(written, total)
        else:
            from urllib.request import urlopen

            with urlopen(url, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                written = 0
                with tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        out.write(chunk)
                        written += len(chunk)
                        if progress:
                            progress(written, total)

        # Atomic rename — replaces any partially-cached file safely.
        if dest.exists():
            dest.unlink()
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def fetch_model(
    name: str,
    *,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Ensure the named model is present locally; return the absolute path.

    For ``url`` entries: downloads if missing or if ``force=True``, then
    verifies SHA-256 if the entry declares one.

    For ``path_env_var`` entries: returns the path the env var points to,
    raising ``ModelFetchError`` if the var isn't set or the file is
    missing.

    Args:
        name:    registry entry name.
        force:   re-download even if already cached (url entries only).
        progress(written, total): optional callback for byte-progress.

    Returns:
        Absolute Path to the on-disk model file.

    Raises:
        KeyError:         entry not in registry.
        ModelFetchError:  download failed, sha mismatch, or env var unset.
    """
    entry = resolve_entry(name)

    if entry.path_env_var:
        raw = os.environ.get(entry.path_env_var)
        if not raw:
            raise ModelFetchError(
                f"{entry.name}: requires user-supplied weights. "
                f"Set {entry.path_env_var}=/path/to/{entry.output_filename}. "
                f"License: {entry.license or 'unspecified'}."
            )
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise ModelFetchError(
                f"{entry.name}: {entry.path_env_var}={raw!s} but file does not exist."
            )
        return path

    # url entry
    cached = _cached_path(entry)
    if cached.exists() and not force:
        if entry.sha256:
            actual = _sha256(cached)
            if actual.lower() == entry.sha256.lower():
                return cached
            # Otherwise fall through to re-download.
        else:
            return cached

    if not entry.url:
        raise ModelFetchError(f"{entry.name}: registry entry has no source")

    _stream_download(entry.url, cached, progress=progress)

    if entry.sha256:
        actual = _sha256(cached)
        if actual.lower() != entry.sha256.lower():
            cached.unlink(missing_ok=True)
            raise ModelFetchError(
                f"{entry.name}: SHA-256 mismatch. "
                f"expected {entry.sha256[:16]}... got {actual[:16]}..."
            )

    return cached


def resolve_model_path(name: str) -> Path:
    """Return the local path for a model without downloading it.

    Useful for fast plan-app start-up: defers the network/IO until the
    caller actually needs the file. Returns the cache path even if the
    file doesn't exist yet (so the caller can present a helpful error).

    For ``path_env_var`` entries this resolves the env var (and raises
    if unset) — there's no on-disk fallback.
    """
    entry = resolve_entry(name)
    if entry.path_env_var:
        return fetch_model(name)  # env-var path needs to exist; let it raise
    return _cached_path(entry)
