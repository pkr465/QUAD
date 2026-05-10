"""Manifest schema + loader for the QUAD model registry.

Registry entries live in ``registry.yaml`` next to this module. Entries
can also be added at runtime via ``register_entry()`` — useful for new
plans (or test fixtures) that want to provision models without editing
the source manifest.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - yaml is in the project deps
    yaml = None


_REGISTRY_LOCK = threading.Lock()
_RUNTIME_ENTRIES: dict[str, "ModelEntry"] = {}


@dataclass(frozen=True)
class ModelEntry:
    """A single model in the registry.

    Required fields:
        name:        unique key (also doubles as the on-disk folder name)
        plan:        owning plan id ("plan1", "plan2", "plan4", "shared", ...)
        description: one-line human description

    Source — exactly one of these is required:
        url:                 direct HTTPS URL (auto-downloadable)
        path_env_var:        env var the user sets to a local file path

    Optional metadata:
        sha256:    expected hex digest of the file (verified after fetch)
        size_mb:   approximate download size, advisory only
        license:   SPDX-style identifier ("Apache-2.0", "MIT", …)
        notes:     additional context shown in `quad models list`
        filename:  output filename (defaults to URL basename or
                   "{name}.onnx" for env-var entries)
    """
    name: str
    plan: str
    description: str
    url: str | None = None
    path_env_var: str | None = None
    sha256: str | None = None
    size_mb: float = 0.0
    license: str = ""
    notes: str = ""
    filename: str = ""

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass to validate after construction.
        if not self.url and not self.path_env_var:
            raise ValueError(
                f"ModelEntry {self.name!r}: must set either url or path_env_var"
            )
        if self.url and self.path_env_var:
            raise ValueError(
                f"ModelEntry {self.name!r}: set url OR path_env_var, not both"
            )

    @property
    def auto_downloadable(self) -> bool:
        return self.url is not None

    @property
    def output_filename(self) -> str:
        if self.filename:
            return self.filename
        if self.url:
            return self.url.rsplit("/", 1)[-1] or f"{self.name}.onnx"
        return f"{self.name}.onnx"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "plan": self.plan,
            "description": self.description,
            "url": self.url,
            "path_env_var": self.path_env_var,
            "sha256": self.sha256,
            "size_mb": self.size_mb,
            "license": self.license,
            "notes": self.notes,
            "filename": self.filename or self.output_filename,
        }


def registry_path() -> Path:
    """Path to the canonical ``registry.yaml``."""
    return Path(__file__).resolve().parent / "registry.yaml"


def _load_yaml() -> list[dict[str, Any]]:
    path = registry_path()
    if not path.exists():
        return []
    if yaml is None:
        raise RuntimeError(
            "PyYAML not installed. Install with `pip install pyyaml` or set "
            "registry entries at runtime via register_entry()."
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("models", []))


def _entry_from_dict(raw: dict[str, Any]) -> ModelEntry:
    return ModelEntry(
        name=raw["name"],
        plan=raw.get("plan", "shared"),
        description=raw.get("description", ""),
        url=raw.get("url"),
        path_env_var=raw.get("path_env_var"),
        sha256=raw.get("sha256"),
        size_mb=float(raw.get("size_mb", 0.0)),
        license=raw.get("license", ""),
        notes=raw.get("notes", ""),
        filename=raw.get("filename", ""),
    )


def _all_entries() -> dict[str, ModelEntry]:
    """Merge file + runtime entries; runtime takes precedence on name clash."""
    out: dict[str, ModelEntry] = {}
    try:
        for raw in _load_yaml():
            try:
                entry = _entry_from_dict(raw)
            except Exception as exc:
                # Skip malformed entries rather than crash CLI / API.
                print(f"[model_registry] skipping malformed entry: {exc!r}")
                continue
            out[entry.name] = entry
    except Exception as exc:
        # Missing file is fine; surface anything else (e.g. yaml parse error).
        if not isinstance(exc, FileNotFoundError):
            print(f"[model_registry] failed to load registry.yaml: {exc!r}")
    with _REGISTRY_LOCK:
        out.update(_RUNTIME_ENTRIES)
    return out


def list_models() -> list[ModelEntry]:
    """Return every entry in registry.yaml + any runtime registrations."""
    return sorted(_all_entries().values(), key=lambda e: (e.plan, e.name))


def list_for_plan(plan: str) -> list[ModelEntry]:
    """Filter entries by plan id (e.g. ``"plan1"``)."""
    return [e for e in list_models() if e.plan == plan]


def resolve_entry(name: str) -> ModelEntry:
    """Look up a single entry by name. Raises KeyError on miss."""
    entries = _all_entries()
    if name not in entries:
        # Friendlier error: list available names.
        avail = ", ".join(sorted(entries)) or "<empty>"
        raise KeyError(f"Model {name!r} not in registry. Known: {avail}")
    return entries[name]


def register_entry(entry: ModelEntry, *, replace: bool = False) -> None:
    """Add a model to the in-memory registry.

    Args:
        entry:   The ModelEntry to add.
        replace: If False (default) raise if the name already exists.
    """
    with _REGISTRY_LOCK:
        if not replace and entry.name in _RUNTIME_ENTRIES:
            raise ValueError(
                f"Entry {entry.name!r} already runtime-registered; "
                "pass replace=True to override."
            )
        _RUNTIME_ENTRIES[entry.name] = entry


def clear_runtime_entries() -> None:
    """Test-helper: drop all runtime-registered entries."""
    with _REGISTRY_LOCK:
        _RUNTIME_ENTRIES.clear()
