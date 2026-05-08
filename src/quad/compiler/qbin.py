"""QUAD Binary (.qbin) — fat binary container format."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quad.compiler.ir import IRGraph, QuadIR


@dataclass
class TargetBinary:
    """Compiled binary for a specific target."""
    target: str  # e.g. "qnpu_v3", "qadreno_x1"
    format: str  # "qnn", "snpe", "hexagon"
    size_bytes: int = 0
    data: bytes = b""


@dataclass
class QBin:
    """QUAD Binary — fat binary containing IR + target-specific compilations.

    Embeds a QIR plus pre-compiled binaries for one or more targets
    (CPU / Adreno / Hexagon HTP, possibly multiple HTP versions).
    At load time, select best match for current hardware; JIT from QIR if none match.
    """
    name: str
    version: str = "1.0"
    ir: IRGraph | None = None
    targets: dict[str, TargetBinary] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    def add_target(self, target: str, format: str, data: bytes = b"") -> None:
        """Add a target-specific compilation."""
        self.targets[target] = TargetBinary(
            target=target,
            format=format,
            size_bytes=len(data),
            data=data,
        )

    def has_target(self, target: str) -> bool:
        """Check if a specific target is compiled."""
        return target in self.targets

    def get_best_target(self, device_capability: str) -> TargetBinary | None:
        """Select best compiled target for given device capability."""
        # Exact match
        if device_capability in self.targets:
            return self.targets[device_capability]
        # Fallback: any NPU target
        for t in self.targets.values():
            if "npu" in t.target or "hexagon" in t.target:
                return t
        # Any target
        return next(iter(self.targets.values())) if self.targets else None

    @property
    def num_targets(self) -> int:
        return len(self.targets)

    @property
    def total_size_bytes(self) -> int:
        return sum(t.size_bytes for t in self.targets.values())

    def save(self, path: str) -> None:
        """Save QBin to file (JSON manifest + binary blobs)."""
        manifest = {
            "format": "qbin",
            "version": self.version,
            "name": self.name,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "targets": {
                name: {"target": t.target, "format": t.format, "size_bytes": t.size_bytes}
                for name, t in self.targets.items()
            },
            "has_ir": self.ir is not None,
            "ir": self.ir.to_dict() if self.ir else None,
        }
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, path: str) -> QBin:
        """Load QBin from file."""
        with open(path, "r") as f:
            manifest = json.load(f)

        qbin = cls(
            name=manifest["name"],
            version=manifest.get("version", "1.0"),
            created_at=manifest.get("created_at", ""),
            metadata=manifest.get("metadata", {}),
        )

        if manifest.get("ir"):
            qbin.ir = IRGraph.from_dict(manifest["ir"])

        for name, t_data in manifest.get("targets", {}).items():
            qbin.targets[name] = TargetBinary(
                target=t_data["target"],
                format=t_data["format"],
                size_bytes=t_data.get("size_bytes", 0),
            )

        return qbin

    def __repr__(self) -> str:
        return (
            f"QBin(name='{self.name}', targets={list(self.targets.keys())}, "
            f"has_ir={self.ir is not None})"
        )
