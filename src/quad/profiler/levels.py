"""Shared profiling level definitions for SNPE/QAIRT.

Single source of truth for profiling level names used across:
- CLI flags (--profiling_level)
- ProfileRequest model
- Adapter dispatch
- snpe-dlc-graph-prepare, snpe-net-run, qnn-profile-viewer
"""

from __future__ import annotations

from enum import Enum


class ProfilingLevel(str, Enum):
    """SNPE/QAIRT profiling levels, in increasing detail order.

    Basic and Detailed are available on all backends.
    Linting and QHAS are HTP-only; non-HTP subnets silently fall back to Detailed.
    """
    OFF = "off"              # No profiling
    BASIC = "basic"          # Summary timing only
    DETAILED = "detailed"    # Per-layer microsecond timing (default)
    LINTING = "linting"      # HTP-only — cycle counts per op, chrometrace export
    QHAS = "qhas"            # HTP-only — full QNN HTP Analysis Summary

    @property
    def is_htp_only(self) -> bool:
        """True if this level requires HTP backend (non-HTP subnets fall back to Detailed)."""
        return self in (ProfilingLevel.LINTING, ProfilingLevel.QHAS)

    @property
    def supports_chrometrace(self) -> bool:
        """True if this level can generate a chrometrace JSON for chrome://tracing."""
        return self in (ProfilingLevel.LINTING, ProfilingLevel.QHAS)

    @property
    def fallback_level(self) -> "ProfilingLevel":
        """Level used for non-HTP subnets when this level is HTP-only."""
        if self.is_htp_only:
            return ProfilingLevel.DETAILED
        return self
