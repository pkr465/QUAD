"""QUAD SNPE Logging Utilities — log levels, initialization, management.

Documents the SNPE Logging APIs:

C API (include "SNPE/SNPEUtil.h"):
  - Snpe_Util_InitializeLogging(level)              — Android: logcat output
  - Snpe_Util_InitializeLoggingPath(level, logPath) — Non-Android: file + console
  - Snpe_Util_SetLogLevel(level)                    — Change level at runtime
  - Snpe_Util_TerminateLogging()                    — Stop logging

Java API (Android, import com.qualcomm.qti.snpe.{SNPE, NeuralNetwork}):
  - SNPE.logger.initializeLogging(application, logLevel)
  - SNPE.logger.setLogLevel(logLevel)
  - SNPE.logger.terminateLogging()

Log Levels (increasing importance — lower levels also captured when set):
  VERBOSE → INFO → WARN → ERROR → FATAL

Performance Note:
  - Enable ONCE at the start of the program to capture logs from all processes
  - Logging has a performance impact — disable (or use WARN/ERROR) in production
"""

from __future__ import annotations

import os
from enum import Enum
from dataclasses import dataclass


# ══════════════════════════════════════════════════════════════════════════════
# Log Levels
# ══════════════════════════════════════════════════════════════════════════════

class SNPELogLevel(str, Enum):
    """SNPE log levels in increasing order of importance.

    When a level is set, that level and all higher levels are captured.
    e.g. setting WARN captures WARN + ERROR + FATAL.
    """
    VERBOSE = "verbose"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"

    # ── C API enum names ──
    @property
    def c_enum(self) -> str:
        """C API: Snpe_LogLevel_t enum value."""
        return f"SNPE_LOG_LEVEL_{self.value.upper()}"

    # ── Java/Android enum names ──
    @property
    def java_enum(self) -> str:
        """Java API: NeuralNetwork.LogLevel enum value."""
        return f"NeuralNetwork.LogLevel.LOG_{self.value.upper()}"

    # ── Numeric severity (for comparison) ──
    @property
    def severity(self) -> int:
        return {"verbose": 0, "info": 1, "warn": 2, "error": 3, "fatal": 4}[self.value]

    def captures(self, other: "SNPELogLevel") -> bool:
        """True if setting this level will capture the other level's messages."""
        return self.severity <= other.severity

    @property
    def is_production_safe(self) -> bool:
        """True if this level is safe for production (minimal performance impact)."""
        return self.severity >= 2  # WARN and above

    @classmethod
    def for_environment(cls, is_production: bool) -> "SNPELogLevel":
        """Recommended log level for the given environment."""
        return cls.WARN if is_production else cls.INFO


# ══════════════════════════════════════════════════════════════════════════════
# Logging Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SNPELoggingConfig:
    """Configuration for SNPE logging initialization.

    Usage rules from SNPE Logging docs:
    - Initialize ONCE at the beginning of the program
    - Captures logs from ALL processes after init
    - Performance impact: use WARN or higher in production
    - Android: logs go to logcat (no log path needed)
    - Non-Android: logs go to console + log files at log_path
    """
    level: SNPELogLevel = SNPELogLevel.WARN
    log_path: str = ""          # Non-Android only: directory for log files
    is_android: bool = False    # Platform type affects which init API is called

    @property
    def c_init_call(self) -> str:
        """C API initialization call string."""
        if self.is_android or not self.log_path:
            return f'Snpe_Util_InitializeLogging({self.level.c_enum});'
        return (
            f'Snpe_Util_InitializeLoggingPath({self.level.c_enum}, "{self.log_path}");'
        )

    @property
    def c_set_level_call(self) -> str:
        """C API: change log level at runtime."""
        return f"Snpe_Util_SetLogLevel({self.level.c_enum});"

    @property
    def c_terminate_call(self) -> str:
        return "Snpe_Util_TerminateLogging();"

    @property
    def java_init_call(self) -> str:
        """Java/Android API initialization call string."""
        return (
            f"SNPE.logger.initializeLogging(application, {self.level.java_enum});"
        )

    @property
    def java_set_level_call(self) -> str:
        return f"SNPE.logger.setLogLevel({self.level.java_enum});"

    @property
    def java_terminate_call(self) -> str:
        return "SNPE.logger.terminateLogging();"

    def performance_warning(self) -> str | None:
        """Return a warning if this config may impact performance."""
        if not self.level.is_production_safe:
            return (
                f"Log level '{self.level.value}' captures verbose output. "
                "Logging has a performance impact — use WARN or ERROR in production."
            )
        return None


def get_logging_config(
    is_production: bool = False,
    is_android: bool = False,
    log_path: str = "",
) -> SNPELoggingConfig:
    """Get a recommended SNPE logging configuration.

    Args:
        is_production: Production = WARN level (minimal overhead)
                       Development = INFO level (useful diagnostics)
        is_android: Android uses logcat (no log_path needed)
        log_path: Non-Android log directory (e.g. "/tmp/snpe_logs")
    """
    level = SNPELogLevel.for_environment(is_production)
    return SNPELoggingConfig(
        level=level,
        log_path=log_path if not is_android else "",
        is_android=is_android,
    )
