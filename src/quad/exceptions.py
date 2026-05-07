"""QUAD exception hierarchy."""

from __future__ import annotations


class QUADError(Exception):
    """Base exception for all QUAD errors."""

    def __init__(self, message: str, code: str = "UNKNOWN_ERROR", recoverable: bool = False):
        self.code = code
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


# ── Platform Errors ──


class PlatformError(QUADError):
    """Base for platform-related errors."""

    pass


class PlatformNotDetectedError(PlatformError):
    """Target platform or device not found."""

    def __init__(self, platform: str):
        super().__init__(
            message=f"Platform '{platform}' not detected or not supported.",
            code="PLATFORM_NOT_DETECTED",
        )


class DeviceNotConnectedError(PlatformError):
    """Target device not reachable."""

    def __init__(self, device: str = ""):
        super().__init__(
            message=f"Device not connected: {device}" if device else "Device not connected.",
            code="DEVICE_NOT_CONNECTED",
            recoverable=True,
        )


# ── SDK Errors ──


class SDKError(QUADError):
    """Base for SDK-related errors."""

    pass


class SDKNotFoundError(SDKError):
    """Required SDK not installed at configured path."""

    def __init__(self, sdk: str, path: str = ""):
        super().__init__(
            message=f"SDK '{sdk}' not found at path: {path}" if path else f"SDK '{sdk}' not found.",
            code="SDK_NOT_FOUND",
        )


class SDKVersionMismatchError(SDKError):
    """SDK version incompatible with requested operation."""

    def __init__(self, sdk: str, required: str, found: str):
        super().__init__(
            message=f"SDK '{sdk}' version mismatch: required {required}, found {found}.",
            code="SDK_VERSION_MISMATCH",
        )


class SDKExecutionError(SDKError):
    """SDK command or API call failed."""

    def __init__(self, sdk: str, command: str, stderr: str = ""):
        super().__init__(
            message=f"SDK '{sdk}' execution failed: {command}. {stderr}".strip(),
            code="SDK_EXECUTION_FAILED",
            recoverable=True,
        )


# ── Conversion Errors ──


class ConversionError(QUADError):
    """Base for model conversion errors."""

    pass


class UnsupportedFormatError(ConversionError):
    """Model format not supported for conversion."""

    def __init__(self, format: str):
        super().__init__(
            message=f"Unsupported model format: '{format}'.",
            code="UNSUPPORTED_FORMAT",
        )


class ConversionFailedError(ConversionError):
    """Model conversion process failed."""

    def __init__(self, reason: str, unsupported_ops: list[str] | None = None):
        self.unsupported_ops = unsupported_ops or []
        super().__init__(
            message=f"Model conversion failed: {reason}",
            code="CONVERSION_FAILED",
        )


class QuantizationError(ConversionError):
    """Quantization step failed."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Quantization failed: {reason}",
            code="QUANTIZATION_FAILED",
        )


# ── Profiling Errors ──


class ProfilingError(QUADError):
    """Base for profiling errors."""

    pass


class ProfilerNotFoundError(ProfilingError):
    """Profiler tool not found."""

    def __init__(self, profiler: str):
        super().__init__(
            message=f"Profiler not found: '{profiler}'.",
            code="PROFILER_NOT_FOUND",
        )


class ProfilingTimeoutError(ProfilingError):
    """Profiling exceeded timeout."""

    def __init__(self, timeout_s: float):
        super().__init__(
            message=f"Profiling timed out after {timeout_s}s.",
            code="PROFILER_TIMEOUT",
            recoverable=True,
        )


class InsufficientMemoryError(ProfilingError):
    """Device does not have enough memory for the model."""

    def __init__(self, required_mb: float, available_mb: float):
        super().__init__(
            message=f"Insufficient memory: need {required_mb:.0f}MB, have {available_mb:.0f}MB.",
            code="MEMORY_EXCEEDED",
        )


# ── Orchestration Errors ──


class OrchestrationError(QUADError):
    """Base for orchestration errors."""

    pass


class InvalidProfileError(OrchestrationError):
    """Profile data is invalid or incomplete."""

    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid profile data: {reason}",
            code="INVALID_PROFILE",
        )


class MemoryExceededError(OrchestrationError):
    """Orchestration exceeds device memory budget."""

    def __init__(self, required_mb: float, budget_mb: float):
        super().__init__(
            message=f"Memory budget exceeded: need {required_mb:.0f}MB, budget {budget_mb:.0f}MB.",
            code="MEMORY_EXCEEDED",
        )


# ── Code Generation Errors ──


class CodegenError(QUADError):
    """Base for code generation errors."""

    pass


class TemplateRenderError(CodegenError):
    """Template rendering failed."""

    def __init__(self, template: str, reason: str):
        super().__init__(
            message=f"Template render failed for '{template}': {reason}",
            code="TEMPLATE_ERROR",
        )


class UnsupportedLanguageError(CodegenError):
    """Language not supported for the given platform/SDK combination."""

    def __init__(self, language: str, platform: str):
        super().__init__(
            message=f"Language '{language}' not supported for platform '{platform}'.",
            code="UNSUPPORTED_LANGUAGE",
        )
