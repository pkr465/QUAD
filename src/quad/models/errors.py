"""Error response model."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ToolError(BaseModel):
    """Structured error response from QUAD tools."""

    code: str
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = False
    suggestion: str | None = None
