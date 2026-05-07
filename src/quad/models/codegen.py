"""Pydantic models for code generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from quad.models.orchestration import AllocationMap


class CodegenRequest(BaseModel):
    """Input for generate_code tool."""

    platform: Literal["windows", "linux", "android"]
    sdk: Literal["qnn", "snpe"] = "qnn"
    language: Literal["cpp", "python", "java", "kotlin", "arduino_sketch"] = "python"
    model_path: str = "model.bin"
    allocation_map: AllocationMap | None = None


class GeneratedCode(BaseModel):
    """Output from generate_code tool."""

    source_files: dict[str, str] = Field(description="filename → source content")
    build_instructions: str
    dependencies: list[str] = Field(default_factory=list)
    language: str
    platform: str
    sdk: str
    sample_input: str = ""
    expected_output_format: str = ""
