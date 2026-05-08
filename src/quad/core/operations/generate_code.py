"""Pure code-generation operation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from quad.codegen.engine import CodegenEngine


async def generate_code(
    platform: Literal["windows", "linux", "android"],
    sdk: Literal["qnn", "snpe"],
    language: Literal["cpp", "python", "java", "kotlin", "arduino_sketch"],
    model_path: str,
    *,
    template_dir: str | None = None,
) -> dict[str, Any]:
    """Generate inference code. Returns GeneratedCode dict (no enrichment)."""
    engine = CodegenEngine(template_dir=template_dir) if template_dir else CodegenEngine()
    variables = {
        "model_path": model_path,
        "sdk": sdk,
        "runtime": "npu",
        "platform": platform,
        "sdk_path": "",
    }
    result = engine.render(platform, language, variables, sdk=sdk)
    return result.model_dump()
