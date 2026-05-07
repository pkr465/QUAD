"""generate_code tool — Generate platform-specific inference code."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from quad.codegen.engine import CodegenEngine
from quad.exceptions import UnsupportedLanguageError


async def generate_code_impl(
    platform: Literal["windows", "linux", "android"],
    sdk: Literal["qnn", "snpe"],
    language: Literal["cpp", "python", "java", "kotlin", "arduino_sketch"],
    model_path: str,
    template_dir: str = "templates",
) -> dict[str, Any]:
    """Generate code and return GeneratedCode as dict."""
    engine = CodegenEngine(template_dir=template_dir)

    variables = {
        "model_path": model_path,
        "sdk": sdk,
        "runtime": "npu",
        "platform": platform,
        "sdk_path": "",  # Will be filled from config in real mode
    }

    result = engine.render(platform, language, variables, sdk=sdk)
    return result.model_dump()
