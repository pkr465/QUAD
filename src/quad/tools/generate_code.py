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
    payload = result.model_dump()

    # Enrich with file summary + tips
    try:
        from quad.tips import get_tips_for
        from quad.ui.formatters import format_table

        files = payload.get("source_files", {}) or {}
        rows = [
            [name, len(content.splitlines()), len(content)]
            for name, content in files.items()
        ]
        payload["ui"] = (
            f"### Generated code: {platform} / {language} / {sdk}\n\n"
            + format_table(["File", "Lines", "Bytes"], rows, align=["l", "r", "r"])
            + f"\n\n**Build:** `{payload.get('build_instructions', '?')}`\n"
            + f"**Deps:** {', '.join(payload.get('dependencies', []) or ['—'])}"
        )
        payload["tips"] = [t.text for t in get_tips_for("codegen", n=2)]
    except Exception:
        pass

    return payload
