"""Jinja2-based code generation engine for QUAD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from quad.codegen.validators import validate_output
from quad.exceptions import TemplateRenderError, UnsupportedLanguageError
from quad.models.codegen import GeneratedCode


class CodegenEngine:
    """Renders platform/language-specific inference code from Jinja2 templates."""

    def __init__(self, template_dir: str | Path = "templates"):
        self._template_dir = Path(template_dir)
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render(
        self,
        platform: str,
        language: str,
        variables: dict[str, Any],
        sdk: str = "qnn",
    ) -> GeneratedCode:
        """Render all templates for the given platform/language combination.

        Args:
            platform: Target platform (windows, linux, android)
            language: Target language (cpp, python, kotlin, arduino_sketch)
            variables: Template variables (model_path, sdk_imports, etc.)
            sdk: Target SDK name

        Returns:
            GeneratedCode with rendered source files and build instructions.

        Raises:
            UnsupportedLanguageError: If no templates exist for the combination.
            TemplateRenderError: If template rendering fails.
        """
        template_path = self._template_dir / platform / language
        if not template_path.exists():
            raise UnsupportedLanguageError(language, platform)

        source_files: dict[str, str] = {}
        template_files = list(template_path.glob("*.j2"))

        if not template_files:
            raise UnsupportedLanguageError(language, platform)

        for tmpl_file in template_files:
            # Template path relative to template_dir
            rel_path = tmpl_file.relative_to(self._template_dir)
            # Output filename = template name without .j2 extension
            output_name = tmpl_file.stem  # e.g. inference.py from inference.py.j2

            try:
                # Jinja2 always expects forward-slash paths regardless of OS
                template = self._env.get_template(rel_path.as_posix())
                rendered = template.render(**variables)
            except TemplateNotFound as e:
                raise TemplateRenderError(str(rel_path), f"Template not found: {e}") from e
            except Exception as e:
                raise TemplateRenderError(str(rel_path), str(e)) from e

            source_files[output_name] = rendered

        # Validate rendered output
        for filename, content in source_files.items():
            errors = validate_output(filename, content)
            if errors:
                raise TemplateRenderError(filename, f"Validation failed: {'; '.join(errors)}")

        # Build instructions based on language
        build_instructions = self._get_build_instructions(language, sdk)
        dependencies = self._get_dependencies(language, sdk)

        return GeneratedCode(
            source_files=source_files,
            build_instructions=build_instructions,
            dependencies=dependencies,
            language=language,
            platform=platform,
            sdk=sdk,
            sample_input="Random tensor [1, 3, 224, 224] (ImageNet-sized)",
            expected_output_format="Tensor [1, 1000] (class probabilities)",
        )

    def _get_build_instructions(self, language: str, sdk: str) -> str:
        instructions = {
            "python": f"pip install -r requirements.txt\npython inference.py",
            "cpp": "mkdir build && cd build\ncmake ..\ncmake --build . --config Release\n./inference",
            "kotlin": "./gradlew assembleRelease",
            "java": "./gradlew assembleRelease",
            "arduino_sketch": "arduino-cli compile --fqbn qualcomm:arm:unoq .\narduino-cli upload --fqbn qualcomm:arm:unoq .",
        }
        return instructions.get(language, f"# Build instructions for {language}")

    def _get_dependencies(self, language: str, sdk: str) -> list[str]:
        deps: dict[str, list[str]] = {
            "python": [f"{'qnn-python' if sdk == 'qnn' else 'snpe-python'}", "numpy", "pillow"],
            "cpp": [f"{'QNN SDK' if sdk == 'qnn' else 'SNPE SDK'}", "CMake 3.22+"],
            "kotlin": ["com.qualcomm.qti:snpe-release:2.x", "androidx.core:core-ktx"],
            "arduino_sketch": ["SNPE runtime libraries", "aarch64-linux-gnu-gcc"],
        }
        return deps.get(language, [])
