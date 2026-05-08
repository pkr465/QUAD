"""Jinja2-based code generation engine for QUAD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from quad.codegen.validators import validate_output
from quad.exceptions import TemplateRenderError, UnsupportedLanguageError
from quad.models.codegen import GeneratedCode

logger = logging.getLogger(__name__)


def resolve_template_dir(template_dir: str | Path | None = None) -> Path:
    """Find the templates directory.

    Resolution order:
      1. Explicit ``template_dir`` argument (if provided and exists)
      2. ``<quad-package-dir>/templates`` — populated by hatch
         ``force-include`` when the package is installed via pip
      3. ``<repo-root>/templates`` — for source-tree development where
         the package is installed in editable mode without re-bundling
      4. ``./templates`` relative to the current working directory
         — last-resort fallback for ad-hoc usage

    Raises ``FileNotFoundError`` if nothing resolves.
    """
    candidates: list[Path] = []
    if template_dir:
        candidates.append(Path(template_dir))

    # Bundled-with-package location (force-included by hatch)
    try:
        import quad
        candidates.append(Path(quad.__file__).resolve().parent / "templates")
    except Exception:
        pass

    # Source-tree fallback: walk up from this file looking for templates/
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidates.append(parent / "templates")
        if parent.name == "src":
            candidates.append(parent.parent / "templates")
            break

    # CWD fallback
    candidates.append(Path.cwd() / "templates")

    for cand in candidates:
        if cand.is_dir() and any(cand.rglob("*.j2")):
            return cand

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate templates directory containing *.j2 files. Tried:\n  {tried}\n"
        "If you installed via pip, this likely means templates were not bundled in the wheel — "
        "ensure pyproject.toml has [tool.hatch.build.targets.wheel.force-include]."
    )


class CodegenEngine:
    """Renders platform/language-specific inference code from Jinja2 templates."""

    def __init__(self, template_dir: str | Path | None = None):
        # If template_dir is explicit, use it directly. If None or
        # missing, fall back to the resolver chain so installed and
        # source-tree usage both work out of the box.
        if template_dir is not None and Path(template_dir).is_dir():
            self._template_dir = Path(template_dir).resolve()
        else:
            self._template_dir = resolve_template_dir(template_dir)
        logger.debug("codegen_engine_template_dir", extra={"path": str(self._template_dir)})
        self._env = Environment(
            loader=FileSystemLoader(str(self._template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    @property
    def template_dir(self) -> Path:
        """The resolved templates directory in use."""
        return self._template_dir

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
