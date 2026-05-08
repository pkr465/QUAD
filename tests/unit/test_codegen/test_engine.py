"""Tests for code generation engine."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from quad.codegen.engine import CodegenEngine, resolve_template_dir
from quad.codegen.validators import (
    ValidatorOptions,
    _strip_string_literals,
    validate_output,
)
from quad.exceptions import TemplateRenderError, UnsupportedLanguageError


TEMPLATE_DIR = Path(__file__).parents[3] / "templates"


class TestResolveTemplateDir:
    """Tests for the templates-directory resolver (T1.7)."""

    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        # Build a fake templates dir with a single .j2 file
        d = tmp_path / "tpl"
        (d / "windows" / "python").mkdir(parents=True)
        (d / "windows" / "python" / "x.j2").write_text("hi")
        assert resolve_template_dir(d).resolve() == d.resolve()

    def test_finds_repo_templates_when_unspecified(self) -> None:
        # In the source tree, calling with no arg should find the repo's
        # templates/ directory.
        result = resolve_template_dir()
        assert result.is_dir()
        assert any(result.rglob("*.j2"))

    def test_explicit_missing_falls_back(self, tmp_path: Path) -> None:
        # If the explicit path doesn't exist, the resolver still finds
        # something (repo templates).
        non_existent = tmp_path / "nope"
        result = resolve_template_dir(non_existent)
        assert result.is_dir()
        assert any(result.rglob("*.j2"))

    def test_engine_uses_default_resolver(self) -> None:
        engine = CodegenEngine()
        assert engine.template_dir.is_dir()
        assert any(engine.template_dir.rglob("*.j2"))

    def test_engine_explicit_path(self) -> None:
        engine = CodegenEngine(template_dir=TEMPLATE_DIR)
        assert engine.template_dir.resolve() == TEMPLATE_DIR.resolve()

    def test_resolver_raises_when_nothing_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Monkeypatch the source-tree walk to a path with no templates
        monkeypatch.chdir(tmp_path)
        # Move the template-engine module's __file__ to a fake location
        # so the source-tree walker can't find anything.
        fake_engine = tmp_path / "fake_engine.py"
        fake_engine.write_text("# fake")
        # Also move quad.__file__ — patch the import
        import quad
        monkeypatch.setattr(quad, "__file__", str(fake_engine))
        # Patch __file__ in the engine module too
        from quad.codegen import engine as engine_mod
        monkeypatch.setattr(engine_mod, "__file__", str(fake_engine))
        with pytest.raises(FileNotFoundError, match="Could not locate templates"):
            resolve_template_dir(tmp_path / "definitely-not-here")


@pytest.fixture
def engine() -> CodegenEngine:
    return CodegenEngine(template_dir=TEMPLATE_DIR)


@pytest.fixture
def basic_vars() -> dict:
    return {
        "model_path": "model.bin",
        "sdk": "qnn",
        "sdk_path": "/opt/qnn",
        "runtime": "npu",
        "platform": "windows",
    }


class TestCodegenEngine:
    def test_render_windows_python(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("windows", "python", basic_vars, sdk="qnn")
        assert "inference.py" in result.source_files
        assert "requirements.txt" in result.source_files
        assert result.language == "python"
        assert result.platform == "windows"

    def test_rendered_python_is_valid_syntax(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("windows", "python", basic_vars)
        python_code = result.source_files["inference.py"]
        # Should parse without error
        ast.parse(python_code)

    def test_render_windows_cpp(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("windows", "cpp", basic_vars, sdk="qnn")
        assert "inference.cpp" in result.source_files
        assert "CMakeLists.txt" in result.source_files
        assert result.language == "cpp"

    def test_render_linux_python(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("linux", "python", basic_vars, sdk="snpe")
        assert "inference.py" in result.source_files
        assert result.sdk == "snpe"

    def test_render_linux_arduino(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("linux", "arduino_sketch", basic_vars, sdk="snpe")
        assert "inference.ino" in result.source_files

    def test_render_android_kotlin(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("android", "kotlin", basic_vars, sdk="snpe")
        assert "InferenceEngine.kt" in result.source_files

    def test_render_android_jni(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("android", "jni", basic_vars, sdk="snpe")
        assert "inference_jni.cpp" in result.source_files

    def test_unsupported_language_raises(self, engine: CodegenEngine, basic_vars: dict) -> None:
        with pytest.raises(UnsupportedLanguageError):
            engine.render("windows", "rust", basic_vars)

    def test_build_instructions_included(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("windows", "python", basic_vars)
        assert "pip install" in result.build_instructions

    def test_dependencies_included(self, engine: CodegenEngine, basic_vars: dict) -> None:
        result = engine.render("windows", "python", basic_vars, sdk="qnn")
        assert any("qnn" in dep for dep in result.dependencies)

    def test_model_path_injected(self, engine: CodegenEngine) -> None:
        vars = {"model_path": "my_model.bin", "sdk": "qnn", "runtime": "npu"}
        result = engine.render("windows", "python", vars)
        assert "my_model.bin" in result.source_files["inference.py"]


class TestValidators:
    def test_valid_python(self) -> None:
        code = "x = 1\nprint(x)\n"
        errors = validate_output("test.py", code)
        assert errors == []

    def test_invalid_python_syntax(self) -> None:
        code = "def foo(\n"  # Incomplete
        errors = validate_output("test.py", code)
        assert len(errors) > 0
        assert "syntax error" in errors[0].lower()

    def test_empty_file_detected(self) -> None:
        errors = validate_output("test.py", "")
        assert any("empty" in e.lower() for e in errors)

    def test_unrendered_jinja_detected(self) -> None:
        code = "x = {{ value }}\n"
        errors = validate_output("test.py", code)
        assert any("jinja" in e.lower() or "placeholder" in e.lower() for e in errors)

    def test_balanced_braces_cpp(self) -> None:
        code = "int main() { return 0; }"
        errors = validate_output("test.cpp", code)
        assert errors == []

    def test_unbalanced_braces_cpp(self) -> None:
        code = "int main() { return 0;"
        errors = validate_output("test.cpp", code)
        assert any("brace" in e.lower() for e in errors)


# ─── T1.8: strengthened validator tests ────────────────────────────────────


class TestStrengthenedValidator:
    """The bracket-balance check used to accept code with empty function
    bodies and TODO scaffolds. T1.8 strengthens it."""

    def test_strip_string_literals_keeps_braces_outside_strings(self) -> None:
        src = 'auto x = "hello {world}";\nint main() { return 0; }'
        stripped = _strip_string_literals(src)
        # Braces in strings collapsed; braces outside preserved
        assert stripped.count("{") == 1
        assert stripped.count("}") == 1

    def test_strip_string_literals_no_multiline_gobble(self) -> None:
        # Multiple short strings on consecutive lines — the regex must
        # not match across newlines. Each line's strings stay independent.
        src = 'a = "x";\nb = 5;\nc = "y";\n'
        stripped = _strip_string_literals(src)
        # The "5" on its own line should be untouched
        assert "b = 5;" in stripped

    def test_strict_todos_off_by_default(self) -> None:
        cpp = "void f() { /* TODO: implement */ }"
        errors = validate_output("a.cpp", cpp)
        # TODOs should NOT fail by default
        assert not any("TODO" in e for e in errors)

    def test_strict_todos_flag_rejects_todos(self) -> None:
        cpp = "void f() { /* TODO: implement */ }"
        opts = ValidatorOptions(strict_todos=True)
        errors = validate_output("a.cpp", cpp, options=opts)
        assert any("TODO" in e or "FIXME" in e or "Strict mode" in e for e in errors)

    def test_strict_todos_via_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("QUAD_STRICT_TODOS", "1")
        # Re-import to pick up the new env var
        import importlib
        import quad.codegen.validators as validators
        importlib.reload(validators)
        cpp = "void f() { /* TODO: implement */ }"
        errors = validators.validate_output("a.cpp", cpp)
        assert any("TODO" in e or "Strict mode" in e for e in errors)
        # Reset for other tests
        monkeypatch.delenv("QUAD_STRICT_TODOS", raising=False)
        importlib.reload(validators)

    def test_unrendered_jinja_template_marker(self) -> None:
        cpp = """\
int main() {
    auto x = {{ undefined_variable }};
    return 0;
}
"""
        errors = validate_output("a.cpp", cpp)
        assert any("Jinja2" in e for e in errors)

    def test_balanced_braces_in_strings_accepted(self) -> None:
        # The new validator strips strings before counting; this used
        # to be a false-positive in the old implementation.
        cpp = 'int f() { auto s = "before { middle } after"; return 0; }'
        errors = validate_output("a.cpp", cpp)
        # No brace error — string contents were stripped
        assert not any("brace" in e.lower() for e in errors)

    def test_balanced_parens_in_strings_accepted(self) -> None:
        cpp = 'int f() { auto s = "(opening only"; return 0; }'
        errors = validate_output("a.cpp", cpp)
        assert not any("paren" in e.lower() for e in errors)


class TestWindowsCppTemplateHasRealQNNCalls:
    """T1.8: verify the Windows C++ template now has real QNN calls,
    not just // TODO scaffolds."""

    def test_template_calls_qnn_init(self) -> None:
        engine = CodegenEngine(template_dir=TEMPLATE_DIR)
        result = engine.render(
            "windows", "cpp",
            {"model_path": "m.bin", "sdk": "qnn", "sdk_path": "/x", "runtime": "npu"},
            sdk="qnn",
        )
        cpp = result.source_files["inference.cpp"]
        # Real QNN API calls should appear (not just commented-out TODOs)
        assert "QnnInterface_getProviders" in cpp
        assert "logCreate" in cpp
        assert "backendCreate" in cpp
        assert "deviceCreate" in cpp
        assert "contextCreate" in cpp
        assert "graphExecute" in cpp

    def test_template_handles_bin_models(self) -> None:
        engine = CodegenEngine(template_dir=TEMPLATE_DIR)
        result = engine.render(
            "windows", "cpp",
            {"model_path": "m.bin", "sdk": "qnn", "sdk_path": "/x", "runtime": "npu"},
            sdk="qnn",
        )
        cpp = result.source_files["inference.cpp"]
        assert "contextCreateFromBinary" in cpp
        assert "contextGetGraphHandles" in cpp

    def test_template_does_io_marshalling(self) -> None:
        engine = CodegenEngine(template_dir=TEMPLATE_DIR)
        result = engine.render(
            "windows", "cpp",
            {"model_path": "m.bin", "sdk": "qnn", "sdk_path": "/x", "runtime": "npu"},
            sdk="qnn",
        )
        cpp = result.source_files["inference.cpp"]
        # Real input loading + output writing
        assert "read_raw_file" in cpp
        assert "write_raw_file" in cpp
        assert "QNN_TENSOR_SET_CLIENT_BUF" in cpp
