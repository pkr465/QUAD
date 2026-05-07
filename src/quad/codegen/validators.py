"""Output validators for generated code."""

from __future__ import annotations

import ast
import re


def validate_output(filename: str, content: str) -> list[str]:
    """Validate generated code output for common issues.

    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Check for empty output
    if not content or not content.strip():
        errors.append("Generated file is empty")
        return errors

    # Check for unrendered Jinja2 placeholders
    if "{{" in content or "}}" in content or "{%" in content:
        errors.append("Contains unrendered Jinja2 placeholders")

    # Language-specific validation
    if filename.endswith(".py"):
        errors.extend(_validate_python(content))
    elif filename.endswith((".cpp", ".c", ".h", ".ino")):
        errors.extend(_validate_c_cpp(content))
    elif filename.endswith((".kt", ".java")):
        errors.extend(_validate_jvm(content))
    elif filename.endswith(".txt") and "requirements" in filename.lower():
        pass  # No special validation for requirements.txt
    elif filename.endswith((".cmake", "CMakeLists.txt")):
        pass  # CMake files don't need syntax validation

    return errors


def _validate_python(content: str) -> list[str]:
    """Validate Python syntax using ast.parse()."""
    errors: list[str] = []
    try:
        ast.parse(content)
    except SyntaxError as e:
        errors.append(f"Python syntax error at line {e.lineno}: {e.msg}")
    return errors


def _validate_c_cpp(content: str) -> list[str]:
    """Validate C/C++ basic structure (bracket balance)."""
    errors: list[str] = []

    # Check bracket/brace balance
    opens = content.count("{")
    closes = content.count("}")
    if opens != closes:
        errors.append(f"Unbalanced braces: {opens} open, {closes} close")

    # Check parenthesis balance
    opens = content.count("(")
    closes = content.count(")")
    if opens != closes:
        errors.append(f"Unbalanced parentheses: {opens} open, {closes} close")

    return errors


def _validate_jvm(content: str) -> list[str]:
    """Validate Kotlin/Java basic structure."""
    errors: list[str] = []

    # Check bracket balance
    opens = content.count("{")
    closes = content.count("}")
    if opens != closes:
        errors.append(f"Unbalanced braces: {opens} open, {closes} close")

    return errors
