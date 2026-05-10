"""Unit tests for quad.model_registry."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from quad.model_registry import (
    ModelEntry,
    ModelFetchError,
    fetch_model,
    list_for_plan,
    list_models,
    register_entry,
    resolve_entry,
    resolve_model_path,
)
from quad.model_registry.manifest import clear_runtime_entries


# ── ModelEntry validation ──────────────────────────────────────────────────


def test_entry_must_have_url_or_env():
    with pytest.raises(ValueError):
        ModelEntry(name="x", plan="test", description="d")


def test_entry_cannot_have_both_url_and_env():
    with pytest.raises(ValueError):
        ModelEntry(name="x", plan="test", description="d",
                   url="https://example.com/m.onnx", path_env_var="X")


def test_entry_output_filename_url_basename():
    e = ModelEntry(name="x", plan="test", description="",
                   url="https://example.com/path/to/foo.onnx")
    assert e.output_filename == "foo.onnx"


def test_entry_output_filename_env_default():
    e = ModelEntry(name="bar", plan="test", description="", path_env_var="BAR")
    assert e.output_filename == "bar.onnx"


def test_entry_explicit_filename_wins():
    e = ModelEntry(name="x", plan="test", description="",
                   url="https://example.com/m.onnx", filename="custom.onnx")
    assert e.output_filename == "custom.onnx"


# ── Manifest loading ───────────────────────────────────────────────────────


def test_yaml_registry_has_required_plans():
    """The shipped registry should cover every plan that's not Plan 3."""
    plan1 = list_for_plan("plan1")
    plan2 = list_for_plan("plan2")
    plan4 = list_for_plan("plan4")
    assert any(e.name == "mobilenetv2" for e in plan1)
    assert any(e.path_env_var == "LLAMA3_8B_PREFILL_ONNX" for e in plan2)
    assert any(e.name.startswith("whisper_tiny") for e in plan4)


def test_resolve_entry_friendly_error_on_miss():
    with pytest.raises(KeyError) as excinfo:
        resolve_entry("does-not-exist")
    msg = str(excinfo.value)
    assert "does-not-exist" in msg
    assert "Known:" in msg


def test_register_entry_adds_runtime_entry():
    clear_runtime_entries()
    e = ModelEntry(
        name="future_plan_xyz",
        plan="planX",
        description="hypothetical model",
        url="https://example.com/x.onnx",
    )
    register_entry(e)
    try:
        names = {m.name for m in list_models()}
        assert "future_plan_xyz" in names
        assert resolve_entry("future_plan_xyz").plan == "planX"
    finally:
        clear_runtime_entries()


def test_register_entry_replace_flag():
    clear_runtime_entries()
    e1 = ModelEntry(name="dup", plan="p", description="v1",
                    url="https://example.com/a.onnx")
    e2 = ModelEntry(name="dup", plan="p", description="v2",
                    url="https://example.com/b.onnx")
    register_entry(e1)
    try:
        with pytest.raises(ValueError):
            register_entry(e2)
        register_entry(e2, replace=True)
        assert resolve_entry("dup").description == "v2"
    finally:
        clear_runtime_entries()


# ── Fetcher ────────────────────────────────────────────────────────────────


def test_path_env_var_unset_raises_helpful_error(monkeypatch):
    clear_runtime_entries()
    e = ModelEntry(
        name="unset_test",
        plan="test",
        description="",
        path_env_var="QUAD_TEST_UNSET_VAR",
    )
    register_entry(e)
    monkeypatch.delenv("QUAD_TEST_UNSET_VAR", raising=False)
    try:
        with pytest.raises(ModelFetchError) as excinfo:
            fetch_model("unset_test")
        assert "QUAD_TEST_UNSET_VAR" in str(excinfo.value)
    finally:
        clear_runtime_entries()


def test_path_env_var_set_returns_resolved_path(tmp_path, monkeypatch):
    clear_runtime_entries()
    fake_model = tmp_path / "fake.onnx"
    fake_model.write_bytes(b"\x00\x01\x02")
    e = ModelEntry(
        name="set_test",
        plan="test",
        description="",
        path_env_var="QUAD_TEST_SET_VAR",
    )
    register_entry(e)
    monkeypatch.setenv("QUAD_TEST_SET_VAR", str(fake_model))
    try:
        path = fetch_model("set_test")
        assert path == fake_model.resolve()
    finally:
        clear_runtime_entries()


def test_path_env_var_set_but_missing_file(tmp_path, monkeypatch):
    clear_runtime_entries()
    e = ModelEntry(
        name="missing_test",
        plan="test",
        description="",
        path_env_var="QUAD_TEST_MISSING_VAR",
    )
    register_entry(e)
    monkeypatch.setenv("QUAD_TEST_MISSING_VAR", str(tmp_path / "no.onnx"))
    try:
        with pytest.raises(ModelFetchError) as excinfo:
            fetch_model("missing_test")
        assert "does not exist" in str(excinfo.value)
    finally:
        clear_runtime_entries()


def test_url_entry_uses_cache_when_present(tmp_path, monkeypatch):
    """When the cached file exists we don't re-download."""
    clear_runtime_entries()
    monkeypatch.setenv("QUAD_MODELS_DIR", str(tmp_path))
    body = b"hello-onnx-bytes"
    e = ModelEntry(
        name="cached_test",
        plan="test",
        description="",
        url="https://example.invalid/m.onnx",
        sha256=hashlib.sha256(body).hexdigest(),
    )
    register_entry(e)
    cached_dir = tmp_path / "cached_test"
    cached_dir.mkdir()
    cached_path = cached_dir / "m.onnx"
    cached_path.write_bytes(body)
    try:
        # If the cache works, fetch_model returns without making a request.
        path = fetch_model("cached_test")
        assert path == cached_path
        assert path.read_bytes() == body
    finally:
        clear_runtime_entries()


def test_url_entry_sha_mismatch_raises(tmp_path, monkeypatch):
    clear_runtime_entries()
    monkeypatch.setenv("QUAD_MODELS_DIR", str(tmp_path))
    body = b"original"
    wrong_sha = "0" * 64
    e = ModelEntry(
        name="badsha_test",
        plan="test",
        description="",
        url="https://example.invalid/m.onnx",
        sha256=wrong_sha,
    )
    register_entry(e)
    cached_dir = tmp_path / "badsha_test"
    cached_dir.mkdir()
    (cached_dir / "m.onnx").write_bytes(body)
    try:
        # Cached file has wrong SHA; without --force it falls through to
        # re-download (which will fail because the URL is invalid). The
        # fetcher should surface either a network error or a SHA error.
        with pytest.raises(Exception):
            fetch_model("badsha_test")
    finally:
        clear_runtime_entries()


def test_resolve_model_path_url_no_download(tmp_path, monkeypatch):
    """resolve_model_path returns the cache path even if not downloaded."""
    clear_runtime_entries()
    monkeypatch.setenv("QUAD_MODELS_DIR", str(tmp_path))
    e = ModelEntry(
        name="resolve_test",
        plan="test",
        description="",
        url="https://example.invalid/m.onnx",
    )
    register_entry(e)
    try:
        path = resolve_model_path("resolve_test")
        assert path == tmp_path / "resolve_test" / "m.onnx"
        assert not path.exists()  # not downloaded yet
    finally:
        clear_runtime_entries()
