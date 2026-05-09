"""Tests for the Sprint 3 real compiler backend (Path A).

Uses a fake adapter so we don't depend on the real qairt-converter
during unit testing. The e2e test exercises the real subprocess.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from quad.compiler import qairt_backend
from quad.compiler.qairt_backend import (
    QairtBackendResult,
    cache_dir,
    cache_key,
    clear_cache,
    compile_with_qairt,
    is_qairt_available,
)


# ─── A fake adapter that pretends qairt-converter ran ──────────────────────


class _FakeAdapter:
    def __init__(self, binary_bytes: bytes = b"FAKE_DLC_BYTES") -> None:
        self._bytes = binary_bytes
        self.calls: list[Any] = []

    async def convert_model(self, request: Any) -> Any:
        self.calls.append(request)
        # Write a fake .dlc next to the source so the file-existence
        # check in compile_with_qairt passes.
        src = Path(request.model_path)
        out = src.with_suffix(".dlc")
        out.write_bytes(self._bytes)

        from quad.models.conversion import ConversionResult
        return ConversionResult(
            output_path=str(out),
            model_size_mb=len(self._bytes) / (1024 * 1024),
            original_size_mb=src.stat().st_size / (1024 * 1024) if src.exists() else 0.1,
            compression_ratio=1.0,
            supported_ops_pct=100.0,
            unsupported_ops=[],
            quantization_applied=request.quantization,
            conversion_time_s=0.1,
            target_sdk=request.target_sdk,
            warnings=[],
            conversion_notes=["fake adapter run"],
            image_format_notes=[],
        )


def _build_tiny_real_onnx(path: Path, *, content_tag: bytes = b"\x00") -> Path:
    """Write a minimal but valid ONNX model so the real frontend parses it.

    A 1-input identity graph is enough for the compiler to populate
    its IR. The ``content_tag`` is appended as a trailing constant
    initializer so we can vary the file content for cache tests.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    identity = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    initializer = numpy_helper.from_array(
        np.frombuffer(content_tag.ljust(4, b"\x00"), dtype=np.uint8).astype(np.int32),
        name="tag",
    )
    graph = helper.make_graph([identity], "tiny", [inp], [out], [initializer])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 7
    onnx.save(model, str(path))
    return path


@pytest.fixture
def tiny_onnx(tmp_path: Path) -> Path:
    return _build_tiny_real_onnx(tmp_path / "tiny.onnx", content_tag=b"\x01")


# ─── is_qairt_available env-var detection ──────────────────────────────────


class TestAvailability:
    def test_no_env_var_says_unavailable(self, monkeypatch) -> None:
        for var in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT"):
            monkeypatch.delenv(var, raising=False)
        assert is_qairt_available() is False

    def test_env_var_pointing_at_existing_dir_is_available(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(tmp_path))
        assert is_qairt_available() is True

    def test_env_var_pointing_at_missing_dir_not_available(
        self, monkeypatch
    ) -> None:
        # Clear all three so a previously-set QNN_SDK_ROOT can't shadow.
        monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
        monkeypatch.delenv("SNPE_ROOT", raising=False)
        monkeypatch.setenv("QAIRT_SDK_ROOT", "/path/that/does/not/exist")
        assert is_qairt_available() is False


# ─── Cache hash + key ──────────────────────────────────────────────────────


class TestCacheKey:
    def test_same_content_same_key(self, tmp_path: Path) -> None:
        a = tmp_path / "a.onnx"
        a.write_bytes(b"abc")
        b = tmp_path / "b.onnx"
        b.write_bytes(b"abc")
        assert cache_key(a, "qnn", "fp32") == cache_key(b, "qnn", "fp32")

    def test_different_quantization_different_key(self, tmp_path: Path) -> None:
        a = tmp_path / "a.onnx"
        a.write_bytes(b"abc")
        assert cache_key(a, "qnn", "fp32") != cache_key(a, "qnn", "int8")

    def test_different_target_different_key(self, tmp_path: Path) -> None:
        a = tmp_path / "a.onnx"
        a.write_bytes(b"abc")
        assert cache_key(a, "qnn", "fp32") != cache_key(a, "snpe", "fp32")


# ─── compile_with_qairt happy path + cache ────────────────────────────────


class TestCompileWithQairt:
    def test_first_compile_invokes_adapter(
        self, tmp_path: Path, monkeypatch, tiny_onnx: Path
    ) -> None:
        monkeypatch.setenv("QUAD_COMPILE_CACHE_DIR", str(tmp_path / "cache"))
        adapter = _FakeAdapter(binary_bytes=b"COMPILED_v1")

        result = compile_with_qairt(
            tiny_onnx, target_sdk="qnn", quantization="fp32",
            project_root=tmp_path, adapter=adapter,
        )
        assert result.cache_hit is False
        assert result.binary == b"COMPILED_v1"
        assert len(adapter.calls) == 1
        assert adapter.calls[0].source_format == "onnx"
        assert adapter.calls[0].target_sdk == "qnn"
        assert adapter.calls[0].quantization == "fp32"

    def test_second_compile_hits_cache(
        self, tmp_path: Path, monkeypatch, tiny_onnx: Path
    ) -> None:
        monkeypatch.setenv("QUAD_COMPILE_CACHE_DIR", str(tmp_path / "cache"))
        adapter = _FakeAdapter(binary_bytes=b"COMPILED_v2")

        # First compile populates cache
        first = compile_with_qairt(
            tiny_onnx, project_root=tmp_path, adapter=adapter,
        )
        # Second compile with the same content should not re-invoke adapter
        second_adapter = _FakeAdapter(binary_bytes=b"WOULD_NOT_BE_USED")
        second = compile_with_qairt(
            tiny_onnx, project_root=tmp_path, adapter=second_adapter,
        )

        assert first.cache_hit is False
        assert second.cache_hit is True
        assert second.binary == first.binary  # served from cache
        assert second_adapter.calls == []

    def test_use_cache_false_forces_recompile(
        self, tmp_path: Path, monkeypatch, tiny_onnx: Path
    ) -> None:
        monkeypatch.setenv("QUAD_COMPILE_CACHE_DIR", str(tmp_path / "cache"))
        a1 = _FakeAdapter(b"FIRST")
        compile_with_qairt(tiny_onnx, project_root=tmp_path, adapter=a1)
        a2 = _FakeAdapter(b"SECOND")
        result = compile_with_qairt(
            tiny_onnx, project_root=tmp_path, adapter=a2, use_cache=False
        )
        assert result.cache_hit is False
        assert result.binary == b"SECOND"
        assert len(a2.calls) == 1

    def test_changed_content_invalidates_cache(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("QUAD_COMPILE_CACHE_DIR", str(tmp_path / "cache"))
        src = tmp_path / "m.onnx"
        # qairt_backend just hashes the source file — we don't need a
        # valid ONNX here, only different bytes between the two runs.
        src.write_bytes(b"version-A")
        a1 = _FakeAdapter(b"A_BIN")
        r1 = compile_with_qairt(src, project_root=tmp_path, adapter=a1)
        # Edit the source — content hash changes, cache must miss
        src.write_bytes(b"version-B-different")
        a2 = _FakeAdapter(b"B_BIN")
        r2 = compile_with_qairt(src, project_root=tmp_path, adapter=a2)
        assert r1.cache_hit is False
        assert r2.cache_hit is False
        assert r2.binary == b"B_BIN"

    def test_clear_cache(self, tmp_path: Path, monkeypatch, tiny_onnx: Path) -> None:
        monkeypatch.setenv("QUAD_COMPILE_CACHE_DIR", str(tmp_path / "cache"))
        compile_with_qairt(tiny_onnx, project_root=tmp_path, adapter=_FakeAdapter())
        assert clear_cache(project_root=tmp_path) >= 1


# ─── compile_model integration ─────────────────────────────────────────────


class TestCompileModelDispatch:
    def test_backend_stub_when_no_sdk(
        self, tmp_path: Path, monkeypatch, tiny_onnx: Path
    ) -> None:
        from quad.compiler.pipeline import BackendNotImplementedError, compile_model
        for v in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.delenv("QUAD_PLACEHOLDER_BACKEND", raising=False)
        with pytest.raises(BackendNotImplementedError):
            compile_model(str(tiny_onnx), backend="auto", targets=["qnpu_v3"])

    def test_backend_qairt_explicit_invokes_qairt(
        self, tmp_path: Path, monkeypatch, tiny_onnx: Path
    ) -> None:
        from quad.compiler.pipeline import compile_model

        # Patch compile_with_qairt so we don't need a real adapter
        called: dict[str, Any] = {}

        def _fake_compile(model_path, *, target_sdk, quantization, use_cache, project_root=None, adapter=None):
            called["model_path"] = str(model_path)
            called["target_sdk"] = target_sdk
            called["quantization"] = quantization
            return QairtBackendResult(
                target=target_sdk,
                target_sdk=target_sdk,
                quantization=quantization,
                binary=b"REAL_DLC",
                binary_format=".dlc",
                output_path="/tmp/x.dlc",
                cache_hit=False,
                conversion_notes=[],
                supported_ops_pct=100.0,
                unsupported_ops=[],
                duration_s=0.01,
            )

        monkeypatch.setattr(qairt_backend, "compile_with_qairt", _fake_compile)
        # Re-import the symbol used inside pipeline.py
        from quad.compiler import pipeline as pl
        monkeypatch.setattr(pl, "BackendNotImplementedError", pl.BackendNotImplementedError)

        # Force the auto path to qairt
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(tmp_path))

        qbin = compile_model(str(tiny_onnx), backend="qairt", targets=["qnpu_v3"])
        # The compiled binary must be present in QBin
        assert qbin.has_target("qnpu_v3")
        target_bin = qbin.targets["qnpu_v3"]
        assert target_bin.data == b"REAL_DLC"
        # Metadata records the backend run
        assert "qairt_backend" in qbin.metadata
        assert qbin.metadata["qairt_backend"]["qnpu_v3"]["target_sdk"] == "qnn"
