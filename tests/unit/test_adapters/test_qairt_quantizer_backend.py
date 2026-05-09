"""Tests for the Sprint 4 ``qairt_quantizer`` backend.

This is the SDK-bundled real-quantization path. Subprocess invocation
is mocked so we don't require a model to exist on disk during unit
tests; the e2e test exercises the full pipeline against the real SDK.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quad.adapters.aimet_adapter import (
    AIMETAdapter,
    AIMETUnavailableError,
    QuantizationConfig,
    qairt_quantizer_available,
    select_backend,
)


# ─── availability detection ────────────────────────────────────────────────


class TestQairtQuantizerAvailability:
    def test_no_sdk_root_means_unavailable(self, monkeypatch) -> None:
        for v in ("QAIRT_SDK_ROOT", "QNN_SDK_ROOT", "SNPE_ROOT"):
            monkeypatch.delenv(v, raising=False)
        assert qairt_quantizer_available() is False

    def test_sdk_root_with_qairt_converter_is_available(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Build a fake SDK shape that resolve_sdk_root will accept
        bin_dir = tmp_path / "bin" / "x86_64-windows-msvc"
        bin_dir.mkdir(parents=True)
        (bin_dir / "qairt-converter").write_bytes(b"")
        (bin_dir / "snpe-net-run.exe").write_bytes(b"")
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(tmp_path))
        for v in ("QNN_SDK_ROOT", "SNPE_ROOT"):
            monkeypatch.delenv(v, raising=False)
        assert qairt_quantizer_available() is True


# ─── backend selection priority ────────────────────────────────────────────


class TestSelectBackend:
    def test_auto_picks_qairt_quantizer_when_only_sdk_present(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # No AIMET available, but a real SDK is — auto should pick qairt_quantizer
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.aimet_torch_available", lambda: False
        )
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.aimet_onnx_available", lambda: False
        )
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.qairt_quantizer_available", lambda: True
        )
        monkeypatch.delenv("QUAD_AIMET_BACKEND", raising=False)
        assert select_backend("auto") == "qairt_quantizer"

    def test_auto_falls_back_to_mock_when_nothing_available(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.aimet_torch_available", lambda: False
        )
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.aimet_onnx_available", lambda: False
        )
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.qairt_quantizer_available", lambda: False
        )
        monkeypatch.delenv("QUAD_AIMET_BACKEND", raising=False)
        assert select_backend("auto") == "mock"

    def test_explicit_qairt_quantizer_when_unavailable_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.qairt_quantizer_available", lambda: False
        )
        monkeypatch.delenv("QUAD_AIMET_BACKEND", raising=False)
        with pytest.raises(AIMETUnavailableError):
            select_backend("qairt_quantizer")

    def test_aimet_torch_still_preferred_over_qairt_quantizer(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.aimet_torch_available", lambda: True
        )
        monkeypatch.setattr(
            "quad.adapters.aimet_adapter.qairt_quantizer_available", lambda: True
        )
        monkeypatch.delenv("QUAD_AIMET_BACKEND", raising=False)
        assert select_backend("auto") == "aimet_torch"


# ─── quantization run with a mocked subprocess ─────────────────────────────


class TestQairtQuantizerRun:
    def _setup_fake_sdk(
        self, tmp_path: Path, monkeypatch, *, with_quantizer: bool = True
    ) -> tuple[Path, Path]:
        bin_dir = tmp_path / "sdk" / "bin" / "x86_64-windows-msvc"
        bin_dir.mkdir(parents=True)
        # Required to look like an SDK root
        (bin_dir / "qairt-converter").write_bytes(b"")
        (bin_dir / "snpe-net-run.exe").write_bytes(b"")
        if with_quantizer:
            ext = ".exe" if os.name == "nt" else ""
            (bin_dir / f"qairt-quantizer{ext}").write_bytes(b"")
        monkeypatch.setenv("QAIRT_SDK_ROOT", str(tmp_path / "sdk"))
        for v in ("QNN_SDK_ROOT", "SNPE_ROOT"):
            monkeypatch.delenv(v, raising=False)
        return tmp_path / "sdk", bin_dir

    def test_dlc_quantizes_via_subprocess(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        sdk_root, bin_dir = self._setup_fake_sdk(tmp_path, monkeypatch)

        src = tmp_path / "model.dlc"
        src.write_bytes(b"FAKE_FP32_DLC_CONTENTS_x" * 100)
        out = tmp_path / "model_int8.dlc"

        # Stub the subprocess to "produce" the output and return success.
        def fake_run_command(cmd, timeout):  # type: ignore[no-untyped-def]
            # Simulate qairt-quantizer writing the output dlc
            out.write_bytes(b"FAKE_QUANTIZED_INT8_DLC")
            cp = subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")
            async def _runner() -> subprocess.CompletedProcess:
                return cp
            return _runner()

        monkeypatch.setattr(
            "quad.adapters.qairt_adapter._run_command", fake_run_command
        )
        # Stub create_input_list_for_model so we don't need real model_io
        def _fake_input_list(model_path, *, sdk_root=None, num_samples=1, calibration_data=None):
            list_path = tmp_path / "input_list.txt"
            list_path.write_text("input.raw\n")
            from quad.adapters.model_inputs import ModelIO
            return str(list_path), ModelIO(inputs=[], source="test")
        monkeypatch.setattr(
            "quad.adapters.model_inputs.create_input_list_for_model", _fake_input_list
        )

        adapter = AIMETAdapter(backend="qairt_quantizer", strict=True)
        result = adapter.quantize(src, output_path=out, config=QuantizationConfig(bitwidth=8))

        assert out.exists()
        assert out.read_bytes() == b"FAKE_QUANTIZED_INT8_DLC"
        assert result.backend == "qairt_quantizer"
        assert result.bitwidth == 8
        assert "qairt-quantizer" in " ".join(result.notes)

    def test_int4_uses_per_row_quantization(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        sdk_root, bin_dir = self._setup_fake_sdk(tmp_path, monkeypatch)
        src = tmp_path / "m.dlc"
        src.write_bytes(b"X")
        out = tmp_path / "m_int4.dlc"

        captured: list[list[str]] = []

        def fake_run_command(cmd, timeout):  # type: ignore[no-untyped-def]
            captured.append(list(cmd))
            out.write_bytes(b"INT4")
            cp = subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            async def _runner() -> subprocess.CompletedProcess:
                return cp
            return _runner()

        monkeypatch.setattr("quad.adapters.qairt_adapter._run_command", fake_run_command)
        def _fake_input_list(model_path, *, sdk_root=None, num_samples=1, calibration_data=None):
            list_path = tmp_path / "input_list.txt"
            list_path.write_text("input.raw\n")
            from quad.adapters.model_inputs import ModelIO
            return str(list_path), ModelIO(inputs=[], source="test")
        monkeypatch.setattr(
            "quad.adapters.model_inputs.create_input_list_for_model", _fake_input_list
        )

        adapter = AIMETAdapter(backend="qairt_quantizer", strict=True)
        adapter.quantize(
            src, output_path=out,
            config=QuantizationConfig(bitwidth=4, scheme="symmetric_per_channel"),
        )

        # The INT4 path must include the block-quant flags
        cmd = captured[0]
        assert "--bitwidth" in cmd
        assert "4" in cmd
        assert "--use_per_row_quantization" in cmd
        assert "--per_row_block_size" in cmd

    def test_subprocess_failure_raises(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from quad.exceptions import QuantizationError

        sdk_root, bin_dir = self._setup_fake_sdk(tmp_path, monkeypatch)
        src = tmp_path / "m.dlc"
        src.write_bytes(b"X")
        out = tmp_path / "m_int8.dlc"

        def fake_run_command(cmd, timeout):  # type: ignore[no-untyped-def]
            cp = subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="something broke\n",
            )
            async def _runner() -> subprocess.CompletedProcess:
                return cp
            return _runner()

        monkeypatch.setattr("quad.adapters.qairt_adapter._run_command", fake_run_command)
        def _fake_input_list(model_path, *, sdk_root=None, num_samples=1, calibration_data=None):
            list_path = tmp_path / "input_list.txt"
            list_path.write_text("input.raw\n")
            from quad.adapters.model_inputs import ModelIO
            return str(list_path), ModelIO(inputs=[], source="test")
        monkeypatch.setattr(
            "quad.adapters.model_inputs.create_input_list_for_model", _fake_input_list
        )

        adapter = AIMETAdapter(backend="qairt_quantizer", strict=True)
        with pytest.raises(QuantizationError):
            adapter.quantize(src, output_path=out)
