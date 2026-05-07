"""Tests for QUAD Compiler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from quad.compiler import (
    ComputeCapability,
    IRGraph,
    IRNode,
    QBin,
    QuadIR,
    compile_model,
    compile_onnx,
    get_capability,
)
from quad.compiler.capabilities import list_capabilities, get_capability_for_chipset


class TestIR:
    def test_create_graph(self) -> None:
        graph = IRGraph(name="test_model")
        assert graph.name == "test_model"
        assert graph.num_nodes == 0

    def test_add_nodes(self) -> None:
        graph = IRGraph(name="test")
        graph.nodes.append(IRNode(
            name="conv1", op_type="Conv", inputs=["input"], outputs=["conv1_out"]
        ))
        assert graph.num_nodes == 1

    def test_serialize_roundtrip(self) -> None:
        graph = IRGraph(name="roundtrip_test")
        graph.nodes.append(IRNode(
            name="relu", op_type="Relu", inputs=["x"], outputs=["y"],
            attributes={"inplace": 1}
        ))
        # Serialize
        json_str = QuadIR.serialize(graph)
        assert "quad_ir" in json_str
        # Deserialize
        restored = QuadIR.deserialize(json_str)
        assert restored.name == "roundtrip_test"
        assert restored.num_nodes == 1
        assert restored.nodes[0].op_type == "Relu"

    def test_save_and_load(self) -> None:
        graph = IRGraph(name="file_test")
        graph.nodes.append(IRNode(
            name="fc", op_type="Gemm", inputs=["x"], outputs=["y"]
        ))
        with tempfile.NamedTemporaryFile(suffix=".qir", mode="w", delete=False) as f:
            QuadIR.save(graph, f.name)
            loaded = QuadIR.load(f.name)
        assert loaded.name == "file_test"
        assert loaded.num_nodes == 1

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Not a QUAD IR"):
            QuadIR.deserialize('{"format": "not_quad"}')


class TestQBin:
    def test_create_qbin(self) -> None:
        qbin = QBin(name="test_model")
        assert qbin.name == "test_model"
        assert qbin.num_targets == 0

    def test_add_targets(self) -> None:
        qbin = QBin(name="test")
        qbin.add_target("qnpu_v3", "qnn", b"binary_data")
        assert qbin.has_target("qnpu_v3")
        assert qbin.num_targets == 1
        assert qbin.total_size_bytes == 11

    def test_get_best_target(self) -> None:
        qbin = QBin(name="test")
        qbin.add_target("qnpu_v3", "qnn")
        qbin.add_target("qdsp_v66", "snpe")
        best = qbin.get_best_target("qnpu_v3")
        assert best is not None
        assert best.target == "qnpu_v3"

    def test_get_best_target_fallback(self) -> None:
        qbin = QBin(name="test")
        qbin.add_target("qnpu_v3", "qnn")
        best = qbin.get_best_target("unknown_target")
        assert best is not None  # Falls back to any NPU target

    def test_save_and_load(self) -> None:
        qbin = QBin(name="persist_test")
        qbin.add_target("qnpu_v3", "qnn")
        qbin.ir = IRGraph(name="persist_test")
        qbin.ir.nodes.append(IRNode(
            name="conv", op_type="Conv", inputs=["x"], outputs=["y"]
        ))
        with tempfile.NamedTemporaryFile(suffix=".qbin", mode="w", delete=False) as f:
            qbin.save(f.name)
            loaded = QBin.load(f.name)
        assert loaded.name == "persist_test"
        assert loaded.has_target("qnpu_v3")
        assert loaded.ir is not None
        assert loaded.ir.num_nodes == 1


class TestCapabilities:
    def test_get_known_capability(self) -> None:
        cap = get_capability("qnpu_v3")
        assert cap.chipset == "Snapdragon X Elite"
        assert cap.npu_tops == 45.0
        assert cap.supports_int4

    def test_unknown_capability_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown compute capability"):
            get_capability("nonexistent")

    def test_list_capabilities(self) -> None:
        caps = list_capabilities()
        assert len(caps) >= 4
        names = [c.name for c in caps]
        assert "qnpu_v3" in names
        assert "qdsp_v66" in names

    def test_get_by_chipset(self) -> None:
        cap = get_capability_for_chipset("Snapdragon X Elite")
        assert cap is not None
        assert cap.name == "qnpu_v3"

    def test_get_by_chipset_partial(self) -> None:
        cap = get_capability_for_chipset("QCS2210")
        assert cap is not None
        assert cap.name == "qdsp_v66"


class TestFrontendONNX:
    def test_compile_generates_ir(self) -> None:
        ir = compile_onnx("mobilenetv2.onnx")
        assert ir.name == "mobilenetv2"
        assert ir.num_nodes > 0
        assert len(ir.inputs) > 0
        assert len(ir.outputs) > 0

    def test_nodes_have_correct_structure(self) -> None:
        ir = compile_onnx("resnet50.onnx")
        for node in ir.nodes:
            assert node.name
            assert node.op_type
            assert len(node.inputs) > 0
            assert len(node.outputs) > 0


class TestCompilePipeline:
    def test_compile_to_all_targets(self) -> None:
        qbin = compile_model("model.onnx")
        assert qbin.name == "model"
        assert qbin.ir is not None
        assert qbin.num_targets >= 4  # All known capabilities

    def test_compile_specific_targets(self) -> None:
        qbin = compile_model("model.onnx", targets=["qnpu_v3", "qdsp_v66"])
        assert qbin.num_targets == 2
        assert qbin.has_target("qnpu_v3")
        assert qbin.has_target("qdsp_v66")

    def test_compile_portable(self) -> None:
        qbin = compile_model("model.onnx", portable=True)
        assert qbin.ir is not None
        assert qbin.num_targets == 0  # No pre-compiled targets

    def test_compile_and_save(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".qbin", delete=False) as f:
            qbin = compile_model("model.onnx", output_path=f.name)
            loaded = QBin.load(f.name)
        assert loaded.name == qbin.name
        assert loaded.num_targets == qbin.num_targets

    def test_metadata_populated(self) -> None:
        qbin = compile_model("resnet50.onnx")
        assert qbin.metadata["source_format"] == "onnx"
        assert qbin.metadata["num_nodes"] > 0

    def test_unsupported_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported source format"):
            compile_model("model.tflite")
