"""ONNX Frontend — converts ONNX models to QUAD IR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quad.compiler.ir import IRGraph, IRNode, IRTensor


def compile_onnx(model_path: str) -> IRGraph:
    """Convert ONNX model to QUAD IR.

    In mock mode: generates a representative IR graph based on common model architectures.
    In real mode: parses actual ONNX protobuf and converts each op to IR node.

    Args:
        model_path: Path to .onnx file

    Returns:
        IRGraph representation of the model
    """
    path = Path(model_path)
    model_name = path.stem

    # Try to load real ONNX if available
    try:
        import onnx
        return _parse_real_onnx(model_path)
    except ImportError:
        pass

    # Mock: generate representative IR based on model name
    return _generate_mock_ir(model_name)


def _parse_real_onnx(model_path: str) -> IRGraph:
    """Parse real ONNX model into QUAD IR."""
    import onnx

    model = onnx.load(model_path)
    graph = model.graph

    ir_graph = IRGraph(name=graph.name or Path(model_path).stem)

    # Parse inputs
    for inp in graph.input:
        shape = []
        if inp.type.tensor_type.shape:
            for dim in inp.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        ir_graph.inputs.append(IRTensor(name=inp.name, shape=shape))

    # Parse outputs
    for out in graph.output:
        shape = []
        if out.type.tensor_type.shape:
            for dim in out.type.tensor_type.shape.dim:
                shape.append(dim.dim_value if dim.dim_value > 0 else 1)
        ir_graph.outputs.append(IRTensor(name=out.name, shape=shape))

    # Parse nodes
    for node in graph.node:
        attrs = {}
        for attr in node.attribute:
            if attr.type == 1:  # FLOAT
                attrs[attr.name] = attr.f
            elif attr.type == 2:  # INT
                attrs[attr.name] = attr.i
            elif attr.type == 7:  # INTS
                attrs[attr.name] = list(attr.ints)
        ir_graph.nodes.append(IRNode(
            name=node.name or f"{node.op_type}_{len(ir_graph.nodes)}",
            op_type=node.op_type,
            inputs=list(node.input),
            outputs=list(node.output),
            attributes=attrs,
        ))

    return ir_graph


def _generate_mock_ir(model_name: str) -> IRGraph:
    """Generate a mock IR graph for testing without ONNX dependency."""
    graph = IRGraph(name=model_name)
    graph.inputs = [IRTensor(name="input", shape=[1, 3, 224, 224])]
    graph.outputs = [IRTensor(name="output", shape=[1, 1000])]

    # Generate representative layers
    layers = [
        ("conv1", "Conv", {"kernel_shape": [7, 7], "strides": [2, 2]}),
        ("bn1", "BatchNormalization", {}),
        ("relu1", "Relu", {}),
        ("pool1", "MaxPool", {"kernel_shape": [3, 3], "strides": [2, 2]}),
        ("conv2", "Conv", {"kernel_shape": [3, 3]}),
        ("bn2", "BatchNormalization", {}),
        ("relu2", "Relu", {}),
        ("conv3", "Conv", {"kernel_shape": [3, 3]}),
        ("bn3", "BatchNormalization", {}),
        ("relu3", "Relu", {}),
        ("avgpool", "GlobalAveragePool", {}),
        ("flatten", "Flatten", {}),
        ("fc", "Gemm", {}),
        ("softmax", "Softmax", {}),
    ]

    prev_output = "input"
    for name, op_type, attrs in layers:
        output = f"{name}_out"
        graph.nodes.append(IRNode(
            name=name,
            op_type=op_type,
            inputs=[prev_output],
            outputs=[output],
            attributes=attrs,
        ))
        prev_output = output

    return graph
