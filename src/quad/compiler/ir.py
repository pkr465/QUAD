"""QUAD IR — Portable Intermediate Representation.

The QUAD IR is a device-agnostic representation of a computation graph:
compile once, JIT to any target at load time. The same QIR can be
finalised for CPU, GPU (Adreno), or NPU (Hexagon HTP) when the model
is loaded on a specific device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class IRTensor:
    """Tensor descriptor in the IR."""
    name: str
    shape: list[int]
    dtype: str = "float32"


@dataclass
class IRNode:
    """Single operation in the computation graph."""
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IRNode:
        return cls(
            name=d["name"],
            op_type=d["op_type"],
            inputs=d["inputs"],
            outputs=d["outputs"],
            attributes=d.get("attributes", {}),
        )


@dataclass
class IRGraph:
    """Complete computation graph in QUAD IR format."""
    name: str
    nodes: list[IRNode] = field(default_factory=list)
    inputs: list[IRTensor] = field(default_factory=list)
    outputs: list[IRTensor] = field(default_factory=list)
    opset_version: int = 1

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_params(self) -> int:
        """Estimate parameter count from tensor shapes."""
        total = 0
        for node in self.nodes:
            for attr_val in node.attributes.values():
                if isinstance(attr_val, list) and all(isinstance(x, int) for x in attr_val):
                    prod = 1
                    for x in attr_val:
                        prod *= x
                    total += prod
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "quad_ir",
            "version": "1.0",
            "name": self.name,
            "opset_version": self.opset_version,
            "inputs": [{"name": t.name, "shape": t.shape, "dtype": t.dtype} for t in self.inputs],
            "outputs": [{"name": t.name, "shape": t.shape, "dtype": t.dtype} for t in self.outputs],
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IRGraph:
        graph = cls(
            name=d["name"],
            opset_version=d.get("opset_version", 1),
        )
        graph.inputs = [IRTensor(**t) for t in d.get("inputs", [])]
        graph.outputs = [IRTensor(**t) for t in d.get("outputs", [])]
        graph.nodes = [IRNode.from_dict(n) for n in d.get("nodes", [])]
        return graph


class QuadIR:
    """QUAD IR file format — serialization/deserialization.

    File extension: .qir
    Format: JSON (human-readable) or MessagePack (binary, future)
    """

    @staticmethod
    def serialize(graph: IRGraph) -> str:
        """Serialize IR graph to JSON string."""
        return json.dumps(graph.to_dict(), indent=2)

    @staticmethod
    def deserialize(data: str) -> IRGraph:
        """Deserialize JSON string to IR graph."""
        d = json.loads(data)
        if d.get("format") != "quad_ir":
            raise ValueError(f"Not a QUAD IR file (format: {d.get('format')})")
        return IRGraph.from_dict(d)

    @staticmethod
    def save(graph: IRGraph, path: str) -> None:
        """Save IR graph to file."""
        with open(path, "w") as f:
            f.write(QuadIR.serialize(graph))

    @staticmethod
    def load(path: str) -> IRGraph:
        """Load IR graph from file."""
        with open(path, "r") as f:
            return QuadIR.deserialize(f.read())
