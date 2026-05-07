"""Graph optimization passes for QUAD IR.

Each pass transforms an IRGraph by fusing, eliminating, or annotating nodes
to produce more efficient execution on target hardware.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from quad.compiler.ir import IRGraph, IRNode, IRTensor


@dataclass
class PassStats:
    """Statistics collected during a pass execution."""

    nodes_removed: int = 0
    nodes_fused: int = 0
    nodes_annotated: int = 0


class OptimizationPass(ABC):
    """Base class for all optimization passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this pass."""
        ...

    @abstractmethod
    def run(self, graph: IRGraph) -> IRGraph:
        """Apply the pass to the graph and return the optimized graph.

        Args:
            graph: Input IR graph.

        Returns:
            Optimized IR graph (may be the same object, mutated).
        """
        ...

    @property
    def stats(self) -> PassStats:
        """Statistics from the last run."""
        return self._stats

    def __init__(self):
        self._stats = PassStats()


class FusionPass(OptimizationPass):
    """Fuses Conv + BatchNormalization + Relu sequences into FusedConvBnRelu nodes.

    Pattern matching:
        Conv -> BatchNormalization -> Relu  =>  FusedConvBnRelu

    This reduces memory bandwidth by avoiding intermediate tensor writes
    and enables specialized NPU kernel dispatch.
    """

    @property
    def name(self) -> str:
        return "FusionPass"

    def run(self, graph: IRGraph) -> IRGraph:
        self._stats = PassStats()
        new_nodes: list[IRNode] = []
        i = 0
        nodes = graph.nodes

        while i < len(nodes):
            # Look for Conv -> BN -> Relu pattern
            if (
                i + 2 < len(nodes)
                and nodes[i].op_type == "Conv"
                and nodes[i + 1].op_type == "BatchNormalization"
                and nodes[i + 2].op_type == "Relu"
            ):
                # Verify the chain is connected
                conv_out = nodes[i].outputs[0]
                bn_out = nodes[i + 1].outputs[0]
                bn_in = nodes[i + 1].inputs[0] if nodes[i + 1].inputs else None
                relu_in = nodes[i + 2].inputs[0] if nodes[i + 2].inputs else None

                if bn_in == conv_out and relu_in == bn_out:
                    # Fuse into single node
                    fused = IRNode(
                        name=f"fused_{nodes[i].name}",
                        op_type="FusedConvBnRelu",
                        inputs=nodes[i].inputs,
                        outputs=nodes[i + 2].outputs,
                        attributes={
                            **nodes[i].attributes,
                            "fused_from": [nodes[i].name, nodes[i + 1].name, nodes[i + 2].name],
                        },
                    )
                    new_nodes.append(fused)
                    self._stats.nodes_fused += 3
                    self._stats.nodes_removed += 2  # 3 nodes become 1
                    i += 3
                    continue

            new_nodes.append(nodes[i])
            i += 1

        graph.nodes = new_nodes
        return graph


class ConstantFoldingPass(OptimizationPass):
    """Removes nodes whose inputs are all constants (pre-computable).

    Nodes with only constant inputs can be evaluated at compile time.
    Their outputs are marked as constants for downstream passes.
    """

    @property
    def name(self) -> str:
        return "ConstantFoldingPass"

    def run(self, graph: IRGraph) -> IRGraph:
        self._stats = PassStats()

        # Track which tensors are constant (graph inputs are not constant)
        input_names = {inp.name for inp in graph.inputs}
        constant_tensors: set[str] = set()

        # Identify initializers / constants from node attributes
        for node in graph.nodes:
            if node.op_type in ("Constant", "Initializer"):
                for out in node.outputs:
                    constant_tensors.add(out)

        new_nodes: list[IRNode] = []
        for node in graph.nodes:
            # Skip constant-defining nodes
            if node.op_type in ("Constant", "Initializer"):
                self._stats.nodes_removed += 1
                continue

            # If all inputs are constant, fold this node
            all_const = (
                len(node.inputs) > 0
                and all(inp in constant_tensors for inp in node.inputs)
                and not any(inp in input_names for inp in node.inputs)
            )

            if all_const:
                # Mark outputs as constant for further folding
                for out in node.outputs:
                    constant_tensors.add(out)
                self._stats.nodes_removed += 1
            else:
                new_nodes.append(node)

        graph.nodes = new_nodes
        return graph


class DeadCodePass(OptimizationPass):
    """Eliminates nodes whose outputs are never consumed.

    A node is dead if none of its outputs appear as inputs to other nodes
    or as graph outputs.
    """

    @property
    def name(self) -> str:
        return "DeadCodePass"

    def run(self, graph: IRGraph) -> IRGraph:
        self._stats = PassStats()

        # Collect all consumed tensor names
        consumed: set[str] = set()
        for node in graph.nodes:
            for inp in node.inputs:
                consumed.add(inp)
        # Graph outputs are always consumed
        for out in graph.outputs:
            consumed.add(out.name)

        # Remove nodes whose outputs are never consumed
        new_nodes: list[IRNode] = []
        for node in graph.nodes:
            if any(out in consumed for out in node.outputs):
                new_nodes.append(node)
            else:
                self._stats.nodes_removed += 1

        graph.nodes = new_nodes
        return graph


class MemoryPlanningPass(OptimizationPass):
    """Annotates nodes with buffer reuse hints for memory optimization.

    Performs liveness analysis to determine which output buffers can be
    reused by later operations, reducing peak memory usage.
    """

    @property
    def name(self) -> str:
        return "MemoryPlanningPass"

    def run(self, graph: IRGraph) -> IRGraph:
        self._stats = PassStats()

        # Compute last-use index for each tensor
        last_use: dict[str, int] = {}
        for i, node in enumerate(graph.nodes):
            for inp in node.inputs:
                last_use[inp] = i

        # Mark graph outputs as live forever
        graph_output_names = {out.name for out in graph.outputs}

        # Assign buffer reuse hints
        # Tensors that die before a node's execution can donate their buffer
        free_buffers: list[str] = []

        for i, node in enumerate(graph.nodes):
            # Check which buffers became free before this node
            newly_free = []
            for tensor_name, last_idx in last_use.items():
                if last_idx < i and tensor_name not in graph_output_names:
                    newly_free.append(tensor_name)

            # Annotate with reuse hints
            if newly_free:
                node.attributes["_buffer_reuse_candidates"] = newly_free[:len(node.outputs)]
                self._stats.nodes_annotated += 1

        return graph
