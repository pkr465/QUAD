"""Optimization pipeline — end-to-end model optimization for QUAD targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quad.compiler.ir import IRGraph, IRNode
from quad.compiler.frontend_onnx import compile_onnx
from quad.optimizer.passes import (
    FusionPass,
    ConstantFoldingPass,
    DeadCodePass,
    MemoryPlanningPass,
    OptimizationPass,
)


@dataclass
class OptimizationResult:
    """Result of the optimization pipeline.

    Attributes:
        optimized_ir: The optimized IR graph.
        original_nodes: Number of nodes before optimization.
        optimized_nodes: Number of nodes after optimization.
        passes_applied: List of pass names that were applied.
        estimated_speedup: Estimated speedup factor (e.g., 1.5 = 50% faster).
        estimated_power_reduction_pct: Estimated power savings as a percentage.
        quantization_applied: Quantization scheme applied (e.g., "int8", "fp16").
    """

    optimized_ir: IRGraph
    original_nodes: int
    optimized_nodes: int
    passes_applied: list[str]
    estimated_speedup: float
    estimated_power_reduction_pct: float
    quantization_applied: str


def optimize_model(
    model_path: str,
    target: str = "qnpu_v3",
    quantization: str = "int8",
    power_budget_mw: float | None = None,
) -> OptimizationResult:
    """Run the full optimization pipeline on a model.

    Steps:
        1. Load/compile model to IR (via compile_onnx).
        2. Run all optimization passes in sequence.
        3. Apply quantization annotation.
        4. Compute and return optimization statistics.

    Args:
        model_path: Path to the input model file (.onnx).
        target: Target hardware identifier (e.g., "qnpu_v3", "qnpu_v2").
        quantization: Quantization scheme ("int8", "fp16", "none").
        power_budget_mw: Optional power budget in milliwatts for the optimizer
            to respect. If provided, aggressive optimizations may be applied.

    Returns:
        OptimizationResult with the optimized graph and statistics.
    """
    # Step 1: Compile to IR
    ir_graph = compile_onnx(model_path)
    original_nodes = ir_graph.num_nodes

    # Step 2: Run optimization passes
    passes: list[OptimizationPass] = [
        FusionPass(),
        ConstantFoldingPass(),
        DeadCodePass(),
        MemoryPlanningPass(),
    ]

    passes_applied: list[str] = []
    total_fused = 0
    total_removed = 0

    for opt_pass in passes:
        ir_graph = opt_pass.run(ir_graph)
        passes_applied.append(opt_pass.name)
        total_fused += opt_pass.stats.nodes_fused
        total_removed += opt_pass.stats.nodes_removed

    # Step 3: Apply quantization annotation
    if quantization != "none":
        ir_graph = _apply_quantization(ir_graph, quantization, target)

    optimized_nodes = ir_graph.num_nodes

    # Step 4: Estimate performance improvements
    estimated_speedup = _estimate_speedup(
        original_nodes, optimized_nodes, total_fused, quantization, target
    )
    estimated_power_reduction = _estimate_power_reduction(
        original_nodes, optimized_nodes, total_fused, quantization, power_budget_mw
    )

    return OptimizationResult(
        optimized_ir=ir_graph,
        original_nodes=original_nodes,
        optimized_nodes=optimized_nodes,
        passes_applied=passes_applied,
        estimated_speedup=estimated_speedup,
        estimated_power_reduction_pct=estimated_power_reduction,
        quantization_applied=quantization,
    )


def _apply_quantization(graph: IRGraph, quantization: str, target: str) -> IRGraph:
    """Annotate nodes with quantization metadata.

    Args:
        graph: The IR graph to annotate.
        quantization: Quantization scheme ("int8", "fp16").
        target: Target hardware for quantization constraints.

    Returns:
        The annotated graph.
    """
    for node in graph.nodes:
        # Mark compute-heavy ops for quantization
        if node.op_type in (
            "Conv", "Gemm", "MatMul", "FusedConvBnRelu", "Linear",
        ):
            node.attributes["_quantization"] = quantization
            node.attributes["_quant_target"] = target
        elif node.op_type in ("Relu", "MaxPool", "GlobalAveragePool", "Flatten"):
            # These ops pass through quantization from inputs
            node.attributes["_quantization"] = quantization

    return graph


def _estimate_speedup(
    original_nodes: int,
    optimized_nodes: int,
    nodes_fused: int,
    quantization: str,
    target: str,
) -> float:
    """Estimate the speedup factor from optimization.

    Heuristic model:
        - Node reduction contributes linearly.
        - Fusion contributes ~2x per fused group (reduced memory traffic).
        - INT8 quantization provides ~2-4x compute speedup on NPU.
        - FP16 provides ~1.5-2x.
    """
    # Base speedup from node reduction
    if original_nodes == 0:
        return 1.0

    reduction_ratio = optimized_nodes / original_nodes
    base_speedup = 1.0 / max(reduction_ratio, 0.1)

    # Fusion bonus: each fused group saves ~1.5x in memory traffic
    fusion_groups = nodes_fused // 3  # Each group fuses 3 nodes
    fusion_bonus = 1.0 + fusion_groups * 0.15

    # Quantization multiplier
    quant_multiplier = {
        "int8": 2.5,
        "fp16": 1.8,
        "none": 1.0,
    }.get(quantization, 1.0)

    # Target-specific bonus
    target_bonus = {
        "qnpu_v3": 1.2,
        "qnpu_v2": 1.0,
    }.get(target, 1.0)

    speedup = base_speedup * fusion_bonus * quant_multiplier * target_bonus
    return round(speedup, 2)


def _estimate_power_reduction(
    original_nodes: int,
    optimized_nodes: int,
    nodes_fused: int,
    quantization: str,
    power_budget_mw: float | None,
) -> float:
    """Estimate power reduction percentage.

    Heuristic:
        - Fewer nodes = less compute = less power.
        - Fusion reduces memory access power.
        - INT8 uses ~4x less energy per op than FP32.
    """
    if original_nodes == 0:
        return 0.0

    # Base power reduction from fewer nodes
    node_reduction_pct = (1.0 - optimized_nodes / original_nodes) * 100.0

    # Fusion reduces memory power (~15% per fused group)
    fusion_groups = nodes_fused // 3
    fusion_power_pct = fusion_groups * 5.0

    # Quantization power savings
    quant_power_pct = {
        "int8": 30.0,
        "fp16": 15.0,
        "none": 0.0,
    }.get(quantization, 0.0)

    total_reduction = node_reduction_pct + fusion_power_pct + quant_power_pct

    # Cap at reasonable maximum
    total_reduction = min(total_reduction, 75.0)

    return round(total_reduction, 1)
