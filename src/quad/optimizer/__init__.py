"""QUAD Optimizer — Graph optimization passes and compilation pipeline."""

from quad.optimizer.pipeline import optimize_model, OptimizationResult
from quad.optimizer.passes import (
    FusionPass,
    ConstantFoldingPass,
    DeadCodePass,
    MemoryPlanningPass,
)

__all__ = [
    "optimize_model",
    "OptimizationResult",
    "FusionPass",
    "ConstantFoldingPass",
    "DeadCodePass",
    "MemoryPlanningPass",
]
