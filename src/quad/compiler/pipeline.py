"""Compilation pipeline — end-to-end model compilation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from quad.compiler.capabilities import ComputeCapability, get_capability, list_capabilities
from quad.compiler.frontend_onnx import compile_onnx
from quad.compiler.ir import IRGraph, QuadIR
from quad.compiler.qbin import QBin


def compile_model(
    model_path: str,
    output_path: str | None = None,
    targets: list[str] | Literal["all"] = "all",
    portable: bool = False,
) -> QBin:
    """Compile a model to QUAD binary format.

    Pipeline:
    1. Frontend: Parse source format → QUAD IR
    2. (Optional) Optimize IR
    3. Backend: Generate target-specific binaries
    4. Package into .qbin fat binary

    Args:
        model_path: Path to source model (.onnx, .pt)
        output_path: Where to save .qbin (auto-generated if None)
        targets: List of target capabilities, or "all" for all known targets
        portable: If True, include only QIR (JIT at load time)

    Returns:
        QBin containing compiled artifacts
    """
    path = Path(model_path)

    # Step 1: Frontend — parse to IR
    if path.suffix in (".onnx",):
        ir_graph = compile_onnx(model_path)
    elif path.suffix in (".pt", ".pth"):
        # PyTorch frontend stub — uses mock IR for now
        ir_graph = compile_onnx(model_path)  # Same mock path
    else:
        raise ValueError(f"Unsupported source format: {path.suffix}")

    # Step 2: Create QBin
    qbin = QBin(name=ir_graph.name, ir=ir_graph)
    qbin.metadata = {
        "source_format": path.suffix.lstrip("."),
        "source_path": str(path),
        "num_nodes": ir_graph.num_nodes,
        "input_shapes": [t.shape for t in ir_graph.inputs],
        "output_shapes": [t.shape for t in ir_graph.outputs],
    }

    # Step 3: Generate target binaries (if not portable-only)
    if not portable:
        if targets == "all":
            target_caps = list_capabilities()
        else:
            target_caps = [get_capability(t) for t in targets]

        for cap in target_caps:
            # Mock: generate placeholder binary for each target
            # Real: invoke QNN/SNPE compiler for each target
            qbin.add_target(
                target=cap.name,
                format="qnn" if "npu_v3" in cap.name else "snpe",
                data=b"QUAD_COMPILED_BINARY",  # Placeholder
            )

    # Step 4: Save if output path specified
    if output_path:
        qbin.save(output_path)

    return qbin
