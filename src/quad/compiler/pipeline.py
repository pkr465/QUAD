"""Compilation pipeline — end-to-end model compilation.

Closes part of GAP_ANALYSIS T1.1: previously this module returned
``b"QUAD_COMPILED_BINARY"`` placeholder bytes for every target,
silently lying about what the user got. Now:

* The frontend (ONNX → IR) is real (uses ``onnx.load`` when the
  package is installed; falls back to a representative mock graph
  when it isn't).
* The backend is **honestly stubbed** — by default it raises
  ``NotImplementedError`` rather than emitting placeholder bytes.
* A new ``coverage`` field on the QBin metadata reports % of ops
  covered per target, with the unsupported list, so the user can
  see what would or wouldn't compile.
* Set ``allow_placeholder_backend=True`` (or
  ``QUAD_PLACEHOLDER_BACKEND=1`` env var) to opt back into the
  legacy behaviour, e.g. for tests that don't need real binaries.

Real backend implementations (QNN context binary generation, SNPE
DLC compilation) require deep SDK integration and are outside the
scope of this phase. The honest stub + coverage report mean the user
can plan around the gap rather than discovering it at runtime.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

from quad.compiler.capabilities import ComputeCapability, get_capability, list_capabilities
from quad.compiler.frontend_onnx import compile_onnx
from quad.compiler.ir import IRGraph, QuadIR
from quad.compiler.op_coverage import compute_coverage_for_targets
from quad.compiler.qbin import QBin

logger = logging.getLogger(__name__)


class BackendNotImplementedError(NotImplementedError):
    """Raised when a real backend compilation is requested but not yet wired.

    Phase E of the gap-closure plan does the *frontend* honestly (real
    ONNX -> IR), but the *backend* (IR -> SDK-compiled binary for a
    specific target) is still pending real SDK integration. Set
    ``allow_placeholder_backend=True`` if you want the legacy
    placeholder bytes (e.g. for testing).
    """


def _placeholder_backend_allowed() -> bool:
    """Whether the legacy placeholder backend is opted into."""
    return os.environ.get("QUAD_PLACEHOLDER_BACKEND", "").strip().lower() in {"1", "true", "yes"}


def compile_model(
    model_path: str,
    output_path: str | None = None,
    targets: list[str] | Literal["all"] = "all",
    portable: bool = False,
    *,
    allow_placeholder_backend: bool | None = None,
    coverage_only: bool = False,
    backend: Literal["auto", "qairt", "stub"] = "auto",
    quantization: Literal["fp32", "int8", "int4"] = "fp32",
    use_cache: bool = True,
) -> QBin:
    """Compile a model to QUAD binary format.

    Pipeline:
      1. Frontend: Parse source format → QUAD IR (real)
      2. Coverage: Compute op-support % for each target (real)
      3. (Optional) Optimize IR — currently a no-op
      4. Backend: Generate target-specific binaries
         * If ``allow_placeholder_backend`` (or
           ``QUAD_PLACEHOLDER_BACKEND=1``): emit literal placeholder
           bytes, same as the historical behaviour. Useful for tests.
         * Otherwise: raise ``BackendNotImplementedError`` for the
           first target (callers can catch and use ``coverage_only=True``
           to get just the IR + coverage report).
      5. Package into .qbin fat binary

    Args:
        model_path: Path to source model (.onnx, .pt, .pth)
        output_path: Where to save .qbin (auto-generated if None)
        targets: List of target capabilities, or "all" for all known targets
        portable: If True, include only QIR (JIT at load time)
        allow_placeholder_backend: Opt into the legacy placeholder
            backend that emits ``b"QUAD_COMPILED_BINARY"``. Defaults to
            the value of ``QUAD_PLACEHOLDER_BACKEND``.
        coverage_only: If True, skip the backend step entirely and
            return a QBin with only the IR + coverage metadata.

    Returns:
        QBin containing compiled artifacts (or just IR + coverage if
        ``coverage_only`` / ``portable`` are set).

    Raises:
        BackendNotImplementedError: in real-backend mode, when the SDK
            integration for at least one target is not yet wired.
        ValueError: when the source format is unsupported.
    """
    path = Path(model_path)

    # Step 1: Frontend — parse to IR (real where possible)
    if path.suffix in (".onnx",):
        ir_graph = compile_onnx(model_path)
    elif path.suffix in (".pt", ".pth"):
        # PyTorch frontend stub — uses mock IR for now (real path
        # requires torch.onnx.export, which torch isn't a core dep)
        logger.warning(
            "pytorch_frontend_uses_mock_ir",
            extra={"path": str(path), "reason": "torch.onnx.export not invoked"},
        )
        ir_graph = compile_onnx(model_path)
    else:
        raise ValueError(f"Unsupported source format: {path.suffix}")

    # Step 2: Resolve target list
    if targets == "all":
        target_caps = list_capabilities()
    else:
        target_caps = [get_capability(t) for t in targets]

    # Step 3: Compute op coverage for each target
    coverage_reports = compute_coverage_for_targets(
        ir_graph,
        targets=[c.name for c in target_caps],
    )

    # Step 4: Build QBin with IR + coverage metadata
    qbin = QBin(name=ir_graph.name, ir=ir_graph)
    qbin.metadata = {
        "source_format": path.suffix.lstrip("."),
        "source_path": str(path),
        "num_nodes": ir_graph.num_nodes,
        "input_shapes": [t.shape for t in ir_graph.inputs],
        "output_shapes": [t.shape for t in ir_graph.outputs],
        "coverage": {target: report.to_dict() for target, report in coverage_reports.items()},
    }

    # Step 5: Backend — three paths:
    #   - portable / coverage_only: skip the backend entirely
    #   - backend="qairt" (or "auto" with SDK reachable): real qairt-converter
    #   - backend="stub" (or "auto" without SDK): placeholder/honest-error
    if portable or coverage_only:
        pass
    else:
        from quad.compiler.qairt_backend import (
            compile_with_qairt,
            is_qairt_available,
        )

        if allow_placeholder_backend is None:
            allow_placeholder_backend = _placeholder_backend_allowed()

        # Choose a backend: explicit overrides > auto > stub
        chosen_backend = backend
        if chosen_backend == "auto":
            chosen_backend = "qairt" if is_qairt_available() else "stub"

        if chosen_backend == "qairt":
            # Real backend — shell out via the QAIRTAdapter for each
            # target. We currently support ONNX as the source format;
            # the converter accepts other formats too but they need the
            # right ConversionRequest fields wired through.
            if path.suffix.lower() != ".onnx":
                raise BackendNotImplementedError(
                    f"QAIRT backend currently supports .onnx sources only "
                    f"(got {path.suffix!r}). Use coverage_only=True or "
                    "convert via QAIRTAdapter.convert_model directly."
                )
            for cap in target_caps:
                # NPU/HTP -> qnn target SDK; everything else -> snpe DLC.
                target_sdk = "qnn" if "npu" in cap.name.lower() or "htp" in cap.name.lower() else "snpe"
                result = compile_with_qairt(
                    str(path),
                    target_sdk=target_sdk,
                    quantization=quantization,
                    use_cache=use_cache,
                )
                qbin.add_target(
                    target=cap.name,
                    format=target_sdk,
                    data=result.binary,
                )
                qbin.metadata.setdefault("qairt_backend", {})[cap.name] = {
                    "target_sdk": result.target_sdk,
                    "quantization": result.quantization,
                    "binary_format": result.binary_format,
                    "binary_size_bytes": len(result.binary),
                    "cache_hit": result.cache_hit,
                    "supported_ops_pct": result.supported_ops_pct,
                    "duration_s": round(result.duration_s, 3),
                }
        elif allow_placeholder_backend:
            for cap in target_caps:
                qbin.add_target(
                    target=cap.name,
                    format="qnn" if "npu" in cap.name.lower() else "snpe",
                    data=b"QUAD_COMPILED_BINARY",  # Legacy placeholder
                )
        else:
            raise BackendNotImplementedError(
                f"Backend SDK call to compile IR -> binary is not yet wired for "
                f"target(s) {[c.name for c in target_caps]}. Options:\n"
                f"  1. Install QAIRT (quad sdk install <archive>) so the auto "
                "backend picks the real qairt-converter path.\n"
                f"  2. Use QAIRTAdapter.convert_model() to invoke the SDK "
                "tools directly.\n"
                "  3. Pass coverage_only=True to skip the backend and get just "
                "the IR + op-coverage report.\n"
                "  4. Pass portable=True to package only the QIR (JIT at load).\n"
                "  5. Set QUAD_PLACEHOLDER_BACKEND=1 to opt into the legacy "
                "placeholder bytes (testing only)."
            )

    # Step 6: Save if output path specified
    if output_path:
        qbin.save(output_path)

    return qbin
