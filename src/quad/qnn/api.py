"""QNN SDK API Components — reference definitions and initialization ordering.

Based on QNN SDK API Review (Nov 12, 2024, 80-63442-50 Oct 17 2024 base).

QNN (Qualcomm® AI Engine Direct) provides a unified C++ API for AI/ML inference
on QTI chipsets and AI acceleration cores (CPU, GPU, HTP/DSP).

API is C-style (for portability). Components are organized by category:
  Core     — QnnBackend, QnnDevice, QnnContext, QnnGraph, QnnTensor, QnnOpPackage
  Utility  — QnnProfile, QnnLog
  System   — QnnProperty, QnnMem, QnnSignal

Supported model formats: .so, .bin, .tflite

Key initialization rules:
  - QnnLog can be initialized BEFORE QnnBackend
  - QnnProperty can discover capabilities WITHOUT QnnBackend initialization
  - Most other APIs require QnnBackend to be initialized first
  - QnnContext binary cache (model.bin) is backend-specific — backend must match
    when loading vs when the cache was saved
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# API Component Definitions
# ══════════════════════════════════════════════════════════════════════════════

class QnnCategory(str, Enum):
    """API component category."""
    CORE = "Core"
    UTILITY = "Utility"
    SYSTEM = "System"


@dataclass
class QnnApiComponent:
    """Describes one QNN API component.

    backend_specialized = True means each hardware backend (CPU/GPU/HTP)
    may have a different implementation; False means universal across backends.
    """
    name: str
    category: QnnCategory
    backend_specialized: bool
    description: str
    key_features: list[str] = field(default_factory=list)
    init_before_backend: bool = False   # True = can init before QnnBackend


# Complete component table from documentation
QNN_API_COMPONENTS: dict[str, QnnApiComponent] = {
    "QnnBackend": QnnApiComponent(
        name="QnnBackend",
        category=QnnCategory.CORE,
        backend_specialized=True,
        description=(
            "Top-level QNN API component. Most QNN APIs require a backend to be "
            "initialized first. Provides QNN OpPackage Registry API."
        ),
        key_features=[
            "Must be initialized before most other APIs",
            "Loaded from backend .so (libQnnCpu.so, libQnnGpu.so, libQnnHtp.so)",
            "Provides OpPackage Registry",
        ],
    ),
    "QnnDevice": QnnApiComponent(
        name="QnnDevice",
        category=QnnCategory.CORE,
        backend_specialized=True,
        description=(
            "Top-level QNN API component for multi-core support. Provides all "
            "constructs required to associate desired hardware accelerator resources "
            "for execution of user composed graphs. A platform may have multiple "
            "devices; devices may have multiple cores. Provides API for performance control."
        ),
        key_features=[
            "Multi-core hardware resource association",
            "Performance control API",
            "Platform → Device → Core hierarchy",
        ],
    ),
    "QnnContext": QnnApiComponent(
        name="QnnContext",
        category=QnnCategory.CORE,
        backend_specialized=True,
        description=(
            "Provides execution environment for graphs and operations. Graphs and "
            "tensors shared between graphs are created within a context. Context "
            "content can be cached into binary form (model.bin) for faster loading. "
            "Provides Priority control configuration."
        ),
        key_features=[
            "Execution environment for graphs",
            "Binary cache (model.bin) for faster init",
            "Priority control configuration",
            "Binary cache is backend-specific — backend must match when loading",
        ],
    ),
    "QnnGraph": QnnApiComponent(
        name="QnnGraph",
        category=QnnCategory.CORE,
        backend_specialized=True,
        description=(
            "Provides composable graph API. A graph is created inside a context "
            "and composed from nodes and tensors. Nodes are connected with tensors. "
            "Must be finalized before execution."
        ),
        key_features=[
            "Created inside a QnnContext",
            "Composed of nodes (ops) connected by tensors",
            "Must be finalized (Finalize) before execution",
            "Available in .so format via QnnModel_composeGraphs()",
        ],
    ),
    "QnnTensor": QnnApiComponent(
        name="QnnTensor",
        category=QnnCategory.CORE,
        backend_specialized=False,
        description=(
            "Tensors hold either operation's static/constant data or input/output "
            "activation data. Tensors can have either Context or Graph scope. "
            "Context-scoped tensors can be shared between graphs in the same context."
        ),
        key_features=[
            "NOT backend-specialized (universal)",
            "Two scopes: Context scope (shared) or Graph scope (local)",
            "Holds static/constant data OR I/O activation data",
        ],
    ),
    "QnnOpPackage": QnnApiComponent(
        name="QnnOpPackage",
        category=QnnCategory.CORE,
        backend_specialized=True,
        description=(
            "Provides interface to the backend to use registered OpPackage libraries. "
            "Required for UDO (User-Defined Operations)."
        ),
        key_features=[
            "Registers OpPackage libraries with backend",
            "Required for UDO/custom ops",
        ],
    ),
    "QnnProfile": QnnApiComponent(
        name="QnnProfile",
        category=QnnCategory.UTILITY,
        backend_specialized=True,
        description=(
            "Provides means to profile QNN backends to evaluate performance "
            "(memory and timing) of graphs and operations."
        ),
        key_features=[
            "Memory and timing profiling",
            "Per-graph and per-op granularity",
        ],
    ),
    "QnnLog": QnnApiComponent(
        name="QnnLog",
        category=QnnCategory.UTILITY,
        backend_specialized=False,
        description=(
            "Provides means for QNN backends to output logging data. "
            "Can be extended to OpPackage as well. "
            "Can be initialized BEFORE QnnBackend."
        ),
        key_features=[
            "NOT backend-specialized",
            "Can be initialized before QnnBackend",
            "Extendable to OpPackages",
        ],
        init_before_backend=True,
    ),
    "QnnProperty": QnnApiComponent(
        name="QnnProperty",
        category=QnnCategory.SYSTEM,
        backend_specialized=False,
        description=(
            "Provides means for client to discover capabilities of a backend. "
            "Can be used WITHOUT QnnBackend initialization."
        ),
        key_features=[
            "NOT backend-specialized",
            "Can discover capabilities WITHOUT QnnBackend init",
            "Use to check hardware support before full init",
        ],
        init_before_backend=True,
    ),
    "QnnMem": QnnApiComponent(
        name="QnnMem",
        category=QnnCategory.SYSTEM,
        backend_specialized=False,
        description=(
            "Provides API to register externally allocated memory with a backend. "
            "Enables zero-copy buffer sharing."
        ),
        key_features=[
            "Register external memory buffers",
            "Zero-copy buffer sharing",
        ],
    ),
    "QnnSignal": QnnApiComponent(
        name="QnnSignal",
        category=QnnCategory.SYSTEM,
        backend_specialized=False,
        description=(
            "Provides means to manage Signal objects. Signal objects are used "
            "to control execution of other components."
        ),
        key_features=[
            "Execution control for async pipelines",
        ],
    ),
}


def get_components_by_category(category: QnnCategory) -> list[QnnApiComponent]:
    """Return all components in a given category."""
    return [c for c in QNN_API_COMPONENTS.values() if c.category == category]


def get_backend_specialized_components() -> list[QnnApiComponent]:
    """Return components that have backend-specific implementations."""
    return [c for c in QNN_API_COMPONENTS.values() if c.backend_specialized]


def get_pre_backend_components() -> list[QnnApiComponent]:
    """Return components that can be used before QnnBackend initialization."""
    return [c for c in QNN_API_COMPONENTS.values() if c.init_before_backend]
