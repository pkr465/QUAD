"""QUAD Serve — Model Zoo / Registry for discovering and managing models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a model in the registry.

    Attributes:
        name: Model name identifier.
        path: Path to model binary (.qbin).
        chipsets: List of compatible chipsets/devices.
        input_shapes: Expected input tensor shapes.
        output_shapes: Expected output tensor shapes.
    """

    name: str
    path: str
    chipsets: list[str] = field(default_factory=lambda: ["snapdragon-8-gen3"])
    input_shapes: dict[str, list[int]] = field(default_factory=dict)
    output_shapes: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Published performance metrics for a model.

    Attributes:
        latency_ms: Typical inference latency.
        accuracy: Model accuracy (top-1 for classification).
        throughput_fps: Frames per second.
        power_mw: Typical power consumption.
    """

    latency_ms: float = 0.0
    accuracy: float = 0.0
    throughput_fps: float = 0.0
    power_mw: float = 0.0


@dataclass
class ModelEntry:
    """A model registered in the model zoo.

    Attributes:
        name: Model name.
        path: Path to model binary.
        version: Model version number.
        chipsets: List of compatible chipsets.
        metrics: Published performance metrics.
        tags: Searchable tags.
        description: Human-readable description.
    """

    name: str
    path: str
    version: int = 1
    chipsets: list[str] = field(default_factory=lambda: ["snapdragon-8-gen3"])
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    tags: list[str] = field(default_factory=list)
    description: str = ""


# Pre-populated mock model entries
_DEFAULT_MODELS: list[ModelEntry] = [
    ModelEntry(
        name="mobilenetv2",
        path="models/mobilenetv2.qbin",
        version=1,
        chipsets=["snapdragon-8-gen3", "snapdragon-8-gen2", "snapdragon-7-gen1"],
        metrics=ModelMetrics(latency_ms=2.1, accuracy=0.718, throughput_fps=476, power_mw=800),
        tags=["classification", "imagenet", "mobile", "efficient"],
        description="MobileNetV2 optimized for Qualcomm NPU — lightweight image classification.",
    ),
    ModelEntry(
        name="resnet50",
        path="models/resnet50.qbin",
        version=2,
        chipsets=["snapdragon-8-gen3", "snapdragon-8-gen2"],
        metrics=ModelMetrics(latency_ms=4.8, accuracy=0.761, throughput_fps=208, power_mw=1500),
        tags=["classification", "imagenet", "backbone"],
        description="ResNet-50 quantized for on-device inference.",
    ),
    ModelEntry(
        name="yolov8n",
        path="models/yolov8n.qbin",
        version=3,
        chipsets=["snapdragon-8-gen3", "snapdragon-8-gen2"],
        metrics=ModelMetrics(latency_ms=6.2, accuracy=0.372, throughput_fps=161, power_mw=1800),
        tags=["detection", "yolo", "realtime", "coco"],
        description="YOLOv8 Nano — real-time object detection for mobile.",
    ),
    ModelEntry(
        name="whisper-tiny",
        path="models/whisper_tiny.qbin",
        version=1,
        chipsets=["snapdragon-8-gen3"],
        metrics=ModelMetrics(latency_ms=45.0, accuracy=0.89, throughput_fps=22, power_mw=2500),
        tags=["speech", "asr", "whisper", "audio"],
        description="Whisper Tiny — on-device speech recognition.",
    ),
    ModelEntry(
        name="llama-7b",
        path="models/llama_7b_4bit.qbin",
        version=1,
        chipsets=["snapdragon-8-gen3"],
        metrics=ModelMetrics(latency_ms=120.0, accuracy=0.0, throughput_fps=8, power_mw=5000),
        tags=["llm", "language", "generative", "4bit"],
        description="LLaMA 7B 4-bit quantized — on-device large language model.",
    ),
]


class ModelRegistry:
    """Model zoo and registry for discovering available models.

    Provides model discovery, version management, and metadata queries.

    Usage:
        registry = ModelRegistry()
        entry = registry.get("yolov8n")
        results = registry.search("detection")
    """

    def __init__(self, preload_defaults: bool = True):
        self._models: dict[str, list[ModelEntry]] = {}
        if preload_defaults:
            for entry in _DEFAULT_MODELS:
                self._add_entry(entry)

    def _add_entry(self, entry: ModelEntry) -> None:
        """Internal: add entry to versioned storage."""
        if entry.name not in self._models:
            self._models[entry.name] = []
        self._models[entry.name].append(entry)

    def register(self, name: str, path: str, metadata: dict[str, Any] | None = None) -> ModelEntry:
        """Register a new model in the registry.

        Args:
            name: Model name identifier.
            path: Path to the model binary.
            metadata: Optional metadata (chipsets, tags, description, metrics).

        Returns:
            The created ModelEntry.
        """
        metadata = metadata or {}
        version = len(self._models.get(name, [])) + 1

        metrics_data = metadata.get("metrics", {})
        metrics = ModelMetrics(**metrics_data) if isinstance(metrics_data, dict) else ModelMetrics()

        entry = ModelEntry(
            name=name,
            path=path,
            version=version,
            chipsets=metadata.get("chipsets", ["snapdragon-8-gen3"]),
            metrics=metrics,
            tags=metadata.get("tags", []),
            description=metadata.get("description", ""),
        )
        self._add_entry(entry)
        return entry

    def get(self, name: str, version: int | None = None) -> ModelEntry:
        """Get a model entry by name and optional version.

        Args:
            name: Model name.
            version: Specific version (latest if None).

        Returns:
            The matching ModelEntry.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        versions = self._models[name]
        if version is not None:
            for entry in versions:
                if entry.version == version:
                    return entry
            raise KeyError(f"Model '{name}' version {version} not found")
        return versions[-1]  # Latest version

    def list_all(self) -> list[ModelEntry]:
        """List all models in the registry (latest versions).

        Returns:
            List of ModelEntry objects.
        """
        return [versions[-1] for versions in self._models.values()]

    def search(self, query: str) -> list[ModelEntry]:
        """Search models by name, tag, or description.

        Args:
            query: Search string to match against names, tags, and descriptions.

        Returns:
            List of matching ModelEntry objects.
        """
        query_lower = query.lower()
        results = []
        for versions in self._models.values():
            entry = versions[-1]
            if (
                query_lower in entry.name.lower()
                or query_lower in entry.description.lower()
                or any(query_lower in tag for tag in entry.tags)
            ):
                results.append(entry)
        return results

    def get_versions(self, name: str) -> list[int]:
        """Get all available versions for a model.

        Args:
            name: Model name.

        Returns:
            List of version numbers.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        return [entry.version for entry in self._models[name]]

    def remove(self, name: str) -> None:
        """Remove a model from the registry.

        Args:
            name: Model name to remove.

        Raises:
            KeyError: If model not found.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry")
        del self._models[name]

    @property
    def count(self) -> int:
        """Number of unique models in the registry."""
        return len(self._models)
