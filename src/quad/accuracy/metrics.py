"""Inference Accuracy Metrics for SNPE/QAIRT models.

Based on SNPE "Inference Accuracy" documentation (80-63442-10 Rev AH, Apr 13 2026).

Qualcomm Neural Processing SDK inference classification precision is measured
against several popular public models. Accuracy scores do not vary per chipset.

Supported metrics:
  mAP  — mean Average Precision (object detection / retrieval models)
  Top-1 — chance the highest-probability predicted class is not the real class
  Top-5 — chance the real class is not in the top-5 predicted classes

mAP Formula:
  AveP = Σ(k=1..n) [P(k) × rel(k)] / (number of relevant documents)

  Where:
    k      = rank in the sequence of retrieved documents
    n      = number of retrieved documents
    P(k)   = precision at cut-off k = tp / (tp + fp)
    rel(k) = 1 if item at rank k is relevant, 0 otherwise

  mAP = mean of AveP across all categories.
  If no relevant documents are retrieved, precision score is zero.

Note: Accuracy scores do not vary per chipset — the same DLC will produce the
same accuracy on any Qualcomm device (Snapdragon X Elite, 8 Elite, QCS2210, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Core Metric Calculations
# ══════════════════════════════════════════════════════════════════════════════

def compute_average_precision(
    retrieved_docs: list[Any],
    relevant_docs: set[Any],
) -> float:
    """Compute Average Precision (AP) for a single category.

    This is the exact algorithm from the SNPE documentation:

        for j in range(len(img_sorted)):
            if img_sorted[j] in anno_imgs:
                count += 1.0
                AP += count/rank
            rank += 1.0
        if count == 0:
            AP = 0
        else:
            AP = AP/count

    Args:
        retrieved_docs: Ordered list of retrieved items (ranked by confidence,
                        highest confidence first). Corresponds to img_sorted.
        relevant_docs:  Set of ground-truth relevant items. Corresponds to anno_imgs.

    Returns:
        Average Precision in [0, 1]. Returns 0.0 if no relevant documents
        are retrieved (as per the documentation specification).

    Example::
        retrieved = ["cat.jpg", "dog.jpg", "cat2.jpg", "bird.jpg"]
        relevant  = {"cat.jpg", "cat2.jpg", "cat3.jpg"}
        ap = compute_average_precision(retrieved, relevant)
        # P@1=1/1, P@3=2/3 → AP = (1.0 + 2/3) / 3 ≈ 0.556
    """
    ap = 0.0
    count = 0.0
    rank = 1.0

    for item in retrieved_docs:
        if item in relevant_docs:
            count += 1.0
            ap += count / rank
        rank += 1.0

    if count == 0:
        return 0.0
    return ap / count


def compute_map(
    per_category_results: dict[str, tuple[list[Any], set[Any]]],
) -> float:
    """Compute mean Average Precision (mAP) across all categories.

    mAP = mean of AP scores across all categories.

    Args:
        per_category_results: Mapping from category name to
                              (retrieved_docs, relevant_docs) tuples.
                              retrieved_docs: ranked list (highest conf first).
                              relevant_docs: ground-truth set.

    Returns:
        mAP in [0, 1]. Returns 0.0 for empty input.

    Example::
        results = {
            "cat": (["a.jpg", "b.jpg", "c.jpg"], {"a.jpg", "c.jpg"}),
            "dog": (["d.jpg", "e.jpg"], {"d.jpg"}),
        }
        map_score = compute_map(results)
    """
    if not per_category_results:
        return 0.0
    ap_scores = [
        compute_average_precision(retrieved, relevant)
        for retrieved, relevant in per_category_results.values()
    ]
    return sum(ap_scores) / len(ap_scores)


def compute_top_k_error(
    predictions: list[list[int]],
    ground_truth: list[int],
    k: int = 1,
) -> float:
    """Compute Top-K error rate for classification models.

    Top-K error rate = fraction of samples where the true class is NOT
    in the top-K predicted classes.

    Args:
        predictions: List of per-sample class rankings. Each inner list
                     contains class indices sorted by descending confidence
                     (index 0 = highest confidence class).
        ground_truth: True class index for each sample.
        k: Number of top classes to consider (1 for Top-1, 5 for Top-5).

    Returns:
        Error rate in [0, 1]. Lower is better.

    Example::
        preds = [[3, 1, 5, 2, 0], [2, 4, 1, 0, 3]]  # Top-5 per sample
        truth = [3, 0]  # true labels
        err = compute_top_k_error(preds, truth, k=1)
        # Sample 0: top-1 = 3 == truth → correct. Sample 1: top-1 = 2 ≠ truth → wrong
        # error = 1/2 = 0.5
    """
    if not predictions or not ground_truth:
        return 0.0
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions length {len(predictions)} != "
            f"ground_truth length {len(ground_truth)}"
        )

    errors = sum(
        1 for pred_ranking, true_class in zip(predictions, ground_truth)
        if true_class not in pred_ranking[:k]
    )
    return errors / len(ground_truth)


def compute_top1_error(
    predictions: list[list[int]],
    ground_truth: list[int],
) -> float:
    """Compute Top-1 error rate (most common classification metric).

    Top-1 error = chance the highest-probability predicted class is NOT
    the real class.

    Args:
        predictions: Per-sample ranked class lists (index 0 = top-1 prediction).
        ground_truth: True class index per sample.

    Returns:
        Top-1 error rate in [0, 1].
    """
    return compute_top_k_error(predictions, ground_truth, k=1)


def compute_top5_error(
    predictions: list[list[int]],
    ground_truth: list[int],
) -> float:
    """Compute Top-5 error rate.

    Top-5 error = chance the real class is NOT contained in the 5 classes
    with highest probability.

    Args:
        predictions: Per-sample ranked class lists (index 0 = highest conf).
        ground_truth: True class index per sample.

    Returns:
        Top-5 error rate in [0, 1].
    """
    return compute_top_k_error(predictions, ground_truth, k=5)


# ══════════════════════════════════════════════════════════════════════════════
# Result Containers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AccuracyReport:
    """Accuracy evaluation results for a SNPE inference run.

    Accuracy scores do not vary per chipset — the same DLC produces the
    same accuracy on any Qualcomm device.
    """
    model_name: str
    dataset: str = ""
    num_samples: int = 0

    # Classification metrics
    top1_error: Optional[float] = None      # Top-1 error rate [0, 1]
    top5_error: Optional[float] = None      # Top-5 error rate [0, 1]

    # Detection metrics
    map_score: Optional[float] = None       # mAP [0, 1]
    per_category_ap: dict[str, float] = field(default_factory=dict)

    # Notes
    notes: list[str] = field(default_factory=list)

    @property
    def top1_accuracy(self) -> Optional[float]:
        """Top-1 accuracy = 1 - Top-1 error rate."""
        if self.top1_error is None:
            return None
        return 1.0 - self.top1_error

    @property
    def top5_accuracy(self) -> Optional[float]:
        """Top-5 accuracy = 1 - Top-5 error rate."""
        if self.top5_error is None:
            return None
        return 1.0 - self.top5_error

    def summary(self) -> str:
        """Return a one-line human-readable accuracy summary."""
        parts = [f"Model: {self.model_name}"]
        if self.num_samples:
            parts.append(f"n={self.num_samples}")
        if self.top1_error is not None:
            parts.append(f"Top-1 err={self.top1_error:.2%}")
        if self.top5_error is not None:
            parts.append(f"Top-5 err={self.top5_error:.2%}")
        if self.map_score is not None:
            parts.append(f"mAP={self.map_score:.4f}")
        return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class AccuracyEvaluator:
    """Evaluate SNPE model inference accuracy against ground truth.

    Supports classification (Top-1/Top-5) and detection (mAP) evaluation.

    Note from SNPE docs: accuracy scores do not vary per chipset. The same
    DLC will produce the same accuracy regardless of target device.
    """

    def evaluate_classification(
        self,
        model_name: str,
        predictions: list[list[int]],
        ground_truth: list[int],
        dataset: str = "",
        top_k: tuple[int, ...] = (1, 5),
    ) -> AccuracyReport:
        """Evaluate classification accuracy (Top-K error rates).

        Args:
            model_name: Model identifier for the report
            predictions: Per-sample ranked class lists (index 0 = top prediction)
            ground_truth: True class index per sample
            dataset: Dataset name (e.g. "ImageNet", "COCO")
            top_k: Which top-K values to evaluate

        Returns:
            AccuracyReport with top1_error and/or top5_error populated
        """
        report = AccuracyReport(
            model_name=model_name,
            dataset=dataset,
            num_samples=len(ground_truth),
        )
        if 1 in top_k:
            report.top1_error = compute_top1_error(predictions, ground_truth)
        if 5 in top_k:
            report.top5_error = compute_top5_error(predictions, ground_truth)

        report.notes.append(
            "Accuracy scores do not vary per Qualcomm chipset."
        )
        return report

    def evaluate_detection(
        self,
        model_name: str,
        per_category_results: dict[str, tuple[list[Any], set[Any]]],
        dataset: str = "",
    ) -> AccuracyReport:
        """Evaluate detection accuracy (mAP).

        Args:
            model_name: Model identifier
            per_category_results: {category: (retrieved_ranked_list, relevant_set)}
            dataset: Dataset name (e.g. "COCO", "VOC2012")

        Returns:
            AccuracyReport with map_score and per_category_ap populated
        """
        per_ap = {
            cat: compute_average_precision(retrieved, relevant)
            for cat, (retrieved, relevant) in per_category_results.items()
        }
        map_score = sum(per_ap.values()) / len(per_ap) if per_ap else 0.0

        total_samples = sum(
            len(retrieved) for retrieved, _ in per_category_results.values()
        )
        report = AccuracyReport(
            model_name=model_name,
            dataset=dataset,
            num_samples=total_samples,
            map_score=map_score,
            per_category_ap=per_ap,
        )
        report.notes.append(
            "Accuracy scores do not vary per Qualcomm chipset."
        )
        return report


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

ACCURACY_NOTES: dict[str, Any] = {
    "description": (
        "SNPE inference classification precision metrics. "
        "Scores do not vary per chipset."
    ),
    "metrics": {
        "mAP": {
            "full_name": "mean Average Precision",
            "use_case": "Object detection and retrieval models",
            "description": (
                "Mean of AP scores across all categories. "
                "Each AP measures how well retrieved results are ranked "
                "relative to ground-truth relevance."
            ),
            "range": "[0, 1] — higher is better",
        },
        "Top-1 error": {
            "description": (
                "Chance the highest-probability predicted class is not the real class."
            ),
            "range": "[0, 1] — lower is better",
            "complement": "Top-1 accuracy = 1 - Top-1 error",
        },
        "Top-5 error": {
            "description": (
                "Chance the real class is not contained in the 5 classes "
                "with highest probability."
            ),
            "range": "[0, 1] — lower is better",
            "complement": "Top-5 accuracy = 1 - Top-5 error",
        },
    },
    "ap_formula": {
        "description": "AveP = Σ(k=1..n) [P(k) × rel(k)] / count_relevant",
        "variables": {
            "k": "rank in the sequence of retrieved documents",
            "n": "number of retrieved documents",
            "P(k)": "precision at cut-off k = tp / (tp + fp)",
            "rel(k)": "1 if item at rank k is relevant, 0 otherwise",
        },
        "edge_case": "If no relevant documents are retrieved, AP = 0.",
        "python_reference": (
            "# From SNPE documentation:\n"
            "for j in range(len(img_sorted)):\n"
            "    if img_sorted[j] in anno_imgs:\n"
            "        count += 1.0\n"
            "        AP += count/rank\n"
            "    rank += 1.0\n"
            "if (count == 0):\n"
            "    AP = 0\n"
            "else:\n"
            "    AP = AP/count"
        ),
    },
    "chipset_note": (
        "Accuracy scores do not vary per chipset. "
        "The same DLC produces the same accuracy on any Qualcomm device "
        "(Snapdragon X Elite, 8 Elite, QCS2210, etc.)."
    ),
}
