"""Tests for SNPE inference accuracy metrics."""

from __future__ import annotations

import pytest

from quad.accuracy.metrics import (
    ACCURACY_NOTES,
    AccuracyEvaluator,
    AccuracyReport,
    compute_average_precision,
    compute_map,
    compute_top1_error,
    compute_top5_error,
    compute_top_k_error,
)


# ══════════════════════════════════════════════════════════════════════════════
# compute_average_precision — exact SNPE reference algorithm
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAveragePrecision:
    def test_documentation_algorithm_perfect_retrieval(self) -> None:
        """All retrieved docs are relevant — AP should be 1.0."""
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b", "c"}
        assert compute_average_precision(retrieved, relevant) == pytest.approx(1.0)

    def test_documentation_algorithm_no_relevant_retrieved(self) -> None:
        """No relevant docs retrieved → AP = 0 (explicit edge case in docs)."""
        retrieved = ["a", "b", "c"]
        relevant = {"x", "y", "z"}
        assert compute_average_precision(retrieved, relevant) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        assert compute_average_precision([], {"a", "b"}) == 0.0

    def test_single_relevant_at_rank_1(self) -> None:
        """One relevant doc at rank 1: P(1)=1/1, AP = 1.0/1 = 1.0."""
        assert compute_average_precision(["a"], {"a"}) == pytest.approx(1.0)

    def test_single_relevant_at_rank_2(self) -> None:
        """One relevant at rank 2: P(2)=1/2, AP = (1/2)/1 = 0.5."""
        result = compute_average_precision(["x", "a"], {"a"})
        assert result == pytest.approx(0.5)

    def test_two_relevant_at_ranks_1_and_3(self) -> None:
        """
        Retrieved: [a, x, b, y]  Relevant: {a, b}
        Rank 1: a ∈ relevant → count=1, AP += 1/1
        Rank 2: x ∉ relevant
        Rank 3: b ∈ relevant → count=2, AP += 2/3
        AP = (1 + 2/3) / 2 = 5/6
        """
        result = compute_average_precision(["a", "x", "b", "y"], {"a", "b"})
        expected = (1.0 + 2.0 / 3.0) / 2.0
        assert result == pytest.approx(expected)

    def test_reproduces_snpe_doc_code_exactly(self) -> None:
        """Run the exact SNPE documentation algorithm and compare."""
        img_sorted = ["img0", "img1", "img2", "img3", "img4"]
        anno_imgs = {"img0", "img2", "img4"}  # relevant at ranks 1, 3, 5

        # SNPE reference implementation
        AP_ref = 0.0
        count_ref = 0.0
        rank_ref = 1.0
        for j in range(len(img_sorted)):
            if img_sorted[j] in anno_imgs:
                count_ref += 1.0
                AP_ref += count_ref / rank_ref
            rank_ref += 1.0
        if count_ref == 0:
            AP_ref = 0.0
        else:
            AP_ref = AP_ref / count_ref

        result = compute_average_precision(img_sorted, anno_imgs)
        assert result == pytest.approx(AP_ref)

    def test_all_irrelevant_except_last(self) -> None:
        """Only last item is relevant: P(5)=1/5, AP = (1/5)/1 = 0.2."""
        result = compute_average_precision(["x", "y", "z", "w", "a"], {"a"})
        assert result == pytest.approx(0.2)

    def test_duplicate_in_retrieved_counts_once(self) -> None:
        """If same item appears twice in retrieved, second match still counts."""
        # This tests that list iteration (not set lookup) drives the loop
        result = compute_average_precision(["a", "a"], {"a"})
        # rank1: a∈rel → count=1, AP+=1/1; rank2: a∈rel → count=2, AP+=2/2
        # AP = (1 + 1) / 2 = 1.0
        assert result == pytest.approx(1.0)

    def test_empty_relevant_set(self) -> None:
        """No ground truth → no relevant docs → AP = 0."""
        result = compute_average_precision(["a", "b"], set())
        assert result == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# compute_map
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeMap:
    def test_empty_input_returns_zero(self) -> None:
        assert compute_map({}) == 0.0

    def test_single_category_perfect(self) -> None:
        result = compute_map({"cat": (["a", "b"], {"a", "b"})})
        assert result == pytest.approx(1.0)

    def test_two_categories_averaged(self) -> None:
        """AP for cat=1.0, AP for dog=0.5 → mAP = 0.75."""
        ap_cat = compute_average_precision(["a"], {"a"})           # 1.0
        ap_dog = compute_average_precision(["x", "b"], {"b"})     # 0.5
        expected = (ap_cat + ap_dog) / 2.0

        result = compute_map({
            "cat": (["a"], {"a"}),
            "dog": (["x", "b"], {"b"}),
        })
        assert result == pytest.approx(expected)

    def test_all_zero_aps(self) -> None:
        result = compute_map({
            "cat": (["x", "y"], {"a", "b"}),
            "dog": (["z"], {"w"}),
        })
        assert result == pytest.approx(0.0)

    def test_map_is_mean_of_per_category_aps(self) -> None:
        categories = {
            "bird": (["b1", "b2", "x"], {"b1", "b2"}),
            "fish": (["f1", "x", "f2"], {"f1", "f2"}),
            "bear": (["x", "y", "br"], {"br"}),
        }
        aps = [
            compute_average_precision(ret, rel)
            for ret, rel in categories.values()
        ]
        expected = sum(aps) / len(aps)
        result = compute_map(categories)
        assert result == pytest.approx(expected)


# ══════════════════════════════════════════════════════════════════════════════
# compute_top_k_error
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeTopKError:
    def test_top1_all_correct(self) -> None:
        preds = [[3], [1], [0]]
        truth = [3, 1, 0]
        assert compute_top_k_error(preds, truth, k=1) == pytest.approx(0.0)

    def test_top1_all_wrong(self) -> None:
        preds = [[0], [0], [0]]
        truth = [1, 2, 3]
        assert compute_top_k_error(preds, truth, k=1) == pytest.approx(1.0)

    def test_top1_half_correct(self) -> None:
        preds = [[3, 1, 2], [2, 0, 1]]
        truth = [3, 0]
        # Sample 0: top-1=3==truth → correct. Sample 1: top-1=2≠0 → wrong
        assert compute_top_k_error(preds, truth, k=1) == pytest.approx(0.5)

    def test_top5_correct_when_in_top5(self) -> None:
        preds = [[0, 1, 2, 3, 4]]
        truth = [4]  # true class is at position 5 (index 4)
        assert compute_top_k_error(preds, truth, k=5) == pytest.approx(0.0)

    def test_top5_error_when_outside_top5(self) -> None:
        preds = [[0, 1, 2, 3, 4]]
        truth = [5]  # true class not in top 5
        assert compute_top_k_error(preds, truth, k=5) == pytest.approx(1.0)

    def test_empty_inputs_return_zero(self) -> None:
        assert compute_top_k_error([], [], k=1) == 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="predictions length"):
            compute_top_k_error([[0]], [0, 1], k=1)

    def test_top1_is_special_case_of_top_k(self) -> None:
        preds = [[3, 1, 2], [2, 0, 1]]
        truth = [3, 0]
        assert compute_top_k_error(preds, truth, k=1) == compute_top1_error(preds, truth)

    def test_top5_is_special_case_of_top_k(self) -> None:
        preds = [[3, 1, 2, 0, 4], [2, 0, 1, 5, 3]]
        truth = [3, 0]
        assert compute_top_k_error(preds, truth, k=5) == compute_top5_error(preds, truth)


# ══════════════════════════════════════════════════════════════════════════════
# compute_top1_error / compute_top5_error
# ══════════════════════════════════════════════════════════════════════════════

class TestTop1AndTop5Error:
    def test_top1_error_perfect(self) -> None:
        preds = [[1, 2, 3, 4, 5], [2, 1, 3, 4, 5]]
        truth = [1, 2]
        assert compute_top1_error(preds, truth) == pytest.approx(0.0)

    def test_top1_error_matches_manual(self) -> None:
        # 3 samples: correct, wrong, correct
        preds = [[0], [1], [2]]
        truth = [0, 0, 2]
        assert compute_top1_error(preds, truth) == pytest.approx(1.0 / 3.0)

    def test_top5_error_better_than_top1_error(self) -> None:
        """Top-5 error should always be <= Top-1 error."""
        preds = [
            [9, 1, 2, 3, 0],  # top-1 wrong, top-5 has 0
            [0, 1, 2, 3, 4],  # both correct
        ]
        truth = [0, 0]
        top1 = compute_top1_error(preds, truth)
        top5 = compute_top5_error(preds, truth)
        assert top5 <= top1

    def test_top5_error_zero_when_all_in_top5(self) -> None:
        preds = [[5, 4, 3, 2, 1], [0, 9, 8, 7, 6]]
        truth = [1, 6]
        assert compute_top5_error(preds, truth) == pytest.approx(0.0)

    def test_top5_error_one_when_all_outside_top5(self) -> None:
        preds = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        truth = [5, 6]
        assert compute_top5_error(preds, truth) == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# AccuracyReport
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracyReport:
    def test_top1_accuracy_is_complement_of_error(self) -> None:
        report = AccuracyReport(model_name="resnet50", top1_error=0.24)
        assert report.top1_accuracy == pytest.approx(0.76)

    def test_top5_accuracy_is_complement_of_error(self) -> None:
        report = AccuracyReport(model_name="resnet50", top5_error=0.07)
        assert report.top5_accuracy == pytest.approx(0.93)

    def test_accuracy_none_when_error_none(self) -> None:
        report = AccuracyReport(model_name="model")
        assert report.top1_accuracy is None
        assert report.top5_accuracy is None

    def test_summary_includes_model_name(self) -> None:
        report = AccuracyReport(model_name="mobilenetv2", top1_error=0.28)
        assert "mobilenetv2" in report.summary()

    def test_summary_includes_top1_error(self) -> None:
        report = AccuracyReport(model_name="m", top1_error=0.28, num_samples=1000)
        summary = report.summary()
        assert "28.00%" in summary or "Top-1" in summary

    def test_summary_includes_map(self) -> None:
        report = AccuracyReport(model_name="m", map_score=0.712)
        assert "0.7120" in report.summary() or "mAP" in report.summary()

    def test_per_category_ap_stored(self) -> None:
        report = AccuracyReport(
            model_name="mobilenet_ssd",
            map_score=0.65,
            per_category_ap={"person": 0.78, "car": 0.52},
        )
        assert report.per_category_ap["person"] == pytest.approx(0.78)
        assert report.per_category_ap["car"] == pytest.approx(0.52)

    def test_chipset_note_in_default_notes(self) -> None:
        evaluator = AccuracyEvaluator()
        report = evaluator.evaluate_classification(
            "resnet50",
            predictions=[[0], [1]],
            ground_truth=[0, 1],
        )
        assert any("chipset" in n.lower() for n in report.notes)


# ══════════════════════════════════════════════════════════════════════════════
# AccuracyEvaluator
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracyEvaluator:
    def test_evaluate_classification_top1_and_top5(self) -> None:
        evaluator = AccuracyEvaluator()
        preds = [
            [0, 1, 2, 3, 4],
            [1, 0, 2, 3, 4],
            [2, 1, 0, 3, 4],
        ]
        truth = [0, 1, 2]
        report = evaluator.evaluate_classification("resnet50", preds, truth, dataset="imagenet")
        assert report.top1_error == pytest.approx(0.0)
        assert report.top5_error == pytest.approx(0.0)
        assert report.model_name == "resnet50"
        assert report.dataset == "imagenet"
        assert report.num_samples == 3

    def test_evaluate_classification_partial_error(self) -> None:
        evaluator = AccuracyEvaluator()
        preds = [[9, 1, 2, 3, 0], [0, 1, 2, 3, 4]]
        truth = [0, 0]
        report = evaluator.evaluate_classification("m", preds, truth)
        # Sample 0: top-1=9≠0 (wrong), top-5 includes 0 (correct)
        # Sample 1: top-1=0=0 (correct)
        assert report.top1_error == pytest.approx(0.5)
        assert report.top5_error == pytest.approx(0.0)

    def test_evaluate_classification_only_top1(self) -> None:
        evaluator = AccuracyEvaluator()
        report = evaluator.evaluate_classification(
            "m", [[0], [1]], [0, 1], top_k=(1,)
        )
        assert report.top1_error is not None
        assert report.top5_error is None

    def test_evaluate_detection_map(self) -> None:
        evaluator = AccuracyEvaluator()
        categories = {
            "cat": (["c1", "c2", "x"], {"c1", "c2"}),
            "dog": (["x", "d1"], {"d1"}),
        }
        report = evaluator.evaluate_detection("mobilenet_ssd", categories, dataset="coco")
        assert report.map_score is not None
        assert 0.0 <= report.map_score <= 1.0
        assert "cat" in report.per_category_ap
        assert "dog" in report.per_category_ap

    def test_evaluate_detection_empty(self) -> None:
        evaluator = AccuracyEvaluator()
        report = evaluator.evaluate_detection("m", {})
        assert report.map_score == pytest.approx(0.0)

    def test_evaluate_detection_perfect_mAP(self) -> None:
        evaluator = AccuracyEvaluator()
        categories = {
            "cat": (["c1", "c2"], {"c1", "c2"}),
            "dog": (["d1", "d2"], {"d1", "d2"}),
        }
        report = evaluator.evaluate_detection("m", categories)
        assert report.map_score == pytest.approx(1.0)

    def test_chipset_note_in_detection_report(self) -> None:
        evaluator = AccuracyEvaluator()
        report = evaluator.evaluate_detection("m", {"cat": (["c1"], {"c1"})})
        assert any("chipset" in n.lower() for n in report.notes)


# ══════════════════════════════════════════════════════════════════════════════
# Boundary / numeric precision tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNumericalEdgeCases:
    def test_ap_range_0_to_1(self) -> None:
        """AP must always be in [0, 1]."""
        import random
        random.seed(42)
        items = list(range(100))
        random.shuffle(items)
        relevant = set(random.sample(items, 30))
        ap = compute_average_precision(items, relevant)
        assert 0.0 <= ap <= 1.0

    def test_map_range_0_to_1(self) -> None:
        result = compute_map({
            "a": (["x1", "r1", "x2", "r2"], {"r1", "r2"}),
            "b": (["r3", "x3"], {"r3"}),
        })
        assert 0.0 <= result <= 1.0

    def test_top_k_error_range_0_to_1(self) -> None:
        preds = [[3, 1, 2, 0, 4], [0, 2, 1, 4, 3]]
        truth = [0, 4]
        err = compute_top_k_error(preds, truth, k=3)
        assert 0.0 <= err <= 1.0

    def test_large_dataset_consistent(self) -> None:
        """AP on large lists should still respect boundaries."""
        retrieved = list(range(1000))
        relevant = set(range(0, 1000, 10))  # every 10th item is relevant
        ap = compute_average_precision(retrieved, relevant)
        assert 0.0 <= ap <= 1.0

    def test_map_equals_ap_for_single_category(self) -> None:
        """mAP with one category equals AP of that category."""
        retrieved = ["a", "x", "b"]
        relevant = {"a", "b"}
        ap = compute_average_precision(retrieved, relevant)
        map_score = compute_map({"only_category": (retrieved, relevant)})
        assert map_score == pytest.approx(ap)


# ══════════════════════════════════════════════════════════════════════════════
# Reference Notes
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracyNotes:
    def test_metrics_keys_present(self) -> None:
        for key in ("mAP", "Top-1 error", "Top-5 error"):
            assert key in ACCURACY_NOTES["metrics"]

    def test_map_description_mentions_categories(self) -> None:
        desc = ACCURACY_NOTES["metrics"]["mAP"]["description"]
        assert "categor" in desc.lower()

    def test_top1_error_description_matches_docs(self) -> None:
        desc = ACCURACY_NOTES["metrics"]["Top-1 error"]["description"]
        assert "highest-probability" in desc.lower() or "highest" in desc

    def test_top5_error_description_matches_docs(self) -> None:
        desc = ACCURACY_NOTES["metrics"]["Top-5 error"]["description"]
        assert "5" in desc

    def test_ap_formula_variables_documented(self) -> None:
        formula = ACCURACY_NOTES["ap_formula"]
        for var in ("k", "n", "P(k)", "rel(k)"):
            assert var in formula["variables"]

    def test_ap_formula_edge_case_documented(self) -> None:
        edge = ACCURACY_NOTES["ap_formula"]["edge_case"]
        assert "0" in edge or "zero" in edge.lower()

    def test_ap_python_reference_matches_docs(self) -> None:
        ref = ACCURACY_NOTES["ap_formula"]["python_reference"]
        assert "img_sorted" in ref
        assert "anno_imgs" in ref
        assert "count/rank" in ref

    def test_chipset_note_present(self) -> None:
        note = ACCURACY_NOTES["chipset_note"]
        assert "chipset" in note.lower()
        assert "vary" in note.lower()

    def test_exported_from_accuracy_package(self) -> None:
        from quad.accuracy import (  # noqa: F401
            ACCURACY_NOTES,
            AccuracyEvaluator,
            AccuracyReport,
            compute_average_precision,
            compute_map,
            compute_top1_error,
            compute_top5_error,
        )
