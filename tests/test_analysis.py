import unittest

from creativeai.analysis import (
    base_vs_instruct_shift,
    compare_backend_trend,
    compute_matched_summary,
    frontier_points,
    homogeneity_audit_from_runs,
    paired_method_deltas,
)
from creativeai.scoring import bootstrap_mean_ci


class AnalysisTest(unittest.TestCase):
    def test_bootstrap_ci_contains_mean(self) -> None:
        vals = [0.1, 0.2, 0.3, 0.4, 0.5]
        low, mean, high = bootstrap_mean_ci(vals, n_boot=300, seed=1)
        self.assertLessEqual(low, mean)
        self.assertGreaterEqual(high, mean)

    def test_frontier_points_grouping(self) -> None:
        rows = [
            {"model_id": "a", "method": "one_shot", "task_id": "cdat", "novelty": 0.4, "appropriateness": 0.6},
            {"model_id": "a", "method": "one_shot", "task_id": "cdat", "novelty": 0.5, "appropriateness": 0.5},
            {"model_id": "a", "method": "restlessness_loop", "task_id": "cdat", "novelty": 0.55, "appropriateness": 0.58},
        ]
        points, summary = frontier_points(rows)
        self.assertEqual(len(points), 2)
        self.assertEqual(len(summary), 2)

    def test_compare_backend_trend(self) -> None:
        local = [
            {"model_id": "a", "method": "m1", "task_id": "cdat", "objective_mean": 0.60},
            {"model_id": "a", "method": "m2", "task_id": "cdat", "objective_mean": 0.40},
            {"model_id": "b", "method": "m1", "task_id": "aut", "objective_mean": 0.70},
        ]
        cuda = [
            {"model_id": "a", "method": "m1", "task_id": "cdat", "objective_mean": 0.61},
            {"model_id": "a", "method": "m2", "task_id": "cdat", "objective_mean": 0.39},
            {"model_id": "b", "method": "m1", "task_id": "aut", "objective_mean": 0.71},
        ]
        out = compare_backend_trend(local, cuda)
        self.assertTrue(out["hardware_stable"])

    def test_shift_pairs_mistral_base_and_instruct(self) -> None:
        rows = [
            {"model_id": "mistral-7b-v0.3", "method": "one_shot", "task_id": "dat", "novelty": 0.7, "appropriateness": 0.7},
            {"model_id": "mistral-7b-instruct-v0.3", "method": "one_shot", "task_id": "dat", "novelty": 0.6, "appropriateness": 0.8},
        ]
        shifts = base_vs_instruct_shift(rows)
        self.assertEqual(len(shifts), 1)
        self.assertEqual(shifts[0]["family"], "mistral-7b-v0.3")

    def test_frontier_exclude_invalid(self) -> None:
        rows = [
            {
                "model_id": "a",
                "method": "one_shot",
                "task_id": "cdat",
                "novelty": 0.4,
                "appropriateness": 0.6,
                "valid_for_primary": True,
            },
            {
                "model_id": "a",
                "method": "one_shot",
                "task_id": "cdat",
                "novelty": 0.9,
                "appropriateness": 0.1,
                "valid_for_primary": False,
            },
        ]
        _, all_summary = frontier_points(rows, exclude_invalid=False)
        _, valid_summary = frontier_points(rows, exclude_invalid=True)
        self.assertEqual(all_summary[0]["sample_count"], 2)
        self.assertEqual(valid_summary[0]["sample_count"], 1)

    def test_paired_and_compute_matched(self) -> None:
        rows = [
            {
                "model_id": "m",
                "method": "one_shot",
                "task_id": "aut",
                "novelty": 0.50,
                "appropriateness": 0.50,
                "valid_for_primary": True,
                "metadata": {
                    "object": "brick",
                    "context": "power outage",
                    "generation_calls": 4,
                    "tokens_total": 380,
                },
                "compute_matched_valid": True,
            },
            {
                "model_id": "m",
                "method": "best_of_k_one_shot",
                "task_id": "aut",
                "novelty": 0.55,
                "appropriateness": 0.56,
                "valid_for_primary": True,
                "metadata": {
                    "object": "brick",
                    "context": "power outage",
                    "generation_calls": 4,
                    "tokens_total": 400,
                },
                "compute_matched_valid": True,
            },
            {
                "model_id": "m",
                "method": "restlessness_best",
                "task_id": "aut",
                "novelty": 0.58,
                "appropriateness": 0.58,
                "valid_for_primary": True,
                "metadata": {
                    "object": "brick",
                    "context": "power outage",
                    "generation_calls": 4,
                    "tokens_total": 410,
                },
                "compute_matched_valid": True,
            },
            {
                "model_id": "m",
                "method": "restlessness_adaptive",
                "task_id": "aut",
                "novelty": 0.57,
                "appropriateness": 0.57,
                "valid_for_primary": True,
                "metadata": {
                    "object": "brick",
                    "context": "power outage",
                    "generation_calls": 3,
                    "tokens_total": 390,
                },
                "compute_matched_valid": True,
            },
        ]
        paired = paired_method_deltas(rows, "restlessness_best", "one_shot")
        self.assertEqual(paired["n_pairs"], 1)
        matched = compute_matched_summary(rows, k=4, paired_by="prompt", token_tolerance=0.25)
        self.assertEqual(matched["primary_comparison"]["n_pairs"], 1)
        self.assertEqual(matched["adaptive_comparison"]["n_pairs"], 1)
        self.assertEqual(matched["baseline_comparison"]["n_pairs"], 1)

    def test_homogeneity_by_task(self) -> None:
        runs = [
            {"method": "one_shot", "task_id": "cdat", "output": ["alpha", "beta"]},
            {"method": "one_shot", "task_id": "aut", "output": ["idea one", "idea two"]},
            {"method": "restlessness_best", "task_id": "cdat", "output": ["gamma", "delta"]},
        ]
        pooled = homogeneity_audit_from_runs(runs, by_task=False)
        by_task = homogeneity_audit_from_runs(runs, by_task=True)
        self.assertGreater(len(by_task), len(pooled))
        self.assertTrue(all("task_id" in row for row in by_task))


if __name__ == "__main__":
    unittest.main()
