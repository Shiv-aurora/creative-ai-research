import unittest

from creativeai.scoring import compute_score_record, homogeneity_metrics, score_cdat, score_dat


class ScoringTest(unittest.TestCase):
    def test_cdat_relevance_changes_appropriateness(self) -> None:
        related = [
            "forestsignal",
            "forestmatrix",
            "forestharbor",
            "forestvector",
            "forestkernel",
            "forestorbit",
            "forestnetwork",
            "forestarchive",
            "forestchannel",
            "forestlattice",
        ]
        unrelated = [
            "quartz",
            "cipher",
            "vortex",
            "isotope",
            "lantern",
            "matrix",
            "nylon",
            "oasis",
            "plasma",
            "zephyr",
        ]
        rel = score_cdat(related, cue="forest")
        unr = score_cdat(unrelated, cue="forest")
        self.assertGreater(rel["appropriateness"], unr["appropriateness"])

    def test_novelty_sanity(self) -> None:
        diverse = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
        repeated = ["alpha"] * 10
        d = score_dat(diverse)
        r = score_dat(repeated)
        self.assertGreater(d["novelty"], r["novelty"])

    def test_high_novelty_low_appropriateness_flag(self) -> None:
        run = {
            "run_id": "r1",
            "task_id": "cdat",
            "method": "one_shot",
            "model_id": "m",
            "metadata": {"cue": "forest"},
            "output": [
                "quartz",
                "cipher",
                "vortex",
                "isotope",
                "lantern",
                "matrix",
                "nylon",
                "oasis",
                "plasma",
                "zephyr",
            ],
            "validity_flags": {},
        }
        scored = compute_score_record(run)
        # The flag should exist even if value depends on scoring surface.
        self.assertIn("high_novelty_low_appropriateness", scored.validity_flags)

    def test_homogeneity_duplicate_texts(self) -> None:
        texts = ["same output text"] * 20
        metrics = homogeneity_metrics(texts)
        self.assertGreater(metrics["nearest_neighbor_similarity"], 0.95)
        self.assertLess(metrics["diversity_index"], 0.2)

    def test_cdat_overlap_penalty_reduces_cheat_signal(self) -> None:
        cheat = [
            "forestcore",
            "foresthub",
            "forestline",
            "forestnode",
            "forestpath",
            "forestmap",
            "forestgrid",
            "forestlink",
            "forestmesh",
            "forestbase",
        ]
        penalized = score_cdat(cheat, cue="forest", overlap_penalty=True)
        unpenalized = score_cdat(cheat, cue="forest", overlap_penalty=False)
        self.assertGreater(penalized["anti_cheat_penalty"], 0.0)
        self.assertLess(penalized["appropriateness"], unpenalized["appropriateness"])

    def test_score_record_marks_invalid_primary_when_json_invalid(self) -> None:
        run = {
            "run_id": "r2",
            "task_id": "dat",
            "method": "one_shot",
            "model_id": "m",
            "metadata": {},
            "output": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"],
            "validity_flags": {"valid": True, "problems": []},
            "json_valid": False,
        }
        scored = compute_score_record(run)
        self.assertFalse(scored.valid_for_primary)
        self.assertIn("non_json_output", scored.validity_flags.get("problems", []))

    def test_score_record_emits_compute_efficiency_fields(self) -> None:
        run = {
            "run_id": "r3",
            "task_id": "aut",
            "method": "restlessness_best",
            "model_id": "m",
            "metadata": {
                "object": "brick",
                "context": "during a power outage",
                "generation_calls": 4,
                "tokens_total": 420,
            },
            "output": [
                "Build a stable signal marker from a brick and reflective tape",
                "Use the brick as an anchor for a makeshift lantern stand",
                "Create a thermal buffer around hot cookware",
                "Use as a door wedge to keep evacuation paths open",
                "Form a counterweight in a pulley demo",
                "Create a simple handwashing station base",
                "Use as a spacing block while assembling shelves",
                "Build a small teaching model for load distribution",
                "Use as a stabilizer for emergency signage",
                "Create a temporary tool rest",
            ],
            "validity_flags": {"valid": True, "json_valid": True},
            "json_valid": True,
        }
        scored = compute_score_record(run)
        self.assertGreater(scored.score_per_1k_tokens, 0.0)
        self.assertTrue(scored.compute_matched_valid)


if __name__ == "__main__":
    unittest.main()
