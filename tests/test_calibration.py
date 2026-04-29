import unittest

from creativeai.calibration import evaluate_human_calibration


class CalibrationTest(unittest.TestCase):
    def test_calibration_gate_passes_for_high_correlation(self) -> None:
        ratings = []
        for i in range(12):
            val = i / 11.0
            ratings.append(
                {
                    "auto_appropriateness": val,
                    "auto_usefulness": val,
                    "human1_appropriateness": val,
                    "human2_appropriateness": val,
                    "human1_usefulness": val,
                    "human2_usefulness": val,
                }
            )
        out = evaluate_human_calibration(ratings, target_corr=0.45)
        self.assertTrue(out["gate_pass"])
        self.assertGreaterEqual(out["spearman_auto_vs_human_appropriateness"], 0.95)

    def test_calibration_gate_fails_for_low_correlation(self) -> None:
        ratings = []
        for i in range(12):
            auto = i / 11.0
            human = (11 - i) / 11.0
            ratings.append(
                {
                    "auto_appropriateness": auto,
                    "auto_usefulness": auto,
                    "human1_appropriateness": human,
                    "human2_appropriateness": human,
                    "human1_usefulness": human,
                    "human2_usefulness": human,
                }
            )
        out = evaluate_human_calibration(ratings, target_corr=0.45)
        self.assertFalse(out["gate_pass"])


if __name__ == "__main__":
    unittest.main()
