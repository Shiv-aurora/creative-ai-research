import unittest
from dataclasses import asdict

from creativeai.decoding import apply_sampler_profile, decoding_fingerprint, decoding_settings
from creativeai.pipeline import _compute_group_id
from creativeai.schemas import GenerationConfig, TaskSpec


class DecodingTest(unittest.TestCase):
    def test_generation_config_serializes_sampler_fields(self) -> None:
        cfg = GenerationConfig(
            model_id="m",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=128,
            sampler_profile="spread_topk_minp",
            top_k=80,
            min_p=0.05,
        )
        payload = asdict(cfg)
        self.assertEqual(payload["sampler_profile"], "spread_topk_minp")
        self.assertEqual(payload["top_k"], 80)
        self.assertEqual(payload["min_p"], 0.05)

    def test_sampler_profile_overrides_config(self) -> None:
        cfg = GenerationConfig(
            model_id="m",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=128,
            sampler_profile="low_temp",
        )
        applied = apply_sampler_profile(cfg)
        self.assertEqual(applied.temperature, 0.2)
        self.assertEqual(applied.sampler_profile, "low_temp")

    def test_decoding_fingerprint_changes_with_sampler_settings(self) -> None:
        base = GenerationConfig("m", "mock", 0.7, 0.9, 11, 128, sampler_profile="manual")
        changed = GenerationConfig("m", "mock", 0.7, 0.9, 11, 128, sampler_profile="manual", top_k=80)
        self.assertNotEqual(decoding_fingerprint(base), decoding_fingerprint(changed))
        self.assertEqual(decoding_settings(changed)["top_k"], 80)

    def test_compute_group_id_changes_with_sampler_settings(self) -> None:
        task = TaskSpec("dat", "Prompt", [], "json", {"dat_prompt_id": 0})
        base = GenerationConfig("m", "mock", 0.7, 0.9, 11, 128, sampler_profile="manual")
        changed = GenerationConfig("m", "mock", 0.7, 0.9, 11, 128, sampler_profile="manual", top_k=80)
        self.assertNotEqual(_compute_group_id(task, base, task.metadata), _compute_group_id(task, changed, task.metadata))


if __name__ == "__main__":
    unittest.main()
