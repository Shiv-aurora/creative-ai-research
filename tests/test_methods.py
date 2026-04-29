import unittest

from creativeai.methods import BestOfKOneShotRunner
from creativeai.methods import BrainstormSelectRunner, OneShotRunner, RestlessnessAdaptiveRunner, RestlessnessRunner
from creativeai.model_backend import create_model_adapter
from creativeai.schemas import GenerationConfig
from creativeai.tasks import build_task


class MethodsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = GenerationConfig(
            model_id="gemma-2-2b",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
        )
        self.model = create_model_adapter("gemma-2-2b", backend="mock")

    def test_one_shot_deterministic(self) -> None:
        task = build_task("dat")
        runner = OneShotRunner()
        a = runner.run(task, self.cfg, self.model)
        b = runner.run(task, self.cfg, self.model)
        self.assertEqual(a.output, b.output)

    def test_restlessness_constraints_are_task_relevant_and_vary(self) -> None:
        task = build_task("cdat", cue="forest")
        runner = RestlessnessRunner(iterations=3)
        result = runner.run(task, self.cfg, self.model)

        iterations = [t for t in result.raw_trace if t["step"].startswith("iteration_")]
        self.assertEqual(len(iterations), 3)
        constraint_sets = [tuple(i.get("constraints", [])) for i in iterations]
        self.assertEqual(len(set(constraint_sets)), 3)
        self.assertTrue(any("forest" in " ".join(c).lower() for c in constraint_sets))

    def test_brainstorm_select_outputs_ten_ideas(self) -> None:
        task = build_task("aut", obj="brick", context="during a power outage")
        runner = BrainstormSelectRunner()
        result = runner.run(task, self.cfg, self.model)
        self.assertEqual(len(result.output), 10)

    def test_strict_json_retries_and_recovers(self) -> None:
        class SequenceModel:
            model_id = "seq"
            backend = "mock"
            model_hash = "seq-hash"

            def __init__(self, responses: list[str]) -> None:
                self.responses = responses
                self.idx = 0

            def generate(self, prompt: str, config: GenerationConfig) -> str:
                out = self.responses[min(self.idx, len(self.responses) - 1)]
                self.idx += 1
                return out

        responses = [
            "not json",
            '["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]',
        ]
        cfg = GenerationConfig(
            model_id="seq-model",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            strict_json=True,
            max_retries=1,
        )
        runner = OneShotRunner()
        result = runner.run(build_task("dat"), cfg, SequenceModel(responses))
        self.assertTrue(result.json_valid)
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.parse_mode, "json_strict")

    def test_strict_json_marks_failure_after_retries(self) -> None:
        class SequenceModel:
            model_id = "seq"
            backend = "mock"
            model_hash = "seq-hash"

            def __init__(self, response: str) -> None:
                self.response = response

            def generate(self, prompt: str, config: GenerationConfig) -> str:
                return self.response

        cfg = GenerationConfig(
            model_id="seq-model",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            strict_json=True,
            max_retries=1,
        )
        runner = OneShotRunner()
        result = runner.run(build_task("dat"), cfg, SequenceModel("still not json output"))
        self.assertFalse(result.json_valid)
        self.assertEqual(result.parse_mode, "strict_retry_exhausted_parse")
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(result.output, [])

    def test_strict_json_rejects_empty_array_without_fabrication(self) -> None:
        class SequenceModel:
            model_id = "seq"
            backend = "mock"
            model_hash = "seq-hash"

            def generate(self, prompt: str, config: GenerationConfig) -> str:
                return "[]"

        cfg = GenerationConfig(
            model_id="seq-model",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            strict_json=True,
            max_retries=1,
        )
        runner = OneShotRunner()
        result = runner.run(build_task("cdat", cue="forest"), cfg, SequenceModel())
        self.assertTrue(result.json_valid)
        self.assertEqual(result.parse_mode, "strict_retry_exhausted_schema")
        self.assertEqual(result.output, [])

    def test_strict_json_retry_increments_seed(self) -> None:
        class SequenceModel:
            model_id = "seq"
            backend = "mock"
            model_hash = "seq-hash"

            def __init__(self) -> None:
                self.seeds: list[int] = []
                self.idx = 0

            def generate(self, prompt: str, config: GenerationConfig) -> str:
                self.seeds.append(config.seed)
                self.idx += 1
                if self.idx == 1:
                    return "not json"
                return '["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]'

        cfg = GenerationConfig(
            model_id="seq-model",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            strict_json=True,
            max_retries=2,
        )
        model = SequenceModel()
        runner = OneShotRunner()
        result = runner.run(build_task("dat"), cfg, model)
        self.assertTrue(result.json_valid)
        self.assertEqual(model.seeds[:2], [11, 12])

    def test_restlessness_best_vs_last_iter_can_diverge(self) -> None:
        class SequenceModel:
            model_id = "seq"
            backend = "mock"
            model_hash = "seq-hash"

            def __init__(self, responses: list[str]) -> None:
                self.responses = responses
                self.idx = 0

            def generate(self, prompt: str, config: GenerationConfig) -> str:
                out = self.responses[min(self.idx, len(self.responses) - 1)]
                self.idx += 1
                return out

        initial = '["forest","tree","leaf","wood","moss","bark","fern","branch","root","grove"]'
        iter1 = '["canopy","deer","owl","lichen","trail","stream","acorn","sapling","fungus","thicket"]'
        iter2 = '["forestcore","foresthub","forestline","forestnode","forestpath","forestmap","forestgrid","forestlink","forestmesh","forestbase"]'
        iter3 = iter2
        responses = [initial, iter1, iter2, iter3]

        cfg = GenerationConfig(
            model_id="seq-model",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            strict_json=True,
            max_retries=1,
        )
        task = build_task("cdat", cue="forest")
        best_runner = RestlessnessRunner(iterations=3, return_strategy="best")
        last_runner = RestlessnessRunner(iterations=3, return_strategy="last")

        best = best_runner.run(task, cfg, SequenceModel(responses))
        last = last_runner.run(task, cfg, SequenceModel(responses))
        self.assertNotEqual(best.output, last.output)
        self.assertEqual(best_runner.method_name, "restlessness_best")
        self.assertEqual(last_runner.method_name, "restlessness_last_iter")

    def test_best_of_k_runner_tracks_generation_calls(self) -> None:
        runner = BestOfKOneShotRunner(k=4)
        task = build_task("dat")
        result = runner.run(task, self.cfg, self.model)
        self.assertEqual(result.generation_calls, 4)
        self.assertEqual(len(result.candidate_objectives), 4)

    def test_task_bound_grammar_mode_is_set(self) -> None:
        class InspectModel:
            model_id = "inspect"
            backend = "mock"
            model_hash = "inspect-hash"

            def __init__(self) -> None:
                self.grammar_modes: list[str] = []

            def generate(self, prompt: str, config: GenerationConfig) -> str:
                self.grammar_modes.append(str(getattr(config, "grammar_mode", "")))
                if "Object:" in prompt:
                    return '["build rack","make stand","signal marker","rope anchor","light diffuser","water funnel","heat guard","tool spacer","noise dampener","sorting tray"]'
                return '["alpha","beta","gamma","delta","epsilon","zeta","eta","theta","iota","kappa"]'

        model = InspectModel()
        cfg = GenerationConfig(
            model_id="inspect-model",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            strict_json=True,
            max_retries=0,
            grammar_mode="auto",
        )

        OneShotRunner().run(build_task("dat"), cfg, model)
        OneShotRunner().run(build_task("aut", obj="brick", context="power outage"), cfg, model)
        self.assertIn("word_list", model.grammar_modes)
        self.assertIn("idea_list", model.grammar_modes)

    def test_restlessness_adaptive_stops_early(self) -> None:
        class SequenceModel:
            model_id = "seq"
            backend = "mock"
            model_hash = "seq-hash"

            def __init__(self, responses: list[str]) -> None:
                self.responses = responses
                self.idx = 0

            def generate(self, prompt: str, config: GenerationConfig) -> str:
                out = self.responses[min(self.idx, len(self.responses) - 1)]
                self.idx += 1
                return out

        # iteration_1 improves strongly; iteration_2 barely changes -> triggers adaptive stop
        initial = '["forest","tree","leaf","wood","moss","bark","fern","branch","root","grove"]'
        iter1 = '["canopy","deer","owl","lichen","trail","stream","acorn","sapling","fungus","thicket"]'
        iter2 = '["canopy","owl","trail","stream","acorn","sapling","fungus","thicket","deer","lichen"]'
        responses = [initial, iter1, iter2, iter2, iter2]

        cfg = GenerationConfig(
            model_id="seq-model",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=256,
            strict_json=True,
            max_retries=0,
            adaptive_stop_delta=0.02,
            adaptive_min_iters=2,
        )
        task = build_task("cdat", cue="forest")
        runner = RestlessnessAdaptiveRunner(iterations=3)
        result = runner.run(task, cfg, SequenceModel(responses))
        self.assertTrue(any(t.get("step") == "adaptive_stop" for t in result.raw_trace))
        self.assertLess(result.generation_calls, 4)
        self.assertGreater(result.tokens_in, 0)
        self.assertGreater(result.tokens_out, 0)


if __name__ == "__main__":
    unittest.main()
