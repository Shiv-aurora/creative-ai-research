import unittest

from creativeai.model_backend import (
    _configured_sampler_kwargs,
    _filter_supported_kwargs,
    _json_array_grammar_exact_count,
    _json_idea_array_grammar_exact_count,
    _json_word_array_grammar_exact_count,
    _resolve_prompt_mode,
    _select_json_grammar,
)
from creativeai.schemas import GenerationConfig


class ModelBackendTest(unittest.TestCase):
    def test_prompt_mode_auto_prefers_chat_for_instruct(self) -> None:
        self.assertEqual(_resolve_prompt_mode("qwen2.5-3b-instruct", "auto"), "chat")
        self.assertEqual(_resolve_prompt_mode("gemma-2b-it", "auto"), "chat")
        self.assertEqual(_resolve_prompt_mode("gemma-2b", "auto"), "completion")

    def test_prompt_mode_explicit_is_respected(self) -> None:
        self.assertEqual(_resolve_prompt_mode("gemma-2b", "chat"), "chat")
        self.assertEqual(_resolve_prompt_mode("gemma-2b-instruct", "completion"), "completion")

    def test_json_grammar_enforces_fixed_item_count(self) -> None:
        grammar = _json_array_grammar_exact_count(10)
        self.assertNotIn("elements?", grammar)
        self.assertEqual(grammar.count('","'), 9)
        self.assertNotIn("ws ::=", grammar)
        self.assertIn('string ::= "\\"" c1 c2 c3 tail "\\""', grammar)
        self.assertIn("tail ::= [A-Za-z0-9 .,;:!?()'/-]*", grammar)

    def test_word_json_grammar_restricts_to_letters(self) -> None:
        grammar = _json_word_array_grammar_exact_count(10)
        self.assertIn('qword ::= "\\"" word "\\""', grammar)
        self.assertIn("word ::= [A-Za-z] [A-Za-z-]*", grammar)
        self.assertEqual(grammar.count('","'), 9)

    def test_selects_word_grammar_by_mode(self) -> None:
        grammar = _select_json_grammar("word_list", count=10)
        self.assertIn("word ::=", grammar)

    def test_selects_idea_grammar_by_mode(self) -> None:
        grammar = _select_json_grammar("idea_list", count=10)
        self.assertIn("qidea ::=", grammar)
        self.assertIn("i1 ::= [A-Za-z0-9]", grammar)

    def test_idea_grammar_enforces_count(self) -> None:
        grammar = _json_idea_array_grammar_exact_count(10)
        self.assertEqual(grammar.count('","'), 9)

    def test_configured_sampler_kwargs_omits_defaults(self) -> None:
        cfg = GenerationConfig(
            model_id="m",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=128,
        )
        self.assertEqual(_configured_sampler_kwargs(cfg), {})

    def test_configured_sampler_kwargs_includes_phase4_controls(self) -> None:
        cfg = GenerationConfig(
            model_id="m",
            backend="mock",
            temperature=0.7,
            top_p=0.9,
            seed=11,
            max_tokens=128,
            top_k=80,
            min_p=0.05,
            repeat_penalty=1.1,
            mirostat_mode=2,
            mirostat_tau=5.0,
            mirostat_eta=0.1,
        )
        kwargs = _configured_sampler_kwargs(cfg)
        self.assertEqual(kwargs["top_k"], 80)
        self.assertEqual(kwargs["min_p"], 0.05)
        self.assertEqual(kwargs["repeat_penalty"], 1.1)
        self.assertEqual(kwargs["mirostat_mode"], 2)

    def test_filter_supported_kwargs_drops_unsupported_sampler_args(self) -> None:
        def fake_completion(prompt: str, temperature: float, top_p: float, max_tokens: int, seed: int) -> None:
            return None

        filtered = _filter_supported_kwargs(
            fake_completion,
            {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 128,
                "seed": 11,
                "top_k": 80,
            },
        )
        self.assertNotIn("top_k", filtered)
        self.assertIn("temperature", filtered)


if __name__ == "__main__":
    unittest.main()
