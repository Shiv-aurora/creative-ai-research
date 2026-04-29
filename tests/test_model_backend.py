import unittest

from creativeai.model_backend import (
    _json_array_grammar_exact_count,
    _json_idea_array_grammar_exact_count,
    _json_word_array_grammar_exact_count,
    _resolve_prompt_mode,
    _select_json_grammar,
)


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


if __name__ == "__main__":
    unittest.main()
