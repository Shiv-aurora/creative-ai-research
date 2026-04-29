import unittest

from creativeai.validators import parse_json_list, validate_output, validate_word_list


class ValidatorsTest(unittest.TestCase):
    def test_parse_json_list_with_wrapped_text(self) -> None:
        raw = "Here is output:\n[\"alpha\", \"beta\"]\nThanks"
        parsed = parse_json_list(raw)
        self.assertEqual(parsed, ["alpha", "beta"])

    def test_validate_word_list_detects_problems(self) -> None:
        flags = validate_word_list(["alpha", "alpha", "bad token"])
        self.assertFalse(flags["valid"])
        self.assertIn("duplicate_items", flags["problems"])

    def test_validate_output_aut(self) -> None:
        out = [f"idea {i} for context" for i in range(10)]
        flags = validate_output("aut", out)
        self.assertTrue(flags["valid"])


if __name__ == "__main__":
    unittest.main()
