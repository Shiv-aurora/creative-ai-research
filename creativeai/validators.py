from __future__ import annotations

import json
import re
from typing import Any

WORD_RE = re.compile(r"^[A-Za-z][A-Za-z-]*$")



def _extract_first_json_array(raw: str) -> str:
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON array found in model output")
    return raw[start : end + 1]



def parse_json_list(raw: str) -> list[str]:
    candidate = raw.strip()
    if not candidate.startswith("["):
        candidate = _extract_first_json_array(candidate)
    parsed = json.loads(candidate)
    if not isinstance(parsed, list):
        raise ValueError("Model output JSON must be a list")
    out: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            raise ValueError("Every output item must be a string")
        cleaned = item.strip()
        if cleaned:
            out.append(cleaned)
    return out



def validate_word_list(words: list[str], expected_count: int = 10) -> dict[str, Any]:
    problems: list[str] = []
    if len(words) != expected_count:
        problems.append(f"expected_{expected_count}_items_got_{len(words)}")

    normalized = [w.strip().lower() for w in words]
    if len(set(normalized)) != len(normalized):
        problems.append("duplicate_items")

    for word in words:
        if " " in word.strip():
            problems.append("contains_multiword_item")
            break

    for word in words:
        if not WORD_RE.match(word.strip()):
            problems.append("invalid_token_pattern")
            break

    return {
        "valid": len(problems) == 0,
        "problems": sorted(set(problems)),
    }



def validate_idea_list(ideas: list[str], expected_count: int = 10) -> dict[str, Any]:
    problems: list[str] = []
    if len(ideas) != expected_count:
        problems.append(f"expected_{expected_count}_items_got_{len(ideas)}")
    normalized = [i.strip().lower() for i in ideas]
    if len(set(normalized)) != len(normalized):
        problems.append("duplicate_items")
    if any(len(i.strip()) < 3 for i in ideas):
        problems.append("too_short_items")
    return {
        "valid": len(problems) == 0,
        "problems": sorted(set(problems)),
    }



def validate_output(task_id: str, output: list[str]) -> dict[str, Any]:
    task = task_id.lower()
    if task in {"dat", "cdat"}:
        return validate_word_list(output)
    if task == "aut":
        return validate_idea_list(output)
    return {"valid": False, "problems": ["unknown_task"]}
