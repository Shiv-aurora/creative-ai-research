from __future__ import annotations

from creativeai.schemas import TaskSpec


DAT_TEMPLATE = (
    "Task: Divergent Association Test style generation. "
    "Generate exactly 10 single-word nouns that are as semantically different from each other as possible. "
    "Return ONLY a JSON list of 10 strings."
)

CDAT_TEMPLATE = (
    "Task: Conditional Divergent Association (CDAT-style). "
    "Cue: \"{cue}\". "
    "Generate exactly 10 single-word nouns related to the cue while maximizing differences between the words. "
    "Return ONLY a JSON list of 10 strings."
)

AUT_TEMPLATE = (
    "Task: Alternative Uses ideation. "
    "Object: \"{object}\". Context: \"{context}\". "
    "Generate exactly 10 distinct, useful ideas for using the object in this context. "
    "Return ONLY a JSON list of 10 strings."
)



def build_dat_task() -> TaskSpec:
    return TaskSpec(
        task_id="dat",
        prompt_template=DAT_TEMPLATE,
        constraints=[
            "Output must be valid JSON.",
            "Exactly 10 items.",
            "Each item must be a single word noun.",
            "Avoid obvious synonyms or near-duplicates.",
        ],
        expected_format="json_word_list_10",
        metadata={"task_group": "word"},
    )



def build_cdat_task(cue: str) -> TaskSpec:
    return TaskSpec(
        task_id="cdat",
        prompt_template=CDAT_TEMPLATE,
        constraints=[
            "Output must be valid JSON.",
            "Exactly 10 items.",
            "Each item must be a single word noun.",
            "Every item must be meaningfully related to the cue.",
            "Maximize semantic dispersion across the 10 items.",
        ],
        expected_format="json_word_list_10",
        metadata={"task_group": "word", "cue": cue},
    )



def build_aut_task(obj: str, context: str) -> TaskSpec:
    return TaskSpec(
        task_id="aut",
        prompt_template=AUT_TEMPLATE,
        constraints=[
            "Output must be valid JSON.",
            "Exactly 10 items.",
            "Items should be concise idea phrases.",
            "Ideas must fit the object and context.",
            "Avoid repeated concepts.",
        ],
        expected_format="json_idea_list_10",
        metadata={"task_group": "ideation", "object": obj, "context": context},
    )



def build_task(task_id: str, cue: str | None = None, obj: str | None = None, context: str | None = None) -> TaskSpec:
    task_id = task_id.strip().lower()
    if task_id == "dat":
        return build_dat_task()
    if task_id == "cdat":
        if not cue:
            raise ValueError("CDAT requires --cue")
        return build_cdat_task(cue)
    if task_id == "aut":
        if not obj or not context:
            raise ValueError("AUT requires --object and --context")
        return build_aut_task(obj, context)
    raise ValueError(f"Unsupported task_id: {task_id}")
