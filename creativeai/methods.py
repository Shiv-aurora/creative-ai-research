from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from creativeai.embeddings import cosine_similarity, text_embedding
from creativeai.io_utils import token_count_text
from creativeai.model_backend import ModelAdapter
from creativeai.scoring import frontier_objective, score_aut, score_cdat, score_dat, selection_objective
from creativeai.schemas import GenerationConfig, TaskSpec
from creativeai.validators import parse_json_list, validate_output


@dataclass
class MethodResult:
    prompt: str
    output: list[str]
    raw_trace: list[dict[str, Any]]
    parse_mode: str = "unknown"
    json_valid: bool = False
    retry_count: int = 0
    candidate_objectives: list[float] = field(default_factory=list)
    generation_calls: int = 1
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class GeneratedList:
    output: list[str]
    raw: str
    json_valid: bool
    parse_mode: str
    retry_count: int
    prompt_tokens: int = 0
    output_tokens: int = 0
    attempts_made: int = 1


class MethodRunner:
    method_name = "base"

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        raise NotImplementedError

    def compute_cost(self, run_record: dict[str, Any]) -> float:
        tokens = int(run_record.get("tokens_total", 0) or run_record.get("metadata", {}).get("tokens_total", 0) or 0)
        if tokens > 0:
            return float(tokens)
        calls = int(run_record.get("effective_calls", 0) or run_record.get("metadata", {}).get("generation_calls", 1) or 1)
        return float(max(1, calls))


def _task_token_cap(task_id: str) -> int:
    if task_id in {"dat", "cdat"}:
        return 128
    if task_id == "aut":
        return 256
    return 256


def _task_grammar_mode(task_id: str) -> str:
    if task_id in {"dat", "cdat"}:
        return "word_list"
    if task_id == "aut":
        return "idea_list"
    return "auto"


def _normalize_output(task_id: str, parsed: list[str]) -> list[str]:
    out = [item.strip() for item in parsed if item.strip()]
    if task_id in {"dat", "cdat"}:
        return [x.lower() for x in out]
    return out


def _fallback_parse(raw: str) -> list[str]:
    return [p.strip(" -\t\n") for p in re.split(r"[,\n]+", raw) if p.strip()]


def _json_list_text(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=True)


def _repair_prompt(
    prompt: str,
    task_id: str,
    fail_kind: str,
    fail_problems: list[str] | None = None,
    bad_output: list[str] | None = None,
) -> str:
    problems = fail_problems or []
    bad_output = bad_output or []
    compact_bad = ", ".join(bad_output[:10])
    feedback: list[str] = []
    if fail_kind == "parse_error":
        feedback.append("Previous output was invalid or truncated JSON.")
    elif fail_kind == "schema_error":
        feedback.append("Previous output was valid JSON but failed task constraints.")
    if problems:
        feedback.append("Failure reasons: " + ", ".join(sorted(set(problems))) + ".")
    if "duplicate_items" in problems and bad_output:
        dups = [item for item, cnt in Counter(bad_output).items() if cnt > 1]
        if dups:
            feedback.append("Duplicate tokens are forbidden in next output: " + ", ".join(dups[:10]) + ".")
    if compact_bad:
        feedback.append("Do not repeat this invalid pattern: " + compact_bad + ".")

    if task_id in {"dat", "cdat"}:
        return (
            prompt
            + "\n\nRETRY: fix your previous output."
            + ("\n" + "\n".join(feedback) if feedback else "")
            + "\nOutput MUST be minified JSON with no extra text."
            + '\nExact format: ["alpha","beta","gamma","delta","ember","falcon","granite","harbor","island","jungle"]'
            + "\nExactly 10 UNIQUE single-word nouns, all alphabetic (hyphen allowed), cue-related."
            + "\nNo placeholders like item/word/token. No duplicates. No numbers. No extra whitespace outside JSON."
        )
    return (
        prompt
        + "\n\nRETRY: fix your previous output."
        + ("\n" + "\n".join(feedback) if feedback else "")
        + "\nOutput MUST be minified JSON with no extra text."
        + '\nExact format: ["idea1","idea2","idea3","idea4","idea5","idea6","idea7","idea8","idea9","idea10"]'
        + "\nExactly 10 UNIQUE concise ideas; each must be meaningful and context-fit."
        + "\nNo duplicates. No placeholders. No extra whitespace outside JSON."
    )


def _generate_list(model: ModelAdapter, prompt: str, config: GenerationConfig, task_id: str) -> GeneratedList:
    max_tokens = config.max_tokens
    if config.token_budget_per_prompt is not None and int(config.token_budget_per_prompt) > 0:
        max_tokens = min(max_tokens, int(config.token_budget_per_prompt))
    effective_cfg = replace(
        config,
        max_tokens=min(max_tokens, _task_token_cap(task_id)),
        grammar_mode=_task_grammar_mode(task_id),
    )
    attempts = max(0, int(effective_cfg.max_retries)) + 1

    last_raw = ""
    last_parsed: list[str] = []
    saw_schema_error = False
    last_problems: list[str] = []
    last_fail_kind = "parse_error"
    prompt_tokens_total = 0
    output_tokens_total = 0
    attempts_made = 0
    for attempt in range(attempts):
        attempt_prompt = (
            prompt
            if attempt == 0
            else _repair_prompt(
                prompt,
                task_id,
                fail_kind=last_fail_kind,
                fail_problems=last_problems,
                bad_output=last_parsed,
            )
        )
        attempt_cfg = replace(effective_cfg, seed=effective_cfg.seed + attempt)
        last_raw = model.generate(attempt_prompt, attempt_cfg)
        attempts_made = attempt + 1
        prompt_tokens_total += token_count_text(attempt_prompt)
        output_tokens_total += token_count_text(last_raw)
        try:
            parsed = parse_json_list(last_raw)
            normalized = _normalize_output(task_id, parsed)
            flags = validate_output(task_id, normalized)
            if flags.get("valid", False):
                return GeneratedList(
                    output=normalized,
                    raw=last_raw,
                    json_valid=True,
                    parse_mode="json_strict",
                    retry_count=attempt,
                    prompt_tokens=prompt_tokens_total,
                    output_tokens=output_tokens_total,
                    attempts_made=attempts_made,
                )
            saw_schema_error = True
            last_parsed = normalized
            last_problems = list(flags.get("problems", []))
            last_fail_kind = "schema_error"
            if not effective_cfg.strict_json:
                return GeneratedList(
                    output=normalized,
                    raw=last_raw,
                    json_valid=True,
                    parse_mode="json_invalid_schema",
                    retry_count=attempt,
                    prompt_tokens=prompt_tokens_total,
                    output_tokens=output_tokens_total,
                    attempts_made=attempts_made,
                )
            continue
        except Exception:
            saw_schema_error = False
            last_parsed = []
            last_problems = ["non_json_output"]
            last_fail_kind = "parse_error"
            if not effective_cfg.strict_json:
                fallback = _normalize_output(task_id, _fallback_parse(last_raw))
                return GeneratedList(
                    output=fallback,
                    raw=last_raw,
                    json_valid=False,
                    parse_mode="fallback_text",
                    retry_count=attempt,
                    prompt_tokens=prompt_tokens_total,
                    output_tokens=output_tokens_total,
                    attempts_made=attempts_made,
                )
            continue

    if not effective_cfg.strict_json:
        return GeneratedList(
            output=_normalize_output(task_id, _fallback_parse(last_raw)),
            raw=last_raw,
            json_valid=False,
            parse_mode="fallback_text",
            retry_count=max(0, int(effective_cfg.max_retries)),
            prompt_tokens=prompt_tokens_total,
            output_tokens=output_tokens_total,
            attempts_made=max(1, attempts_made),
        )

    parse_mode = "strict_retry_exhausted_schema" if saw_schema_error else "strict_retry_exhausted_parse"
    return GeneratedList(
        output=last_parsed if saw_schema_error else [],
        raw=last_raw,
        json_valid=saw_schema_error,
        parse_mode=parse_mode,
        retry_count=max(0, int(effective_cfg.max_retries)),
        prompt_tokens=prompt_tokens_total,
        output_tokens=output_tokens_total,
        attempts_made=max(1, attempts_made),
    )


def _candidate_objective(task: TaskSpec, candidate: list[str], mode: str = "selection") -> tuple[float, dict[str, Any]]:
    task_id = task.task_id
    if task_id == "dat":
        metrics = score_dat(candidate)
    elif task_id == "cdat":
        cue = str(task.metadata.get("cue", ""))
        metrics = score_cdat(candidate, cue)
    elif task_id == "aut":
        obj = str(task.metadata.get("object", ""))
        context = str(task.metadata.get("context", ""))
        metrics = score_aut(candidate, object_name=obj, context=context)
    else:
        metrics = {"novelty": 0.0, "appropriateness": 0.0, "usefulness": 0.0}

    if mode == "frontier":
        objective = frontier_objective(
            float(metrics.get("novelty", 0.0)),
            float(metrics.get("appropriateness", 0.0)),
        )
    else:
        objective = selection_objective(task_id, metrics)
    return objective, metrics


class OneShotRunner(MethodRunner):
    method_name = "one_shot"

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        prompt = task_spec.render_prompt(task_spec.metadata)
        generated = _generate_list(model, prompt, config, task_spec.task_id)
        objective, _ = _candidate_objective(task_spec, generated.output, mode="selection")
        return MethodResult(
            prompt=prompt,
            output=generated.output,
            parse_mode=generated.parse_mode,
            json_valid=generated.json_valid,
            retry_count=generated.retry_count,
            candidate_objectives=[objective],
            generation_calls=generated.attempts_made,
            tokens_in=generated.prompt_tokens,
            tokens_out=generated.output_tokens,
            raw_trace=[
                {
                    "step": "one_shot",
                    "prompt": prompt,
                    "raw_model_text": generated.raw,
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                }
            ],
        )


class BestOfKOneShotRunner(MethodRunner):
    method_name = "best_of_k_one_shot"

    def __init__(self, k: int = 4) -> None:
        self.k = max(1, k)

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        prompt = task_spec.render_prompt(task_spec.metadata)
        traces: list[dict[str, Any]] = []
        candidate_objs: list[float] = []
        generation_calls = 0
        tokens_in = 0
        tokens_out = 0

        best_gen: GeneratedList | None = None
        best_obj = float("-inf")

        for i in range(self.k):
            cfg = replace(config, seed=config.seed + i)
            generated = _generate_list(model, prompt, cfg, task_spec.task_id)
            generation_calls += generated.attempts_made
            tokens_in += generated.prompt_tokens
            tokens_out += generated.output_tokens
            objective, metrics = _candidate_objective(task_spec, generated.output, mode="selection")
            candidate_objs.append(objective)
            traces.append(
                {
                    "step": f"sample_{i + 1}",
                    "seed": cfg.seed,
                    "raw_model_text": generated.raw,
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                    "objective": objective,
                    "metrics": metrics,
                }
            )
            if objective > best_obj:
                best_obj = objective
                best_gen = generated

        assert best_gen is not None
        traces.append({"step": "selection", "selected_objective": best_obj, "k": self.k})
        return MethodResult(
            prompt=prompt,
            output=best_gen.output,
            raw_trace=traces,
            parse_mode=best_gen.parse_mode,
            json_valid=best_gen.json_valid,
            retry_count=best_gen.retry_count,
            candidate_objectives=candidate_objs,
            generation_calls=generation_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


class BrainstormSelectRunner(MethodRunner):
    method_name = "brainstorm_then_select"

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        if task_spec.task_id != "aut":
            return OneShotRunner().run(task_spec, config, model)

        prompt = (
            task_spec.render_prompt(task_spec.metadata)
            + "\n- First produce broad options, including unusual but feasible ideas."
            + "\n- Try to provide 20 candidates when possible."
        )

        pool: list[str] = []
        traces: list[dict[str, Any]] = []
        json_valid_any = False
        retry_count = 0
        parse_mode = "unknown"
        generation_calls = 0
        tokens_in = 0
        tokens_out = 0

        for idx, delta in enumerate([0, 10_000]):
            cfg = replace(config, seed=config.seed + delta)
            generated = _generate_list(model, prompt, cfg, task_spec.task_id)
            generation_calls += generated.attempts_made
            tokens_in += generated.prompt_tokens
            tokens_out += generated.output_tokens
            traces.append(
                {
                    "step": f"brainstorm_{idx}",
                    "seed": cfg.seed,
                    "raw_model_text": generated.raw,
                    "candidate_count": len(generated.output),
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                }
            )
            pool.extend(generated.output)
            json_valid_any = json_valid_any or generated.json_valid
            retry_count = max(retry_count, generated.retry_count)
            parse_mode = generated.parse_mode

        pool = list(dict.fromkeys(pool))
        if not pool:
            pool = []

        selected = _select_diverse_useful_ideas(
            pool,
            object_name=str(task_spec.metadata.get("object", "")),
            context=str(task_spec.metadata.get("context", "")),
            count=10,
        )

        objective, _ = _candidate_objective(task_spec, selected, mode="selection")
        traces.append(
            {
                "step": "selection",
                "pool_size": len(pool),
                "selected": selected,
                "objective": objective,
            }
        )

        return MethodResult(
            prompt=prompt,
            output=selected,
            raw_trace=traces,
            parse_mode=parse_mode,
            json_valid=json_valid_any,
            retry_count=retry_count,
            candidate_objectives=[objective],
            generation_calls=generation_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


def _select_diverse_useful_ideas(pool: list[str], object_name: str, context: str, count: int = 10) -> list[str]:
    if len(pool) <= count:
        return pool[:]

    from creativeai.scoring import idea_usefulness_proxy

    scored = [
        (
            item,
            idea_usefulness_proxy(item, object_name=object_name, context=context),
            text_embedding(item),
        )
        for item in pool
    ]

    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [scored[0]]

    while len(selected) < count:
        best_item = None
        best_value = -1.0
        for cand in scored:
            if cand in selected:
                continue
            min_dist = min(1.0 - cosine_similarity(cand[2], s[2]) for s in selected)
            value = 0.6 * min_dist + 0.4 * cand[1]
            if value > best_value:
                best_value = value
                best_item = cand
        if best_item is None:
            break
        selected.append(best_item)

        out = [item[0] for item in selected]
    return out[:count]


def _naive_followup(task_spec: TaskSpec, previous_output: list[str], turn_index: int) -> str:
    task = task_spec.task_id
    if task == "cdat":
        ask = "Please improve the previous list. Keep words related to the cue and increase diversity."
    elif task == "aut":
        ask = "Please improve the previous list. Keep ideas practical for the given object/context and make them more original."
    else:
        ask = "Please improve the previous list while keeping the requested format."
    return (
        task_spec.render_prompt(task_spec.metadata)
        + "\n\nPrevious answer JSON:\n"
        + _json_list_text(previous_output)
        + "\n\n"
        + ask
        + "\nReturn ONLY a JSON list of 10 strings."
    )


def _dialogue_followup(task_spec: TaskSpec, previous_output: list[str], turn_index: int) -> str:
    task = task_spec.task_id
    idx = turn_index % 3
    if task == "cdat":
        prompts = [
            "Make items less stereotyped while staying clearly related to the cue.",
            "Remove weak items and replace with stronger cue-linked alternatives.",
            "Return a clean final list with maximum spread and no near-duplicates.",
        ]
    elif task == "aut":
        prompts = [
            "Make ideas more unusual but still feasible in context.",
            "Cut generic ideas and replace with concrete, context-specific ones.",
            "Return a clean final list balancing creativity and usefulness.",
        ]
    else:
        prompts = [
            "Increase variety.",
            "Remove duplicates and weak items.",
            "Return a clean final list.",
        ]
    ask = prompts[idx]
    return (
        task_spec.render_prompt(task_spec.metadata)
        + "\n\nPrevious answer JSON:\n"
        + _json_list_text(previous_output)
        + "\n\nUser follow-up: "
        + ask
        + "\nReturn ONLY a JSON list of 10 strings."
    )


class NaiveMultiTurnRunner(MethodRunner):
    method_name = "naive_multiturn"

    def __init__(self, turns: int = 4) -> None:
        self.turns = max(2, int(turns))

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        base_prompt = task_spec.render_prompt(task_spec.metadata)
        current_prompt = base_prompt

        trace: list[dict[str, Any]] = []
        candidate_objectives: list[float] = []
        selected_gen: GeneratedList | None = None
        selected_output: list[str] = []
        generation_calls = 0
        tokens_in = 0
        tokens_out = 0

        for i in range(self.turns):
            cfg = replace(config, seed=config.seed + i)
            generated = _generate_list(model, current_prompt, cfg, task_spec.task_id)
            generation_calls += generated.attempts_made
            tokens_in += generated.prompt_tokens
            tokens_out += generated.output_tokens
            objective, metrics = _candidate_objective(task_spec, generated.output, mode="selection")
            candidate_objectives.append(objective)

            trace.append(
                {
                    "step": f"turn_{i + 1}",
                    "seed": cfg.seed,
                    "prompt": current_prompt,
                    "raw_model_text": generated.raw,
                    "objective": objective,
                    "metrics": metrics,
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                }
            )

            selected_gen = generated
            selected_output = generated.output
            if i < self.turns - 1:
                current_prompt = _naive_followup(task_spec, selected_output, i + 1)

        assert selected_gen is not None
        trace.append({"step": "selection", "strategy": "last_turn", "turns": self.turns})
        return MethodResult(
            prompt=base_prompt,
            output=selected_output,
            raw_trace=trace,
            parse_mode=selected_gen.parse_mode,
            json_valid=selected_gen.json_valid,
            retry_count=selected_gen.retry_count,
            candidate_objectives=candidate_objectives,
            generation_calls=generation_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


class DialogueMultiTurnRunner(MethodRunner):
    method_name = "dialogue_multiturn"

    def __init__(self, turns: int = 4) -> None:
        self.turns = max(2, int(turns))

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        base_prompt = task_spec.render_prompt(task_spec.metadata)
        current_prompt = base_prompt

        trace: list[dict[str, Any]] = []
        candidate_objectives: list[float] = []
        selected_gen: GeneratedList | None = None
        selected_output: list[str] = []
        generation_calls = 0
        tokens_in = 0
        tokens_out = 0

        for i in range(self.turns):
            cfg = replace(config, seed=config.seed + i)
            generated = _generate_list(model, current_prompt, cfg, task_spec.task_id)
            generation_calls += generated.attempts_made
            tokens_in += generated.prompt_tokens
            tokens_out += generated.output_tokens
            objective, metrics = _candidate_objective(task_spec, generated.output, mode="selection")
            candidate_objectives.append(objective)

            trace.append(
                {
                    "step": f"turn_{i + 1}",
                    "seed": cfg.seed,
                    "prompt": current_prompt,
                    "raw_model_text": generated.raw,
                    "objective": objective,
                    "metrics": metrics,
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                }
            )

            selected_gen = generated
            selected_output = generated.output
            if i < self.turns - 1:
                current_prompt = _dialogue_followup(task_spec, selected_output, i + 1)

        assert selected_gen is not None
        trace.append({"step": "selection", "strategy": "last_turn", "turns": self.turns})
        return MethodResult(
            prompt=base_prompt,
            output=selected_output,
            raw_trace=trace,
            parse_mode=selected_gen.parse_mode,
            json_valid=selected_gen.json_valid,
            retry_count=selected_gen.retry_count,
            candidate_objectives=candidate_objectives,
            generation_calls=generation_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


class RestlessnessRunner(MethodRunner):
    method_name = "restlessness_best"

    def __init__(self, iterations: int = 3, return_strategy: str = "best") -> None:
        self.iterations = max(1, iterations)
        if return_strategy not in {"best", "last"}:
            raise ValueError("return_strategy must be one of {'best', 'last'}")
        self.return_strategy = return_strategy

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        base_prompt = task_spec.render_prompt(task_spec.metadata)
        initial_gen = _generate_list(model, base_prompt, config, task_spec.task_id)
        best_output = initial_gen.output
        best_obj, best_metrics = _candidate_objective(task_spec, best_output, mode="selection")
        selected_gen = initial_gen
        selected_step = "initial"

        trace: list[dict[str, Any]] = [
            {
                "step": "initial",
                "prompt": base_prompt,
                "raw_model_text": initial_gen.raw,
                "objective": best_obj,
                "metrics": best_metrics,
                "constraints": [],
                "parse_mode": initial_gen.parse_mode,
                "json_valid": initial_gen.json_valid,
                "retry_count": initial_gen.retry_count,
            }
        ]

        current_output = best_output
        current_gen = initial_gen
        extra_constraints: list[str] = []
        candidate_objectives: list[float] = [best_obj]
        generation_calls = initial_gen.attempts_made
        tokens_in = initial_gen.prompt_tokens
        tokens_out = initial_gen.output_tokens

        for i in range(self.iterations):
            critique_constraints = _build_restless_constraints(task_spec, current_output, i)
            extra_constraints.extend(critique_constraints)
            loop_prompt = task_spec.render_prompt(task_spec.metadata, extra_constraints=extra_constraints)
            cfg = replace(config, seed=config.seed + i + 1)
            generated = _generate_list(model, loop_prompt, cfg, task_spec.task_id)
            generation_calls += generated.attempts_made
            tokens_in += generated.prompt_tokens
            tokens_out += generated.output_tokens
            objective, metrics = _candidate_objective(task_spec, generated.output, mode="selection")
            candidate_objectives.append(objective)

            trace.append(
                {
                    "step": f"iteration_{i + 1}",
                    "seed": cfg.seed,
                    "prompt": loop_prompt,
                    "constraints": critique_constraints,
                    "raw_model_text": generated.raw,
                    "objective": objective,
                    "metrics": metrics,
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                }
            )

            current_output = generated.output
            current_gen = generated
            if objective > best_obj:
                best_obj = objective
                best_output = generated.output
                selected_gen = generated
                selected_step = f"iteration_{i + 1}"

        if self.return_strategy == "last":
            chosen_output = current_output
            chosen_gen = current_gen
            chosen_step = f"iteration_{self.iterations}"
        else:
            chosen_output = best_output
            chosen_gen = selected_gen
            chosen_step = selected_step

        trace.append({"step": "selection", "strategy": self.return_strategy, "selected_step": chosen_step})

        method_name = "restlessness_best" if self.return_strategy == "best" else "restlessness_last_iter"
        self.method_name = method_name
        return MethodResult(
            prompt=base_prompt,
            output=chosen_output,
            raw_trace=trace,
            parse_mode=chosen_gen.parse_mode,
            json_valid=chosen_gen.json_valid,
            retry_count=chosen_gen.retry_count,
            candidate_objectives=candidate_objectives,
            generation_calls=generation_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


class RestlessnessAdaptiveRunner(MethodRunner):
    method_name = "restlessness_adaptive"

    def __init__(
        self,
        iterations: int = 3,
        adaptive_stop_delta: float | None = None,
        adaptive_min_iters: int = 1,
    ) -> None:
        self.iterations = max(1, iterations)
        self.default_stop_delta = 0.0 if adaptive_stop_delta is None else max(0.0, float(adaptive_stop_delta))
        self.default_min_iters = max(1, int(adaptive_min_iters))

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        base_prompt = task_spec.render_prompt(task_spec.metadata)
        initial_gen = _generate_list(model, base_prompt, config, task_spec.task_id)
        best_output = initial_gen.output
        best_obj, best_metrics = _candidate_objective(task_spec, best_output, mode="selection")
        best_gen = initial_gen
        best_step = "initial"

        trace: list[dict[str, Any]] = [
            {
                "step": "initial",
                "prompt": base_prompt,
                "raw_model_text": initial_gen.raw,
                "objective": best_obj,
                "metrics": best_metrics,
                "constraints": [],
                "parse_mode": initial_gen.parse_mode,
                "json_valid": initial_gen.json_valid,
                "retry_count": initial_gen.retry_count,
            }
        ]

        current_output = best_output
        extra_constraints: list[str] = []
        candidate_objectives: list[float] = [best_obj]
        generation_calls = initial_gen.attempts_made
        tokens_in = initial_gen.prompt_tokens
        tokens_out = initial_gen.output_tokens

        stop_delta = self.default_stop_delta
        if config.adaptive_stop_delta is not None:
            stop_delta = max(0.0, float(config.adaptive_stop_delta))
        min_iters = max(self.default_min_iters, int(getattr(config, "adaptive_min_iters", 1) or 1))
        used_iters = 0
        stopped_early = False

        for i in range(self.iterations):
            used_iters = i + 1
            critique_constraints = _build_restless_constraints(task_spec, current_output, i)
            extra_constraints.extend(critique_constraints)
            loop_prompt = task_spec.render_prompt(task_spec.metadata, extra_constraints=extra_constraints)
            cfg = replace(config, seed=config.seed + i + 1)
            generated = _generate_list(model, loop_prompt, cfg, task_spec.task_id)
            generation_calls += generated.attempts_made
            tokens_in += generated.prompt_tokens
            tokens_out += generated.output_tokens

            objective, metrics = _candidate_objective(task_spec, generated.output, mode="selection")
            candidate_objectives.append(objective)
            prev_best = best_obj
            if objective > best_obj:
                best_obj = objective
                best_output = generated.output
                best_gen = generated
                best_step = f"iteration_{i + 1}"
            improvement = best_obj - prev_best

            trace.append(
                {
                    "step": f"iteration_{i + 1}",
                    "seed": cfg.seed,
                    "prompt": loop_prompt,
                    "constraints": critique_constraints,
                    "raw_model_text": generated.raw,
                    "objective": objective,
                    "metrics": metrics,
                    "improvement_over_best": improvement,
                    "stop_delta": stop_delta,
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                }
            )
            current_output = generated.output
            if stop_delta > 0 and used_iters >= min_iters and improvement < stop_delta:
                stopped_early = True
                trace.append(
                    {
                        "step": "adaptive_stop",
                        "reason": "improvement_below_delta",
                        "iteration": used_iters,
                        "improvement_over_best": improvement,
                        "stop_delta": stop_delta,
                    }
                )
                break

        trace.append(
            {
                "step": "selection",
                "strategy": "best",
                "selected_step": best_step,
                "used_iterations": used_iters,
                "configured_iterations": self.iterations,
                "stopped_early": stopped_early,
            }
        )

        return MethodResult(
            prompt=base_prompt,
            output=best_output,
            raw_trace=trace,
            parse_mode=best_gen.parse_mode,
            json_valid=best_gen.json_valid,
            retry_count=best_gen.retry_count,
            candidate_objectives=candidate_objectives,
            generation_calls=generation_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


class RestlessnessTriggeredRunner(MethodRunner):
    method_name = "restlessness_triggered"

    def __init__(self, iterations: int = 3, default_trigger: float = 0.64, stop_delta: float = 0.01) -> None:
        self.iterations = max(1, int(iterations))
        self.default_trigger = max(0.0, min(1.0, float(default_trigger)))
        self.stop_delta = max(0.0, float(stop_delta))

    def run(self, task_spec: TaskSpec, config: GenerationConfig, model: ModelAdapter) -> MethodResult:
        base_prompt = task_spec.render_prompt(task_spec.metadata)
        initial_gen = _generate_list(model, base_prompt, config, task_spec.task_id)
        best_output = initial_gen.output
        best_obj, best_metrics = _candidate_objective(task_spec, best_output, mode="selection")
        best_gen = initial_gen
        best_step = "initial"

        trigger_threshold = self.default_trigger
        if config.trigger_objective is not None:
            trigger_threshold = max(0.0, min(1.0, float(config.trigger_objective)))

        trace: list[dict[str, Any]] = [
            {
                "step": "initial",
                "prompt": base_prompt,
                "raw_model_text": initial_gen.raw,
                "objective": best_obj,
                "metrics": best_metrics,
                "parse_mode": initial_gen.parse_mode,
                "json_valid": initial_gen.json_valid,
                "retry_count": initial_gen.retry_count,
                "trigger_threshold": trigger_threshold,
            }
        ]

        candidate_objectives: list[float] = [best_obj]
        generation_calls = initial_gen.attempts_made
        tokens_in = initial_gen.prompt_tokens
        tokens_out = initial_gen.output_tokens

        if best_obj >= trigger_threshold:
            trace.append(
                {
                    "step": "trigger_skip",
                    "reason": "initial_objective_above_threshold",
                    "initial_objective": best_obj,
                    "trigger_threshold": trigger_threshold,
                }
            )
            trace.append({"step": "selection", "strategy": "initial"})
            return MethodResult(
                prompt=base_prompt,
                output=best_output,
                raw_trace=trace,
                parse_mode=best_gen.parse_mode,
                json_valid=best_gen.json_valid,
                retry_count=best_gen.retry_count,
                candidate_objectives=candidate_objectives,
                generation_calls=generation_calls,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

        extra_constraints: list[str] = []
        current_output = best_output
        used_iters = 0
        stopped_early = False
        stop_delta = self.stop_delta
        if config.adaptive_stop_delta is not None:
            stop_delta = max(0.0, float(config.adaptive_stop_delta))

        for i in range(self.iterations):
            used_iters = i + 1
            critique_constraints = _build_restless_constraints(task_spec, current_output, i)
            extra_constraints.extend(critique_constraints)
            loop_prompt = task_spec.render_prompt(task_spec.metadata, extra_constraints=extra_constraints)
            cfg = replace(config, seed=config.seed + i + 1)
            generated = _generate_list(model, loop_prompt, cfg, task_spec.task_id)
            generation_calls += generated.attempts_made
            tokens_in += generated.prompt_tokens
            tokens_out += generated.output_tokens
            objective, metrics = _candidate_objective(task_spec, generated.output, mode="selection")
            candidate_objectives.append(objective)

            prev_best = best_obj
            if objective > best_obj:
                best_obj = objective
                best_output = generated.output
                best_gen = generated
                best_step = f"iteration_{i + 1}"
            improvement = best_obj - prev_best

            trace.append(
                {
                    "step": f"iteration_{i + 1}",
                    "seed": cfg.seed,
                    "prompt": loop_prompt,
                    "constraints": critique_constraints,
                    "raw_model_text": generated.raw,
                    "objective": objective,
                    "metrics": metrics,
                    "improvement_over_best": improvement,
                    "stop_delta": stop_delta,
                    "parse_mode": generated.parse_mode,
                    "json_valid": generated.json_valid,
                    "retry_count": generated.retry_count,
                }
            )
            current_output = generated.output

            if stop_delta > 0 and improvement < stop_delta:
                stopped_early = True
                trace.append(
                    {
                        "step": "adaptive_stop",
                        "reason": "improvement_below_delta",
                        "iteration": used_iters,
                        "improvement_over_best": improvement,
                        "stop_delta": stop_delta,
                    }
                )
                break

        trace.append(
            {
                "step": "selection",
                "strategy": "best",
                "selected_step": best_step,
                "used_iterations": used_iters,
                "configured_iterations": self.iterations,
                "stopped_early": stopped_early,
            }
        )

        return MethodResult(
            prompt=base_prompt,
            output=best_output,
            raw_trace=trace,
            parse_mode=best_gen.parse_mode,
            json_valid=best_gen.json_valid,
            retry_count=best_gen.retry_count,
            candidate_objectives=candidate_objectives,
            generation_calls=generation_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


def _build_restless_constraints(task_spec: TaskSpec, output: list[str], iteration: int) -> list[str]:
    token_stream = " ".join(output).lower()
    most_common_stems = _top_stems(output, k=2)

    constraints: list[str] = [
        f"Iteration {iteration + 1}: avoid repeating the previously dominant semantic pattern.",
        "Force at least 4 items from clearly different semantic regions than prior attempt.",
    ]

    if most_common_stems:
        constraints.append("Avoid tokens that begin with: " + ", ".join(most_common_stems) + ".")

    if task_spec.task_id == "cdat":
        cue = str(task_spec.metadata.get("cue", ""))
        constraints.append(
            f"Keep every item explicitly related to cue \"{cue}\" while exploring less stereotyped associations."
        )
    elif task_spec.task_id == "aut":
        obj = str(task_spec.metadata.get("object", ""))
        ctx = str(task_spec.metadata.get("context", ""))
        constraints.append(
            f"Maintain feasibility for object \"{obj}\" and context \"{ctx}\"; reject impractical ideas."
        )
    else:
        constraints.append("Keep output as valid single-word nouns.")

    constraints.append(f"Novelty pressure level: {iteration + 1}.")
    if "fallback" in token_stream:
        constraints.append("Replace placeholder or generic terms with concrete alternatives.")

    return constraints


def _top_stems(items: list[str], k: int = 2) -> list[str]:
    stems = []
    for item in items:
        token = re.sub(r"[^a-z]", "", item.lower())
        if len(token) >= 4:
            stems.append(token[:4])
    counts: dict[str, int] = {}
    for stem in stems:
        counts[stem] = counts.get(stem, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [stem for stem, _ in itertools.islice(ranked, k)]


def build_method_runner(
    method: str,
    restlessness_k: int = 3,
    best_of_k: int = 4,
    adaptive_stop_delta: float | None = None,
    adaptive_min_iters: int = 1,
    trigger_objective: float | None = None,
) -> MethodRunner:
    name = method.strip().lower()
    if name == "one_shot":
        return OneShotRunner()
    if name == "best_of_k_one_shot":
        return BestOfKOneShotRunner(k=best_of_k)
    if name == "naive_multiturn":
        return NaiveMultiTurnRunner(turns=best_of_k)
    if name == "dialogue_multiturn":
        return DialogueMultiTurnRunner(turns=best_of_k)
    if name == "brainstorm_then_select":
        return BrainstormSelectRunner()
    if name in {"restlessness_loop", "restlessness_best"}:
        return RestlessnessRunner(iterations=restlessness_k, return_strategy="best")
    if name == "restlessness_triggered":
        return RestlessnessTriggeredRunner(
            iterations=restlessness_k,
            default_trigger=trigger_objective if trigger_objective is not None and float(trigger_objective) > 0 else 0.64,
        )
    if name == "restlessness_last_iter":
        return RestlessnessRunner(iterations=restlessness_k, return_strategy="last")
    if name == "restlessness_adaptive":
        return RestlessnessAdaptiveRunner(
            iterations=restlessness_k,
            adaptive_stop_delta=adaptive_stop_delta,
            adaptive_min_iters=adaptive_min_iters,
        )
    raise ValueError(f"Unsupported method: {method}")
