from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any

from creativeai.embeddings import (
    active_embedding_backend,
    compactness_similarity,
    cosine_similarity,
    mean_pairwise_cosine_distance,
    nearest_neighbor_similarity,
    text_embedding,
)
from creativeai.schemas import ScoreRecord

_COMMON_IDEAS = [
    "use as a paperweight",
    "use as a doorstop",
    "use as decoration",
    "use as a container",
    "use as a pointer",
    "use as a holder",
    "use for training",
    "use as a support",
    "use as a marker",
    "use for storage",
]

_HARMFUL_TERMS = {
    "weapon",
    "attack",
    "harm",
    "poison",
    "explode",
    "illegal",
}

_ACTION_TERMS = {
    "build",
    "filter",
    "signal",
    "stabilize",
    "measure",
    "reflect",
    "organize",
    "secure",
    "rescue",
    "teach",
    "repair",
}



def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))



def _normalized_similarity(a: str, b: str) -> float:
    sim = cosine_similarity(text_embedding(a), text_embedding(b))
    return clamp01((sim + 1.0) / 2.0)



def score_dat(words: list[str]) -> dict[str, float]:
    novelty = mean_pairwise_cosine_distance(words)
    return {
        "novelty": novelty,
        "appropriateness": 1.0,
        "usefulness": 0.0,
    }



def lexical_overlap_ratio(words: list[str], cue: str) -> float:
    cue_alpha = re.sub(r"[^a-z]", "", cue.lower())
    if not cue_alpha or not words:
        return 0.0
    hits = 0
    for word in words:
        w = re.sub(r"[^a-z]", "", str(word).lower())
        if not w:
            continue
        if cue_alpha in w or w in cue_alpha:
            hits += 1
    return hits / max(len(words), 1)



def score_cdat(
    words: list[str],
    cue: str,
    relevance_threshold: float = 0.12,
    overlap_penalty: bool = True,
) -> dict[str, float]:
    novelty = mean_pairwise_cosine_distance(words)
    sims_raw = [cosine_similarity(text_embedding(w), text_embedding(cue)) for w in words]
    if not sims_raw:
        return {"novelty": 0.0, "appropriateness": 0.0, "usefulness": 0.0}
    pos_rel = [max(0.0, s) for s in sims_raw]
    avg_rel = sum(pos_rel) / len(pos_rel)
    rel_ratio = sum(1 for s in sims_raw if s >= relevance_threshold) / len(sims_raw)
    lexical_overlap = lexical_overlap_ratio(words, cue)

    base_appropriateness = clamp01(0.65 * avg_rel + 0.35 * rel_ratio)
    anti_cheat_penalty = 0.0
    if overlap_penalty:
        anti_cheat_penalty = 0.35 * clamp01((lexical_overlap - 0.45) / 0.55)
    appropriateness = clamp01(base_appropriateness - anti_cheat_penalty)
    return {
        "novelty": novelty,
        "appropriateness": appropriateness,
        "usefulness": 0.0,
        "avg_relevance": avg_rel,
        "relevant_ratio": rel_ratio,
        "lexical_overlap_ratio": lexical_overlap,
        "base_appropriateness": base_appropriateness,
        "anti_cheat_penalty": anti_cheat_penalty,
    }



def idea_usefulness_proxy(idea: str, object_name: str, context: str) -> float:
    text = idea.lower()
    score = 0.45

    obj_tokens = set(re.findall(r"[a-zA-Z]+", object_name.lower()))
    if obj_tokens and any(token in text for token in obj_tokens):
        score += 0.12

    context_tokens = set(re.findall(r"[a-zA-Z]+", context.lower()))
    if context_tokens and sum(token in text for token in context_tokens) >= 1:
        score += 0.2

    if any(term in text for term in _ACTION_TERMS):
        score += 0.12

    if any(term in text for term in _HARMFUL_TERMS):
        score -= 0.65

    if len(text.split()) < 5:
        score -= 0.12

    return clamp01(score)



def score_aut(ideas: list[str], object_name: str, context: str) -> dict[str, float]:
    pairwise_div = mean_pairwise_cosine_distance(ideas)

    common_embeds = [text_embedding(item) for item in _COMMON_IDEAS]
    common_dist = []
    for idea in ideas:
        emb = text_embedding(idea)
        max_sim = max(cosine_similarity(emb, ref) for ref in common_embeds)
        common_dist.append(1.0 - max_sim)
    novelty_vs_common = sum(common_dist) / len(common_dist) if common_dist else 0.0
    novelty = 0.65 * pairwise_div + 0.35 * novelty_vs_common

    idea_scores = [idea_usefulness_proxy(i, object_name=object_name, context=context) for i in ideas]
    usefulness = sum(idea_scores) / len(idea_scores) if idea_scores else 0.0

    context_fit = sum(
        _normalized_similarity(i, f"{object_name} {context}") for i in ideas
    ) / max(len(ideas), 1)
    appropriateness = clamp01(0.55 * usefulness + 0.45 * context_fit)

    return {
        "novelty": novelty,
        "appropriateness": appropriateness,
        "usefulness": usefulness,
        "pairwise_diversity": pairwise_div,
        "novelty_vs_common": novelty_vs_common,
        "context_fit": context_fit,
    }



def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]



def _bleu_precision(candidate: str, references: list[str], n: int = 2) -> float:
    cand_tokens = re.findall(r"[a-zA-Z0-9]+", candidate.lower())
    cgrams = _ngrams(cand_tokens, n)
    if not cgrams:
        return 0.0
    cand_count = Counter(cgrams)
    max_ref_counts: Counter[tuple[str, ...]] = Counter()
    for ref in references:
        ref_tokens = re.findall(r"[a-zA-Z0-9]+", ref.lower())
        ref_counts = Counter(_ngrams(ref_tokens, n))
        for gram, cnt in ref_counts.items():
            if cnt > max_ref_counts[gram]:
                max_ref_counts[gram] = cnt
    clipped = 0
    total = 0
    for gram, cnt in cand_count.items():
        clipped += min(cnt, max_ref_counts.get(gram, 0))
        total += cnt
    if total == 0:
        return 0.0
    return clipped / total



def self_bleu(texts: list[str], n: int = 2) -> float:
    if len(texts) < 2:
        return 0.0
    vals: list[float] = []
    for i, cand in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        vals.append(_bleu_precision(cand, refs, n=n))
    return sum(vals) / len(vals)



def homogeneity_metrics(texts: list[str]) -> dict[str, float]:
    if not texts:
        return {
            "compactness": 0.0,
            "nearest_neighbor_similarity": 0.0,
            "self_bleu": 0.0,
            "diversity_index": 0.0,
        }

    compactness = compactness_similarity(texts)
    nn_sim = nearest_neighbor_similarity(texts)
    sbleu = self_bleu(texts, n=2)
    diversity = clamp01(1.0 - (0.45 * nn_sim + 0.35 * sbleu + 0.2 * compactness))
    return {
        "compactness": compactness,
        "nearest_neighbor_similarity": nn_sim,
        "self_bleu": sbleu,
        "diversity_index": diversity,
    }



def flatten_output(output: list[str]) -> str:
    return " || ".join(item.strip() for item in output if item.strip())



def selection_objective(task_id: str, metrics: dict[str, float]) -> float:
    novelty = float(metrics.get("novelty", 0.0))
    app = float(metrics.get("appropriateness", 0.0))
    use = float(metrics.get("usefulness", 0.0))
    task = task_id.lower()

    if task in {"dat", "cdat"}:
        return clamp01(0.55 * novelty + 0.45 * app)
    if task == "aut":
        return clamp01(0.45 * novelty + 0.35 * app + 0.2 * use)
    return clamp01(0.5 * novelty + 0.5 * app)



def compute_score_record(run: dict[str, Any]) -> ScoreRecord:
    task_id = str(run.get("task_id", "")).lower()
    output = [str(x) for x in run.get("output", [])]
    metadata = dict(run.get("metadata", {}))
    session_id = str(run.get("session_id", metadata.get("session_id", ""))).strip()
    if session_id:
        metadata["session_id"] = session_id

    if task_id == "dat":
        metrics = score_dat(output)
    elif task_id == "cdat":
        cue = str(metadata.get("cue", ""))
        metrics = score_cdat(output, cue=cue)
    elif task_id == "aut":
        obj = str(metadata.get("object", ""))
        context = str(metadata.get("context", ""))
        metrics = score_aut(output, object_name=obj, context=context)
    else:
        metrics = {"novelty": 0.0, "appropriateness": 0.0, "usefulness": 0.0}

    novelty = float(metrics.get("novelty", 0.0))
    appropriateness = float(metrics.get("appropriateness", 0.0))
    usefulness = float(metrics.get("usefulness", 0.0))
    lexical_overlap = float(metrics.get("lexical_overlap_ratio", 0.0))
    objective = frontier_objective(novelty, appropriateness)

    validity_flags = dict(run.get("validity_flags", {}))
    validity_flags["high_novelty_low_appropriateness"] = novelty > 0.55 and appropriateness < 0.35
    json_valid = bool(run.get("json_valid", validity_flags.get("json_valid", True)))
    validity_flags["json_valid"] = json_valid
    if not json_valid:
        validity_flags["valid"] = False
        problems = list(validity_flags.get("problems", []))
        if "non_json_output" not in problems:
            problems.append("non_json_output")
        validity_flags["problems"] = sorted(set(problems))
    valid_for_primary = bool(validity_flags.get("valid", False)) and json_valid

    metadata.setdefault("generation_calls", int(run.get("effective_calls", metadata.get("generation_calls", 1) or 1)))
    metadata.setdefault("effective_calls", int(run.get("effective_calls", metadata.get("generation_calls", 1) or 1)))
    metadata.setdefault("tokens_in", int(run.get("tokens_in", metadata.get("tokens_in", 0) or 0)))
    metadata.setdefault("tokens_out", int(run.get("tokens_out", metadata.get("tokens_out", 0) or 0)))
    metadata.setdefault("tokens_total", int(run.get("tokens_total", metadata.get("tokens_total", 0) or 0)))
    metadata.setdefault("compute_group_id", str(run.get("compute_group_id", metadata.get("compute_group_id", ""))))
    metadata.setdefault("phase3_stage", str(run.get("phase3_stage", metadata.get("phase3_stage", ""))))

    tokens_total = int(metadata.get("tokens_total", 0) or 0)
    if tokens_total <= 0:
        fallback_tokens = int(run.get("token_count", 0) or metadata.get("token_count", 0) or 0)
        tokens_total = fallback_tokens
        metadata["tokens_total"] = tokens_total
    score_per_1k_tokens = (1000.0 * objective / tokens_total) if tokens_total > 0 else 0.0

    method = str(run.get("method", "unknown-method"))
    generation_calls = int(metadata.get("generation_calls", 1) or 1)
    compute_matched_valid = valid_for_primary and tokens_total > 0
    if method in {
        "best_of_k_one_shot",
        "naive_multiturn",
        "dialogue_multiturn",
        "restlessness_best",
        "restlessness_triggered",
        "restlessness_adaptive",
        "restlessness_last_iter",
    }:
        compute_matched_valid = compute_matched_valid and generation_calls >= 2

    return ScoreRecord(
        run_id=str(run.get("run_id", "unknown-run")),
        task_id=task_id,
        method=method,
        model_id=str(run.get("model_id", "unknown-model")),
        novelty=novelty,
        appropriateness=appropriateness,
        usefulness=usefulness,
        validity_flags=validity_flags,
        metrics=metrics,
        valid_for_primary=valid_for_primary,
        metric_backend=active_embedding_backend(),
        lexical_overlap_ratio=lexical_overlap,
        session_id=session_id,
        metadata=metadata,
        score_per_1k_tokens=score_per_1k_tokens,
        compute_matched_valid=compute_matched_valid,
    )



def bootstrap_mean_ci(
    values: list[float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 7,
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(max(50, n_boot)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(values) / n
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, int(alpha * len(means)) - 1)
    high_idx = min(len(means) - 1, int((1.0 - alpha) * len(means)) - 1)
    return means[low_idx], mean, means[high_idx]



def frontier_objective(novelty: float, appropriateness: float) -> float:
    return math.sqrt(max(novelty, 0.0) * max(appropriateness, 0.0))
