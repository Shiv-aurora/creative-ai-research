from __future__ import annotations

import random
from collections import defaultdict
from typing import Any



def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks



def _pearson(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    if da == 0 or db == 0:
        return 0.0
    return num / ((da * db) ** 0.5)



def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_rank(a), _rank(b))



def stratified_human_slice(scores: list[dict[str, Any]], n: int = 180, seed: int = 19) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        key = (str(row.get("task_id", "")), str(row.get("method", "")), str(row.get("model_id", "")))
        grouped[key].append(row)

    rng = random.Random(seed)
    buckets = list(grouped.values())
    if not buckets:
        return []

    quota = max(1, n // len(buckets))
    selected: list[dict[str, Any]] = []
    for bucket in buckets:
        rng.shuffle(bucket)
        selected.extend(bucket[:quota])

    if len(selected) < n:
        leftovers = [row for bucket in buckets for row in bucket if row not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: n - len(selected)])

    selected = selected[:n]
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        out.append(
            {
                "rating_id": f"R{idx:04d}",
                "run_id": row.get("run_id"),
                "task_id": row.get("task_id"),
                "model_id": row.get("model_id"),
                "method": row.get("method"),
                "output": row.get("metadata", {}).get("output_preview", ""),
                "auto_appropriateness": row.get("appropriateness", 0.0),
                "auto_usefulness": row.get("usefulness", 0.0),
                "human1_appropriateness": None,
                "human2_appropriateness": None,
                "human1_usefulness": None,
                "human2_usefulness": None,
            }
        )
    return out



def evaluate_human_calibration(ratings: list[dict[str, Any]], target_corr: float = 0.45) -> dict[str, Any]:
    auto_app: list[float] = []
    auto_use: list[float] = []
    hum_app: list[float] = []
    hum_use: list[float] = []
    r1_app: list[float] = []
    r2_app: list[float] = []

    for row in ratings:
        h1a = row.get("human1_appropriateness")
        h2a = row.get("human2_appropriateness")
        h1u = row.get("human1_usefulness")
        h2u = row.get("human2_usefulness")
        if None in {h1a, h2a, h1u, h2u}:
            continue

        avg_app = (float(h1a) + float(h2a)) / 2.0
        avg_use = (float(h1u) + float(h2u)) / 2.0

        auto_app.append(float(row.get("auto_appropriateness", 0.0)))
        auto_use.append(float(row.get("auto_usefulness", 0.0)))
        hum_app.append(avg_app)
        hum_use.append(avg_use)
        r1_app.append(float(h1a))
        r2_app.append(float(h2a))

    app_corr = spearman(auto_app, hum_app) if auto_app else 0.0
    use_corr = spearman(auto_use, hum_use) if auto_use else 0.0
    inter_rater_app = spearman(r1_app, r2_app) if r1_app else 0.0

    gate_pass = app_corr >= target_corr and use_corr >= target_corr
    return {
        "n_scored": len(auto_app),
        "spearman_auto_vs_human_appropriateness": app_corr,
        "spearman_auto_vs_human_usefulness": use_corr,
        "inter_rater_appropriateness": inter_rater_app,
        "target_corr": target_corr,
        "gate_pass": gate_pass,
    }
