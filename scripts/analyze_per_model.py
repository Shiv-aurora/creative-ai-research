#!/usr/bin/env python3
"""Per-model sampler profile breakdown.

Answers: does anti_repetition's gain vary by model family (Gemma vs Qwen vs
Mistral) and by model type (base vs instruct)? Reads scores.jsonl, computes
paired deltas vs default_nucleus for each model separately.

Usage:
    python scripts/analyze_per_model.py \
        --scores outputs/phase5_cuda_confirm/scores/scores.jsonl \
        --baseline default_nucleus \
        --output outputs/phase5_cuda_confirm/analysis/per_model_sampler.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    mean = sum(values) / n
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return lo, mean, hi


def load_scores(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_profile(record: dict) -> str:
    return str(record.get("metadata", {}).get("sampler_profile", record.get("sampler_profile", "manual")))


def get_model_family(model_id: str) -> str:
    m = model_id.lower()
    if "gemma" in m:
        return "gemma"
    if "qwen" in m:
        return "qwen"
    if "mistral" in m:
        return "mistral"
    return "other"


def get_model_type(model_id: str) -> str:
    m = model_id.lower()
    if "instruct" in m or "it" in m.split("-"):
        return "instruct"
    return "base"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--baseline", default="default_nucleus")
    p.add_argument("--output", default="outputs/analysis/per_model_sampler.json")
    p.add_argument("--exclude-invalid", action="store_true", default=True)
    args = p.parse_args()

    records = load_scores(Path(args.scores))
    if args.exclude_invalid:
        records = [r for r in records if r.get("valid_for_primary", True)]

    # index: model_id → profile → list[objective]
    by_model_profile: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # for paired deltas: (model_id, prompt_key, seed) → profile → objective
    paired_index: dict[tuple, dict[str, float]] = defaultdict(dict)

    for r in records:
        model_id = str(r.get("model_id", ""))
        profile = get_profile(r)
        obj = float(r.get("objective", math.sqrt(
            max(float(r.get("novelty", 0)), 0) * max(float(r.get("appropriateness", 0)), 0)
        )))
        by_model_profile[model_id][profile].append(obj)

        # Build paired key: same prompt × same seed
        meta = r.get("metadata", {})
        task = str(r.get("task_id", ""))
        seed = meta.get("seed", r.get("seed", 0))
        cue = meta.get("cue") or meta.get("object") or meta.get("dat_prompt_id", 0)
        key = (model_id, task, cue, seed)
        paired_index[key][profile] = obj

    baseline = args.baseline
    results = []

    for model_id in sorted(by_model_profile):
        profiles = by_model_profile[model_id]
        if baseline not in profiles:
            continue

        base_obj = profiles[baseline]
        _, base_mean, _ = bootstrap_ci(base_obj)

        model_result: dict = {
            "model_id": model_id,
            "family": get_model_family(model_id),
            "type": get_model_type(model_id),
            "baseline_n": len(base_obj),
            "baseline_mean": round(base_mean, 4),
            "profiles": {},
        }

        for profile, obj_list in sorted(profiles.items()):
            if profile == baseline:
                continue
            lo, mean, hi = bootstrap_ci(obj_list)

            # Paired deltas
            deltas = []
            for key, pmap in paired_index.items():
                if key[0] != model_id:
                    continue
                if baseline in pmap and profile in pmap:
                    deltas.append(pmap[profile] - pmap[baseline])

            if deltas:
                d_lo, d_mean, d_hi = bootstrap_ci(deltas)
            else:
                d_lo = d_mean = d_hi = 0.0

            model_result["profiles"][profile] = {
                "n": len(obj_list),
                "mean": round(mean, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "paired_n": len(deltas),
                "paired_delta": round(d_mean, 4),
                "paired_ci_low": round(d_lo, 4),
                "paired_ci_high": round(d_hi, 4),
                "ci_above_zero": d_lo > 0,
            }

        results.append(model_result)

    # Summary: which profiles beat baseline for each family/type
    print("\n=== Per-Model Sampler Profile Breakdown ===\n")
    print(f"{'Model':<35} {'Profile':<25} {'Delta':>8}  {'95% CI':<22}  {'Above0'}")
    print("-" * 100)
    for r in results:
        for profile, stats in r["profiles"].items():
            ci_str = f"[{stats['paired_ci_low']:+.4f}, {stats['paired_ci_high']:+.4f}]"
            above = "✓" if stats["ci_above_zero"] else " "
            print(f"{r['model_id']:<35} {profile:<25} {stats['paired_delta']:>+8.4f}  {ci_str:<22}  {above}")
        print()

    # Family-level summary
    family_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        fam = r["family"]
        for profile, stats in r["profiles"].items():
            if stats["paired_n"] > 0:
                family_deltas[fam][profile].append(stats["paired_delta"])

    print("\n=== Family-Level Summary (mean delta across models in family) ===\n")
    print(f"{'Family':<12} {'Profile':<25} {'Mean Delta':>12}")
    print("-" * 55)
    for fam in sorted(family_deltas):
        for profile in sorted(family_deltas[fam]):
            vals = family_deltas[fam][profile]
            print(f"{fam:<12} {profile:<25} {sum(vals)/len(vals):>+12.4f}")

    # Type-level (base vs instruct)
    type_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        mtype = r["type"]
        for profile, stats in r["profiles"].items():
            if stats["paired_n"] > 0:
                type_deltas[mtype][profile].append(stats["paired_delta"])

    print("\n=== Base vs Instruct ===\n")
    print(f"{'Type':<12} {'Profile':<25} {'Mean Delta':>12}")
    print("-" * 55)
    for mtype in sorted(type_deltas):
        for profile in sorted(type_deltas[mtype]):
            vals = type_deltas[mtype][profile]
            print(f"{mtype:<12} {profile:<25} {sum(vals)/len(vals):>+12.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "baseline": baseline,
        "scores_path": str(args.scores),
        "per_model": results,
        "family_summary": {
            fam: {
                profile: round(sum(vals) / len(vals), 4)
                for profile, vals in profiles.items()
            }
            for fam, profiles in family_deltas.items()
        },
        "type_summary": {
            mtype: {
                profile: round(sum(vals) / len(vals), 4)
                for profile, vals in profiles.items()
            }
            for mtype, profiles in type_deltas.items()
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
