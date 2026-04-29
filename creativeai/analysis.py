from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from creativeai.scoring import bootstrap_mean_ci, flatten_output, frontier_objective, homogeneity_metrics
from creativeai.schemas import FrontierPoint



def _group_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("model_id", "unknown-model")),
        str(record.get("method", "unknown-method")),
        str(record.get("task_id", "unknown-task")),
    )


def _is_primary_valid(record: dict[str, Any]) -> bool:
    if "valid_for_primary" in record:
        return bool(record.get("valid_for_primary"))
    flags = record.get("validity_flags", {})
    if not isinstance(flags, dict):
        return True
    return bool(flags.get("valid", True)) and bool(flags.get("json_valid", True))


def _prompt_key(record: dict[str, Any]) -> tuple[Any, ...]:
    task_id = str(record.get("task_id", "")).lower()
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    if task_id == "cdat":
        return ("cdat", metadata.get("cue"))
    if task_id == "aut":
        return ("aut", metadata.get("object"), metadata.get("context"))
    if task_id == "dat":
        return ("dat", metadata.get("dat_prompt_id", 0))
    return (task_id, "default")


def _replicate_key(record: dict[str, Any]) -> tuple[Any, ...]:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    compute_group_id = str(
        metadata.get("compute_group_id", "") or record.get("compute_group_id", "")
    ).strip()
    if compute_group_id:
        return ("compute_group", compute_group_id)

    # Fallback for legacy records that do not include compute_group_id.
    seed = metadata.get("seed", record.get("seed"))
    temperature = metadata.get("temperature", record.get("temperature"))
    stage = metadata.get("phase3_stage", record.get("phase3_stage"))
    return ("legacy", seed, temperature, stage)



def frontier_points(
    records: list[dict[str, Any]],
    exclude_invalid: bool = False,
) -> tuple[list[FrontierPoint], list[dict[str, Any]]]:
    rows = [r for r in records if _is_primary_valid(r)] if exclude_invalid else records
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in rows:
        grouped[_group_key(rec)].append(rec)

    points: list[FrontierPoint] = []
    extended: list[dict[str, Any]] = []

    for (model_id, method, task), rows in grouped.items():
        nov = [float(r.get("novelty", 0.0)) for r in rows]
        app = [float(r.get("appropriateness", 0.0)) for r in rows]
        obj = [frontier_objective(n, a) for n, a in zip(nov, app)]

        obj_low, obj_mean, obj_high = bootstrap_mean_ci(obj)
        _, nov_mean, _ = bootstrap_mean_ci(nov)
        _, app_mean, _ = bootstrap_mean_ci(app)

        points.append(
            FrontierPoint(
                model_id=model_id,
                method=method,
                task_group=task,
                novelty_mean=nov_mean,
                appropriateness_mean=app_mean,
                ci_low=obj_low,
                ci_high=obj_high,
            )
        )

        extended.append(
            {
                "model_id": model_id,
                "method": method,
                "task_id": task,
                "sample_count": len(rows),
                "novelty_mean": nov_mean,
                "appropriateness_mean": app_mean,
                "objective_mean": obj_mean,
                "objective_ci_low": obj_low,
                "objective_ci_high": obj_high,
            }
        )

    return points, extended


def _sampler_profile(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    profile = str(metadata.get("sampler_profile", "")).strip()
    if profile:
        return profile
    settings = metadata.get("decoding_settings", {})
    if isinstance(settings, dict):
        profile = str(settings.get("sampler_profile", "")).strip()
        if profile:
            return profile
    return "manual"


def _sampler_pair_key(record: dict[str, Any]) -> tuple[Any, ...]:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return (
        str(record.get("model_id", "")),
        str(record.get("method", "")),
        str(record.get("task_id", "")),
        _prompt_key(record),
        metadata.get("seed"),
    )


def sampler_profile_analysis(
    records: list[dict[str, Any]],
    baseline_profile: str = "default_nucleus",
    exclude_invalid: bool = True,
) -> dict[str, Any]:
    rows = [r for r in records if _is_primary_valid(r)] if exclude_invalid else list(records)
    all_rows = list(records)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in rows:
        grouped[_sampler_profile(rec)].append(rec)
    for rec in all_rows:
        all_grouped[_sampler_profile(rec)].append(rec)

    summary: list[dict[str, Any]] = []
    for profile in sorted(all_grouped):
        profile_all = all_grouped[profile]
        profile_valid = grouped.get(profile, [])
        objectives = [
            frontier_objective(float(r.get("novelty", 0.0)), float(r.get("appropriateness", 0.0)))
            for r in profile_valid
        ]
        novelty = [float(r.get("novelty", 0.0)) for r in profile_valid]
        appropriateness = [float(r.get("appropriateness", 0.0)) for r in profile_valid]
        usefulness = [float(r.get("usefulness", 0.0)) for r in profile_valid]
        tokens = [
            float((r.get("metadata", {}) if isinstance(r.get("metadata", {}), dict) else {}).get("tokens_total", 0.0))
            for r in profile_valid
        ]
        retries = [
            float((r.get("metadata", {}) if isinstance(r.get("metadata", {}), dict) else {}).get("retry_count", 0.0))
            for r in profile_valid
        ]
        valid_n = len(profile_valid)
        total_n = len(profile_all)
        json_valid_n = 0
        for r in profile_all:
            flags = r.get("validity_flags", {}) if isinstance(r.get("validity_flags", {}), dict) else {}
            metadata = r.get("metadata", {}) if isinstance(r.get("metadata", {}), dict) else {}
            json_valid_n += 1 if bool(metadata.get("json_valid", flags.get("json_valid", False))) else 0
        low, mean, high = bootstrap_mean_ci(objectives)
        summary.append(
            {
                "sampler_profile": profile,
                "n_total": total_n,
                "n_primary_valid": valid_n,
                "primary_valid_rate": (valid_n / total_n) if total_n else 0.0,
                "json_valid_rate": (json_valid_n / total_n) if total_n else 0.0,
                "objective_mean": mean,
                "objective_ci_low": low,
                "objective_ci_high": high,
                "novelty_mean": (sum(novelty) / len(novelty)) if novelty else 0.0,
                "appropriateness_mean": (sum(appropriateness) / len(appropriateness)) if appropriateness else 0.0,
                "usefulness_mean": (sum(usefulness) / len(usefulness)) if usefulness else 0.0,
                "tokens_mean": (sum(tokens) / len(tokens)) if tokens else 0.0,
                "retry_count_mean": (sum(retries) / len(retries)) if retries else 0.0,
            }
        )

    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in rows:
        by_key[_sampler_pair_key(rec)][_sampler_profile(rec)] = rec

    paired_deltas: list[dict[str, Any]] = []
    profiles = sorted(grouped)
    for profile in profiles:
        if profile == baseline_profile:
            continue
        obj_deltas: list[float] = []
        novelty_deltas: list[float] = []
        app_deltas: list[float] = []
        for bucket in by_key.values():
            if profile not in bucket or baseline_profile not in bucket:
                continue
            a = bucket[profile]
            b = bucket[baseline_profile]
            obj_a = frontier_objective(float(a.get("novelty", 0.0)), float(a.get("appropriateness", 0.0)))
            obj_b = frontier_objective(float(b.get("novelty", 0.0)), float(b.get("appropriateness", 0.0)))
            obj_deltas.append(obj_a - obj_b)
            novelty_deltas.append(float(a.get("novelty", 0.0)) - float(b.get("novelty", 0.0)))
            app_deltas.append(float(a.get("appropriateness", 0.0)) - float(b.get("appropriateness", 0.0)))
        obj_low, obj_mean, obj_high = bootstrap_mean_ci(obj_deltas)
        _, nov_mean, _ = bootstrap_mean_ci(novelty_deltas)
        _, app_mean, _ = bootstrap_mean_ci(app_deltas)
        paired_deltas.append(
            {
                "sampler_profile": profile,
                "baseline_profile": baseline_profile,
                "n_pairs": len(obj_deltas),
                "objective_delta_mean": obj_mean,
                "objective_delta_ci_low": obj_low,
                "objective_delta_ci_high": obj_high,
                "novelty_delta_mean": nov_mean,
                "appropriateness_delta_mean": app_mean,
            }
        )

    pareto: list[dict[str, Any]] = []
    for row in summary:
        dominated = False
        for other in summary:
            if other is row:
                continue
            better_or_equal = (
                float(other.get("novelty_mean", 0.0)) >= float(row.get("novelty_mean", 0.0))
                and float(other.get("appropriateness_mean", 0.0)) >= float(row.get("appropriateness_mean", 0.0))
            )
            strictly_better = (
                float(other.get("novelty_mean", 0.0)) > float(row.get("novelty_mean", 0.0))
                or float(other.get("appropriateness_mean", 0.0)) > float(row.get("appropriateness_mean", 0.0))
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(row)

    return {
        "baseline_profile": baseline_profile,
        "exclude_invalid": exclude_invalid,
        "sampler_summary": summary,
        "paired_deltas_vs_baseline": paired_deltas,
        "pareto_profiles": pareto,
    }



def _normalize_model_family(model_id: str) -> tuple[str, bool]:
    m = model_id.lower()
    is_instruct = m.endswith("-it") or ("instruct" in m)
    if m.endswith("-it"):
        m = m[:-3]
    # Keep separator structure while removing instruct marker.
    m = re.sub(r"(?:-|_)?instruct(?:ion)?(?:-|_)?", "-", m)
    m = re.sub(r"[-_]{2,}", "-", m).strip("-_")
    return m, is_instruct



def base_vs_instruct_shift(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"base": [], "instruct": []})

    for rec in records:
        model_id = str(rec.get("model_id", ""))
        family, is_instruct = _normalize_model_family(model_id)
        task_id = str(rec.get("task_id", ""))
        key = (family, task_id)
        objective = frontier_objective(float(rec.get("novelty", 0.0)), float(rec.get("appropriateness", 0.0)))
        grouped[key]["instruct" if is_instruct else "base"].append(objective)

    shifts: list[dict[str, Any]] = []
    for (family, task_id), split in grouped.items():
        if not split["base"] or not split["instruct"]:
            continue
        base_mean = sum(split["base"]) / len(split["base"])
        instruct_mean = sum(split["instruct"]) / len(split["instruct"])
        shifts.append(
            {
                "family": family,
                "task_id": task_id,
                "base_mean_objective": base_mean,
                "instruct_mean_objective": instruct_mean,
                "delta_instruct_minus_base": instruct_mean - base_mean,
            }
        )
    return shifts



def best_of_n_at_budget(
    records: list[dict[str, Any]],
    token_budget: int = 800,
    n_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    n_values = n_values or [1, 2, 4, 8]
    grouped: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)

    for rec in records:
        key = _group_key(rec)
        objective = frontier_objective(float(rec.get("novelty", 0.0)), float(rec.get("appropriateness", 0.0)))
        metadata = rec.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        token_count = int(
            metadata.get("tokens_total", 0)
            or rec.get("tokens_total", 0)
            or metadata.get("token_count", 0)
            or rec.get("token_count", 0)
            or 1
        )
        grouped[key].append((objective, max(token_count, 1)))

    out: list[dict[str, Any]] = []
    for (model_id, method, task_id), vals in grouped.items():
        vals_sorted = sorted(vals, key=lambda x: x[0], reverse=True)
        for n in n_values:
            take = vals_sorted[:n]
            if not take:
                continue
            total_tokens = sum(v[1] for v in take)
            if total_tokens > token_budget:
                scale = token_budget / total_tokens
            else:
                scale = 1.0
            est = max(v[0] for v in take) * scale
            out.append(
                {
                    "model_id": model_id,
                    "method": method,
                    "task_id": task_id,
                    "n": n,
                    "token_budget": token_budget,
                    "estimated_best_objective": est,
                }
            )
    return out


def paired_method_deltas(
    records: list[dict[str, Any]],
    method_a: str,
    method_b: str,
    paired_by: str = "prompt",
    exclude_invalid: bool = False,
) -> dict[str, Any]:
    rows = [r for r in records if _is_primary_valid(r)] if exclude_invalid else records
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)

    for rec in rows:
        model_id = str(rec.get("model_id", ""))
        task_id = str(rec.get("task_id", ""))
        method = str(rec.get("method", ""))
        if paired_by == "task":
            key = (model_id, task_id)
        else:
            key = (model_id, task_id, _prompt_key(rec), _replicate_key(rec))
        grouped[key][method] = rec

    deltas: list[float] = []
    task_deltas: dict[str, list[float]] = defaultdict(list)
    pair_rows: list[dict[str, Any]] = []

    for key, split in grouped.items():
        if method_a not in split or method_b not in split:
            continue
        a = split[method_a]
        b = split[method_b]
        a_obj = frontier_objective(float(a.get("novelty", 0.0)), float(a.get("appropriateness", 0.0)))
        b_obj = frontier_objective(float(b.get("novelty", 0.0)), float(b.get("appropriateness", 0.0)))
        delta = a_obj - b_obj
        deltas.append(delta)
        task_id = str(a.get("task_id", ""))
        task_deltas[task_id].append(delta)
        pair_rows.append(
            {
                "pair_key": str(key),
                "model_id": str(a.get("model_id", "")),
                "task_id": task_id,
                "delta_objective": delta,
                "a_objective": a_obj,
                "b_objective": b_obj,
            }
        )

    ci_low, mean_delta, ci_high = bootstrap_mean_ci(deltas)
    task_summary = []
    for task_id, vals in sorted(task_deltas.items()):
        t_low, t_mean, t_high = bootstrap_mean_ci(vals)
        task_summary.append(
            {
                "task_id": task_id,
                "n": len(vals),
                "mean_delta": t_mean,
                "ci_low": t_low,
                "ci_high": t_high,
                "wins": sum(1 for x in vals if x > 0),
            }
        )

    return {
        "method_a": method_a,
        "method_b": method_b,
        "paired_by": paired_by,
        "n_pairs": len(deltas),
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": sum(1 for x in deltas if x > 0),
        "task_summary": task_summary,
        "pairs": pair_rows,
    }


def _record_cost_tokens(rec: dict[str, Any]) -> int:
    metadata = rec.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return int(
        metadata.get("tokens_total", 0)
        or rec.get("tokens_total", 0)
        or metadata.get("token_count", 0)
        or rec.get("token_count", 0)
        or 0
    )


def _record_calls(rec: dict[str, Any]) -> int:
    metadata = rec.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return int(metadata.get("effective_calls", 0) or rec.get("effective_calls", 0) or metadata.get("generation_calls", 1) or 1)


def _compute_key(rec: dict[str, Any], paired_by: str = "prompt") -> tuple[Any, ...]:
    model_id = str(rec.get("model_id", ""))
    task_id = str(rec.get("task_id", ""))
    if paired_by == "task":
        return (model_id, task_id)
    return (model_id, task_id, _prompt_key(rec), _replicate_key(rec))


def _is_compute_matched(a: dict[str, Any], b: dict[str, Any], token_tolerance: float) -> bool:
    token_a = _record_cost_tokens(a)
    token_b = _record_cost_tokens(b)
    if token_a > 0 and token_b > 0:
        lo = min(token_a, token_b)
        hi = max(token_a, token_b)
        return (lo / hi) >= (1.0 - token_tolerance)

    calls_a = _record_calls(a)
    calls_b = _record_calls(b)
    if calls_a <= 0 or calls_b <= 0:
        return False
    lo = min(calls_a, calls_b)
    hi = max(calls_a, calls_b)
    return (lo / hi) >= (1.0 - token_tolerance)


def _comparison_from_pairs(
    records: list[dict[str, Any]],
    method_a: str,
    method_b: str,
    paired_by: str,
    token_tolerance: float,
    exclude_invalid: bool,
) -> dict[str, Any]:
    rows = [r for r in records if _is_primary_valid(r)] if exclude_invalid else records
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for rec in rows:
        method = str(rec.get("method", ""))
        if method not in {method_a, method_b}:
            continue
        groups[_compute_key(rec, paired_by=paired_by)][method] = rec

    subset: list[dict[str, Any]] = []
    skipped_unmatched = 0
    for split in groups.values():
        if method_a not in split or method_b not in split:
            continue
        a = split[method_a]
        b = split[method_b]
        compute_ok_a = bool(a.get("compute_matched_valid", a.get("valid_for_primary", True)))
        compute_ok_b = bool(b.get("compute_matched_valid", b.get("valid_for_primary", True)))
        if not (compute_ok_a and compute_ok_b):
            skipped_unmatched += 1
            continue
        if not _is_compute_matched(a, b, token_tolerance=token_tolerance):
            skipped_unmatched += 1
            continue
        subset.extend([a, b])

    summary = paired_method_deltas(
        records=subset,
        method_a=method_a,
        method_b=method_b,
        paired_by=paired_by,
        exclude_invalid=exclude_invalid,
    )
    summary["matched_pairs"] = summary.get("n_pairs", 0)
    summary["unmatched_pairs"] = skipped_unmatched
    summary["token_tolerance"] = token_tolerance
    return summary


def compute_matched_summary(
    records: list[dict[str, Any]],
    k: int = 4,
    exclude_invalid: bool = False,
    paired_by: str = "prompt",
    token_tolerance: float = 0.25,
) -> dict[str, Any]:
    primary = _comparison_from_pairs(
        records=records,
        method_a="restlessness_best",
        method_b="best_of_k_one_shot",
        paired_by=paired_by,
        token_tolerance=token_tolerance,
        exclude_invalid=exclude_invalid,
    )
    adaptive = _comparison_from_pairs(
        records=records,
        method_a="restlessness_adaptive",
        method_b="best_of_k_one_shot",
        paired_by=paired_by,
        token_tolerance=token_tolerance,
        exclude_invalid=exclude_invalid,
    )
    baseline = _comparison_from_pairs(
        records=records,
        method_a="best_of_k_one_shot",
        method_b="one_shot",
        paired_by=paired_by,
        token_tolerance=token_tolerance,
        exclude_invalid=exclude_invalid,
    )
    return {
        "k": k,
        "compute_matched_by": paired_by,
        "token_tolerance": token_tolerance,
        "primary_comparison": primary,
        "adaptive_comparison": adaptive,
        "baseline_comparison": baseline,
    }


def efficiency_summary(records: list[dict[str, Any]], exclude_invalid: bool = False) -> list[dict[str, Any]]:
    rows = [r for r in records if _is_primary_valid(r)] if exclude_invalid else records
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in rows:
        grouped[_group_key(rec)].append(rec)

    out: list[dict[str, Any]] = []
    for (model_id, method, task_id), bucket in sorted(grouped.items()):
        objectives = [frontier_objective(float(r.get("novelty", 0.0)), float(r.get("appropriateness", 0.0))) for r in bucket]
        tokens = [_record_cost_tokens(r) for r in bucket if _record_cost_tokens(r) > 0]
        per_1k = [float(r.get("score_per_1k_tokens", 0.0)) for r in bucket if float(r.get("score_per_1k_tokens", 0.0)) > 0]
        calls = [_record_calls(r) for r in bucket]
        out.append(
            {
                "model_id": model_id,
                "method": method,
                "task_id": task_id,
                "n": len(bucket),
                "objective_mean": (sum(objectives) / len(objectives)) if objectives else 0.0,
                "tokens_mean": (sum(tokens) / len(tokens)) if tokens else 0.0,
                "calls_mean": (sum(calls) / len(calls)) if calls else 0.0,
                "score_per_1k_tokens_mean": (sum(per_1k) / len(per_1k)) if per_1k else 0.0,
            }
        )
    return out



def homogeneity_audit_from_runs(
    run_records: list[dict[str, Any]],
    by_task: bool = False,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in run_records:
        method = str(rec.get("method", "unknown-method"))
        task_id = str(rec.get("task_id", "unknown-task"))
        key = (method, task_id if by_task else "")
        output = rec.get("output", [])
        if isinstance(output, list):
            grouped[key].append(flatten_output([str(x) for x in output]))

    rows: list[dict[str, Any]] = []
    for (method, task_id), texts in grouped.items():
        metrics = homogeneity_metrics(texts)
        row = {"method": method, "sample_count": len(texts), **metrics}
        if by_task:
            row["task_id"] = task_id
        rows.append(row)
    return rows



def save_frontier_plot(points: list[FrontierPoint], out_path: str | Path, title: str = "Creativity Frontier") -> str | None:
    if not points:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    methods = sorted({p.method for p in points})
    colors = {m: c for m, c in zip(methods, ["#005f73", "#9b2226", "#0a9396", "#bb3e03", "#ca6702"])}

    plt.figure(figsize=(10, 6))
    for p in points:
        x = p.appropriateness_mean
        y = p.novelty_mean
        err = max(0.0, p.ci_high - p.ci_low) / 2.0
        plt.errorbar(
            x,
            y,
            yerr=err,
            fmt="o",
            color=colors.get(p.method, "#333333"),
            label=f"{p.method}" if p.method not in plt.gca().get_legend_handles_labels()[1] else None,
            alpha=0.85,
        )
        plt.text(x + 0.005, y + 0.005, f"{p.model_id}:{p.task_group}", fontsize=8)

    plt.xlabel("Appropriateness")
    plt.ylabel("Novelty")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend(loc="best")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return str(out)



def compare_backend_trend(local_rows: list[dict[str, Any]], cuda_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def to_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
        out: dict[tuple[str, str, str], float] = {}
        for row in rows:
            key = (
                str(row.get("model_id", "")),
                str(row.get("method", "")),
                str(row.get("task_id", "")),
            )
            out[key] = float(row.get("objective_mean", 0.0))
        return out

    a = to_map(local_rows)
    b = to_map(cuda_rows)
    common = sorted(set(a.keys()) & set(b.keys()))

    if not common:
        return {"hardware_stable": False, "matched_cells": 0, "agreement_ratio": 0.0}

    # Directional stability uses rank-order agreement by pairwise comparisons.
    agree = 0
    total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            ka, kb = common[i], common[j]
            dir_a = math.copysign(1, a[ka] - a[kb]) if a[ka] != a[kb] else 0
            dir_b = math.copysign(1, b[ka] - b[kb]) if b[ka] != b[kb] else 0
            total += 1
            if dir_a == dir_b:
                agree += 1

    ratio = (agree / total) if total else 1.0
    return {
        "hardware_stable": ratio >= 0.75,
        "matched_cells": len(common),
        "agreement_ratio": ratio,
    }
