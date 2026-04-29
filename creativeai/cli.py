from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from creativeai.analysis import (
    base_vs_instruct_shift,
    best_of_n_at_budget,
    compute_matched_summary,
    compare_backend_trend,
    efficiency_summary,
    frontier_points,
    homogeneity_audit_from_runs,
    paired_method_deltas,
    sampler_profile_analysis,
    save_frontier_plot,
)
from creativeai.calibration import evaluate_human_calibration, stratified_human_slice
from creativeai.datasets import default_aut_prompts, default_cdat_cues
from creativeai.decoding import apply_sampler_profile, sampler_profile_names
from creativeai.io_utils import append_jsonl, environment_snapshot, infer_records, snapshot_tabular, write_json
from creativeai.methods import build_method_runner
from creativeai.model_backend import create_model_adapter
from creativeai.pipeline import generate_run, save_analysis_artifact, save_score_records
from creativeai.scoring import compute_score_record
from creativeai.schemas import GenerationConfig, utc_now_iso
from creativeai.tasks import build_task



def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]



def _csv_ints(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]



def _csv_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _csv_sampler_profiles(value: str) -> list[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    return profiles or ["manual"]


def _format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    whole = int(seconds)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _new_session_id() -> str:
    ts = int(time.time())
    return f"sess-{ts}-{uuid.uuid4().hex[:8]}"


def _extract_session_id(record: dict[str, Any]) -> str:
    direct = str(record.get("session_id", "")).strip()
    if direct:
        return direct

    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        sid = str(metadata.get("session_id", "")).strip()
        if sid:
            return sid

    manifest = record.get("manifest", {})
    if isinstance(manifest, dict):
        sid = str(manifest.get("session_id", "")).strip()
        if sid:
            return sid
        extra = manifest.get("extra", {})
        if isinstance(extra, dict):
            sid = str(extra.get("session_id", "")).strip()
            if sid:
                return sid
    return ""


def _require_single_session(records: list[dict[str, Any]], source_label: str) -> str:
    session_ids: set[str] = set()
    for record in records:
        sid = _extract_session_id(record)
        session_ids.add(sid if sid else "<missing>")

    if not session_ids:
        return ""
    if "<missing>" in session_ids:
        raise RuntimeError(
            f"Session lineage check failed for {source_label}: missing session_id on at least one record."
        )
    if len(session_ids) > 1:
        joined = ", ".join(sorted(session_ids))
        raise RuntimeError(f"Mixed sessions detected in {source_label}: {joined}")
    return next(iter(session_ids))



def _load_model_path_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("model path map must be a JSON object {model_id: model_path}")
    return {str(k): str(v) for k, v in payload.items()}



def _build_generation_config(args: argparse.Namespace, model_id: str, seed: int, temperature: float) -> GenerationConfig:
    stop_tokens = _csv_list(args.stop) if getattr(args, "stop", "") else []
    cfg = GenerationConfig(
        model_id=model_id,
        backend=args.backend,
        temperature=temperature,
        top_p=args.top_p,
        seed=seed,
        max_tokens=args.max_tokens,
        quantization=args.quantization,
        strict_json=bool(getattr(args, "strict_json", False)),
        max_retries=int(getattr(args, "max_retries", 2)),
        prompt_mode=str(getattr(args, "prompt_mode", "auto")),
        grammar_mode=str(getattr(args, "grammar_mode", "auto")),
        stop=stop_tokens,
        sampler_profile=str(getattr(args, "sampler_profile", "manual")),
        top_k=max(0, int(getattr(args, "top_k", 0))),
        min_p=max(0.0, float(getattr(args, "min_p", 0.0))),
        typical_p=max(0.0, float(getattr(args, "typical_p", 0.0))),
        repeat_penalty=max(0.0, float(getattr(args, "repeat_penalty", 1.0))),
        frequency_penalty=float(getattr(args, "frequency_penalty", 0.0)),
        presence_penalty=float(getattr(args, "presence_penalty", 0.0)),
        mirostat_mode=max(0, int(getattr(args, "mirostat_mode", 0))),
        mirostat_tau=max(0.0, float(getattr(args, "mirostat_tau", 0.0))),
        mirostat_eta=max(0.0, float(getattr(args, "mirostat_eta", 0.0))),
        n_ctx=int(getattr(args, "n_ctx", 4096)),
        n_threads=int(getattr(args, "n_threads", 0)),
        n_batch=int(getattr(args, "n_batch", 512)),
        n_ubatch=int(getattr(args, "n_ubatch", 512)),
        n_threads_batch=int(getattr(args, "n_threads_batch", 0)),
        token_budget_per_prompt=(
            int(getattr(args, "token_budget_per_prompt", 0)) if int(getattr(args, "token_budget_per_prompt", 0) or 0) > 0 else None
        ),
        compute_tag=str(getattr(args, "compute_tag", "")),
        adaptive_stop_delta=(
            float(getattr(args, "adaptive_stop_delta", 0.0))
            if getattr(args, "adaptive_stop_delta", None) is not None and float(getattr(args, "adaptive_stop_delta", 0.0)) > 0
            else None
        ),
        adaptive_min_iters=max(1, int(getattr(args, "adaptive_min_iters", 1))),
        trigger_objective=(
            float(getattr(args, "trigger_objective", 0.0))
            if getattr(args, "trigger_objective", None) is not None and float(getattr(args, "trigger_objective", 0.0)) > 0
            else None
        ),
    )
    return apply_sampler_profile(cfg)



def cmd_generate(args: argparse.Namespace) -> int:
    task = build_task(args.task, cue=args.cue, obj=args.object_name, context=args.context)
    cfg = _build_generation_config(args, model_id=args.model, seed=args.seed, temperature=args.temperature)
    session_id = str(getattr(args, "session_id", "")).strip() or _new_session_id()

    model = create_model_adapter(
        model_id=args.model,
        backend=args.backend,
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        n_threads_batch=args.n_threads_batch,
    )
    method = build_method_runner(
        args.method,
        restlessness_k=args.restlessness_k,
        best_of_k=args.best_of_k,
        adaptive_stop_delta=getattr(args, "adaptive_stop_delta", None),
        adaptive_min_iters=getattr(args, "adaptive_min_iters", 1),
        trigger_objective=getattr(args, "trigger_objective", None),
    )

    record = generate_run(
        task,
        method,
        model,
        cfg,
        args.output_dir,
        session_id=session_id,
        phase3_stage=str(getattr(args, "stage", "")).strip(),
    )
    print(json.dumps(record.to_dict(), indent=2))
    return 0



def cmd_generate_grid(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_jsonl = out_dir / "runs.jsonl"
    if runs_jsonl.exists() and not bool(getattr(args, "append_runs", False)):
        raise RuntimeError(
            f"Refusing to append into existing run file: {runs_jsonl}. "
            "Use a new --output-dir or pass --append-runs to continue."
        )

    session_id = str(getattr(args, "session_id", "")).strip() or _new_session_id()
    tasks = _csv_list(args.tasks)
    models = _csv_list(args.models)
    methods = _csv_list(args.methods)
    temperatures = _csv_floats(args.temperatures)
    sampler_profiles = _csv_sampler_profiles(getattr(args, "sampler_profiles", "manual"))
    seeds = _csv_ints(args.seeds)
    model_paths = _load_model_path_map(args.model_path_map)
    progress_every = max(1, int(getattr(args, "progress_every", 10)))
    show_progress = bool(getattr(args, "progress", True))
    health_window = max(1, int(getattr(args, "health_window", 20)))
    health_min_json = float(getattr(args, "health_min_json", 0.95))
    health_min_valid = float(getattr(args, "health_min_valid", 0.90))
    health_min_samples = max(1, int(getattr(args, "health_min_samples", 20)))
    health_action = str(getattr(args, "health_action", "quarantine_cell")).strip().lower()
    if health_action not in {"quarantine_cell", "stop_run", "retry_once"}:
        raise ValueError(f"Unsupported --health-action: {health_action}")
    if not (0.0 <= health_min_json <= 1.0):
        raise ValueError("--health-min-json must be in [0,1]")
    if not (0.0 <= health_min_valid <= 1.0):
        raise ValueError("--health-min-valid must be in [0,1]")

    health_events_raw = getattr(args, "health_events", None)
    health_events_arg = str(health_events_raw).strip() if health_events_raw is not None else ""
    if health_events_arg.lower() == "none":
        health_events_arg = ""
    health_events_path = Path(health_events_arg) if health_events_arg else (out_dir / "health_events.jsonl")
    if health_events_path.exists() and not bool(getattr(args, "append_runs", False)):
        health_events_path.unlink()

    phase3_stage = str(getattr(args, "stage", "")).strip().lower()
    if not str(getattr(args, "compute_tag", "")).strip():
        setattr(args, "compute_tag", phase3_stage or "default")
    dat_repeats = max(1, int(getattr(args, "dat_repeats", 1)))
    cue_offset = max(0, int(getattr(args, "cue_offset", 0)))
    aut_offset = max(0, int(getattr(args, "aut_offset", 0)))
    token_budget_cap = int(getattr(args, "token_budget_cap", 0) or 0)
    if token_budget_cap < 0:
        raise ValueError("--token-budget-cap must be >= 0")

    cues = default_cdat_cues()
    if cue_offset:
        cues = cues[cue_offset:]
    if args.limit_cues:
        cues = cues[: args.limit_cues]
    aut_prompts = default_aut_prompts()
    if aut_offset:
        aut_prompts = aut_prompts[aut_offset:]
    if args.limit_aut:
        aut_prompts = aut_prompts[: args.limit_aut]

    total_planned = 0
    for _model_id in models:
        for method_name in methods:
            for _profile in sampler_profiles:
                for _temp in temperatures:
                    for _seed in seeds:
                        if "dat" in tasks and method_name != "brainstorm_then_select":
                            total_planned += dat_repeats
                        if "cdat" in tasks and method_name != "brainstorm_then_select":
                            total_planned += len(cues)
                        if "aut" in tasks:
                            total_planned += len(aut_prompts)

    start = time.monotonic()
    last_print_done = 0
    if show_progress:
        print(
            f"[generate-grid] planned_runs={total_planned} "
            f"models={len(models)} methods={len(methods)} sampler_profiles={len(sampler_profiles)} "
            f"temps={len(temperatures)} seeds={len(seeds)} "
            f"session_id={session_id} stage={phase3_stage or 'none'} token_budget_cap={token_budget_cap or 0}",
            flush=True,
        )

    total_created = 0
    total_done = 0
    total_skipped = 0
    total_health_events = 0
    quarantined_cells = 0
    total_tokens = 0
    budget_exhausted = False
    health_state: dict[tuple[str, str, str, str, int, float], dict[str, Any]] = {}

    def report_progress(task_label: str, model_id: str, method_name: str, sampler_profile: str) -> None:
        nonlocal last_print_done
        if not show_progress:
            return
        should_print = (
            total_done <= 1
            or total_done == total_planned
            or (total_done - last_print_done) >= progress_every
        )
        if not should_print:
            return

        elapsed = max(0.0, time.monotonic() - start)
        rate = (total_done / elapsed) if elapsed > 0 else 0.0
        remaining = max(0, total_planned - total_done)
        eta_seconds = (remaining / rate) if rate > 0 else 0.0
        pct = (100.0 * total_done / total_planned) if total_planned > 0 else 100.0
        print(
            f"[{total_done}/{total_planned} {pct:5.1f}%] "
            f"elapsed={_format_seconds(elapsed)} "
            f"eta={_format_seconds(eta_seconds)} "
            f"rate={rate:.2f} runs/s "
            f"model={model_id} method={method_name} profile={sampler_profile} task={task_label}",
            flush=True,
        )
        last_print_done = total_done

    def emit_health_event(
        *,
        event_type: str,
        cell_key: tuple[str, str, str, str, int, float],
        window_n: int,
        json_rate: float,
        valid_rate: float,
        action: str,
        skipped_estimate: int,
    ) -> None:
        nonlocal total_health_events
        payload = {
            "event_type": event_type,
            "cell_key": {
                "model_id": cell_key[0],
                "method": cell_key[1],
                "task_id": cell_key[2],
                "sampler_profile": cell_key[3],
                "seed": cell_key[4],
                "temperature": cell_key[5],
            },
            "window_n": window_n,
            "json_rate": round(json_rate, 6),
            "valid_rate": round(valid_rate, 6),
            "action": action,
            "timestamp_utc": utc_now_iso(),
            "skipped_estimate": int(skipped_estimate),
            "session_id": session_id,
        }
        append_jsonl(health_events_path, payload)
        total_health_events += 1
        if show_progress:
            print(
                f"[health] event={event_type} action={action} "
                f"model={cell_key[0]} method={cell_key[1]} task={cell_key[2]} "
                f"profile={cell_key[3]} seed={cell_key[4]} temp={cell_key[5]} "
                f"window_n={window_n} json_rate={json_rate:.3f} valid_rate={valid_rate:.3f} "
                f"skipped={skipped_estimate}",
                flush=True,
            )

    def run_cell(
        *,
        model_id: str,
        method_name: str,
        method: Any,
        model: Any,
        cfg: GenerationConfig,
        seed: int,
        task_id: str,
        cells: list[dict[str, Any]],
    ) -> None:
        nonlocal total_created, total_done, total_skipped, quarantined_cells, total_tokens, budget_exhausted
        if not cells:
            return
        cell_key = (model_id, method_name, task_id, cfg.sampler_profile, seed, cfg.temperature)
        state = health_state.setdefault(
            cell_key,
            {"window": deque(maxlen=health_window), "retry_used": False, "quarantined": False},
        )
        for idx, spec in enumerate(cells):
            if budget_exhausted or bool(state["quarantined"]):
                break

            if token_budget_cap > 0 and total_tokens >= token_budget_cap:
                remaining = max(0, len(cells) - idx)
                total_skipped += remaining
                total_done += remaining
                budget_exhausted = True
                if show_progress:
                    print(
                        f"[budget] token_budget_cap reached cap={token_budget_cap} tokens={total_tokens} "
                        f"skipping_remaining={remaining} cell={cell_key}",
                        flush=True,
                    )
                report_progress(f"{task_id}-budget-stop", model_id, method_name, cfg.sampler_profile)
                break

            task = build_task(
                task_id,
                cue=spec.get("cue"),
                obj=spec.get("object"),
                context=spec.get("context"),
            )
            record = generate_run(
                task,
                method,
                model,
                cfg,
                args.output_dir,
                session_id=session_id,
                extra_metadata=spec.get("extra_metadata"),
                phase3_stage=phase3_stage,
            )
            total_created += 1
            total_done += 1
            total_tokens += int(getattr(record, "tokens_total", 0) or 0)
            report_progress(task_id, model_id, method_name, cfg.sampler_profile)

            rolling = state["window"]
            rolling.append(
                (
                    bool(record.json_valid),
                    bool(record.validity_flags.get("valid", False)),
                )
            )
            if len(rolling) < health_min_samples:
                continue

            window_n = len(rolling)
            json_rate = sum(1 for j, _ in rolling if j) / window_n
            valid_rate = sum(1 for _, v in rolling if v) / window_n
            gate_failed = json_rate < health_min_json or valid_rate < health_min_valid
            if not gate_failed:
                continue

            remaining = max(0, len(cells) - idx - 1)
            if health_action == "retry_once" and not bool(state["retry_used"]):
                state["retry_used"] = True
                rolling.clear()
                emit_health_event(
                    event_type="health_gate_trip",
                    cell_key=cell_key,
                    window_n=window_n,
                    json_rate=json_rate,
                    valid_rate=valid_rate,
                    action="retry_once",
                    skipped_estimate=0,
                )
                continue

            if health_action == "stop_run":
                emit_health_event(
                    event_type="health_gate_trip",
                    cell_key=cell_key,
                    window_n=window_n,
                    json_rate=json_rate,
                    valid_rate=valid_rate,
                    action="stop_run",
                    skipped_estimate=remaining,
                )
                raise RuntimeError(
                    f"Health gate stop: cell={cell_key} "
                    f"json_rate={json_rate:.3f} valid_rate={valid_rate:.3f}"
                )

            state["quarantined"] = True
            quarantined_cells += 1
            total_skipped += remaining
            total_done += remaining
            emit_health_event(
                event_type="cell_quarantined",
                cell_key=cell_key,
                window_n=window_n,
                json_rate=json_rate,
                valid_rate=valid_rate,
                action="quarantine_cell",
                skipped_estimate=remaining,
            )
            report_progress(f"{task_id}-quarantined", model_id, method_name, cfg.sampler_profile)
            break

    for model_id in models:
        model = create_model_adapter(
            model_id=model_id,
            backend=args.backend,
            model_path=model_paths.get(model_id, args.model_path),
            n_gpu_layers=args.n_gpu_layers,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            n_batch=args.n_batch,
            n_ubatch=args.n_ubatch,
            n_threads_batch=args.n_threads_batch,
        )

        for method_name in methods:
            method = build_method_runner(
                method_name,
                restlessness_k=args.restlessness_k,
                best_of_k=args.best_of_k,
                adaptive_stop_delta=getattr(args, "adaptive_stop_delta", None),
                adaptive_min_iters=getattr(args, "adaptive_min_iters", 1),
                trigger_objective=getattr(args, "trigger_objective", None),
            )
            for sampler_profile in sampler_profiles:
                setattr(args, "sampler_profile", sampler_profile)
                for temp in temperatures:
                    for seed in seeds:
                        cfg = _build_generation_config(args, model_id=model_id, seed=seed, temperature=temp)

                        if "dat" in tasks and method_name != "brainstorm_then_select":
                            run_cell(
                                model_id=model_id,
                                method_name=method_name,
                                method=method,
                                model=model,
                                cfg=cfg,
                                seed=seed,
                                task_id="dat",
                                cells=[
                                    {
                                        "cue": None,
                                        "object": None,
                                        "context": None,
                                        "extra_metadata": {"dat_prompt_id": idx},
                                    }
                                    for idx in range(dat_repeats)
                                ],
                            )

                        if "cdat" in tasks and method_name != "brainstorm_then_select":
                            cdat_cells = [
                                {
                                    "cue": cue_spec.cue,
                                    "object": None,
                                    "context": None,
                                    "extra_metadata": {
                                        "cue_category": cue_spec.category,
                                        "cue_index": cue_offset + idx,
                                    },
                                }
                                for idx, cue_spec in enumerate(cues)
                            ]
                            run_cell(
                                model_id=model_id,
                                method_name=method_name,
                                method=method,
                                model=model,
                                cfg=cfg,
                                seed=seed,
                                task_id="cdat",
                                cells=cdat_cells,
                            )
                            if budget_exhausted:
                                break

                        if "aut" in tasks:
                            aut_cells = [
                                {
                                    "cue": None,
                                    "object": p.object_name,
                                    "context": p.context,
                                    "extra_metadata": {"aut_prompt_index": aut_offset + idx},
                                }
                                for idx, p in enumerate(aut_prompts)
                            ]
                            run_cell(
                                model_id=model_id,
                                method_name=method_name,
                                method=method,
                                model=model,
                                cfg=cfg,
                                seed=seed,
                                task_id="aut",
                                cells=aut_cells,
                            )
                            if budget_exhausted:
                                break
                    if budget_exhausted:
                        break
                if budget_exhausted:
                    break
            if budget_exhausted:
                break
        if budget_exhausted:
            break

    experiment_manifest = {
        "schema_version": 1,
        "kind": "creativeai_experiment_manifest",
        "created_at_utc": utc_now_iso(),
        "session_id": session_id,
        "stage": phase3_stage,
        "command": sys.argv,
        "grid": {
            "tasks": tasks,
            "models": models,
            "methods": methods,
            "temperatures": temperatures,
            "seeds": seeds,
            "sampler_profiles": sampler_profiles,
            "top_p": float(getattr(args, "top_p", 0.0)),
            "top_k": int(getattr(args, "top_k", 0)),
            "min_p": float(getattr(args, "min_p", 0.0)),
            "typical_p": float(getattr(args, "typical_p", 0.0)),
            "repeat_penalty": float(getattr(args, "repeat_penalty", 1.0)),
            "frequency_penalty": float(getattr(args, "frequency_penalty", 0.0)),
            "presence_penalty": float(getattr(args, "presence_penalty", 0.0)),
            "mirostat_mode": int(getattr(args, "mirostat_mode", 0)),
            "mirostat_tau": float(getattr(args, "mirostat_tau", 0.0)),
            "mirostat_eta": float(getattr(args, "mirostat_eta", 0.0)),
            "strict_json": bool(getattr(args, "strict_json", False)),
            "max_retries": int(getattr(args, "max_retries", 0)),
            "prompt_mode": str(getattr(args, "prompt_mode", "")),
            "grammar_mode": str(getattr(args, "grammar_mode", "")),
        },
        "runtime": {
            "backend": str(getattr(args, "backend", "")),
            "model_path_map": str(getattr(args, "model_path_map", "")),
            "n_gpu_layers": int(getattr(args, "n_gpu_layers", 0)),
            "n_ctx": int(getattr(args, "n_ctx", 0)),
            "n_threads": int(getattr(args, "n_threads", 0)),
            "n_batch": int(getattr(args, "n_batch", 0)),
            "n_ubatch": int(getattr(args, "n_ubatch", 0)),
            "n_threads_batch": int(getattr(args, "n_threads_batch", 0)),
        },
        "health_gate": {
            "health_window": health_window,
            "health_min_json": health_min_json,
            "health_min_valid": health_min_valid,
            "health_min_samples": health_min_samples,
            "health_action": health_action,
            "health_events_path": str(health_events_path),
        },
        "outputs": {
            "runs_dir": str(out_dir),
            "runs_jsonl": str(runs_jsonl),
            "experiment_manifest": str(out_dir / "experiment_manifest.json"),
        },
        "results": {
            "planned_runs": total_planned,
            "runs_created": total_created,
            "runs_skipped": total_skipped,
            "quarantined_cells": quarantined_cells,
            "health_events": total_health_events,
            "tokens_total": total_tokens,
            "token_budget_cap": token_budget_cap,
            "budget_exhausted": budget_exhausted,
        },
        "environment": environment_snapshot(),
    }
    write_json(out_dir / "experiment_manifest.json", experiment_manifest)

    print(
        json.dumps(
            {
                "status": "ok",
                "session_id": session_id,
                "runs_created": total_created,
                "runs_skipped": total_skipped,
                "quarantined_cells": quarantined_cells,
                "health_events": total_health_events,
                "health_events_path": str(health_events_path),
                "tokens_total": total_tokens,
                "token_budget_cap": token_budget_cap,
                "budget_exhausted": budget_exhausted,
                "stage": phase3_stage,
                "output_dir": args.output_dir,
            },
            indent=2,
        )
    )
    return 0



def cmd_score(args: argparse.Namespace) -> int:
    runs = infer_records(args.input)
    session_id = ""
    if bool(getattr(args, "require_single_session", True)):
        session_id = _require_single_session(runs, source_label=f"runs input ({args.input})")
    scored: list[dict[str, Any]] = []
    for run in runs:
        score = compute_score_record(run)
        payload = score.to_dict()
        run_session_id = _extract_session_id(run)
        payload["session_id"] = run_session_id
        payload["metadata"] = dict(payload.get("metadata", {}))
        payload["metadata"]["session_id"] = run_session_id
        payload["metadata"]["token_count"] = run.get("token_count", 0)
        payload["metadata"]["generation_calls"] = run.get("metadata", {}).get("generation_calls", 1)
        payload["metadata"]["effective_calls"] = run.get("effective_calls", run.get("metadata", {}).get("effective_calls", 1))
        payload["metadata"]["tokens_in"] = run.get("tokens_in", run.get("metadata", {}).get("tokens_in", 0))
        payload["metadata"]["tokens_out"] = run.get("tokens_out", run.get("metadata", {}).get("tokens_out", 0))
        payload["metadata"]["tokens_total"] = run.get("tokens_total", run.get("metadata", {}).get("tokens_total", 0))
        payload["metadata"]["compute_group_id"] = run.get(
            "compute_group_id", run.get("metadata", {}).get("compute_group_id", "")
        )
        payload["metadata"]["decoding_fingerprint"] = run.get("metadata", {}).get("decoding_fingerprint", "")
        payload["metadata"]["decoding_settings"] = run.get("metadata", {}).get("decoding_settings", {})
        payload["metadata"]["sampler_profile"] = run.get("metadata", {}).get(
            "sampler_profile", run.get("config", {}).get("sampler_profile", "")
        )
        payload["metadata"]["phase3_stage"] = run.get("phase3_stage", run.get("metadata", {}).get("phase3_stage", ""))
        payload["metadata"]["parse_mode"] = run.get("parse_mode", run.get("metadata", {}).get("parse_mode", "unknown"))
        payload["metadata"]["retry_count"] = run.get("retry_count", run.get("metadata", {}).get("retry_count", 0))
        payload["metadata"]["json_valid"] = run.get("json_valid", run.get("validity_flags", {}).get("json_valid", False))
        payload["metadata"]["output_preview"] = " | ".join(run.get("output", [])[:3])
        scored.append(payload)

    jsonl_path, snapshot_path = save_score_records(scored, args.output_dir, append=bool(getattr(args, "append_scores", False)))
    reported_session = session_id
    if not reported_session and scored:
        reported_session = _extract_session_id(scored[0])
    print(
        json.dumps(
            {
                "scores_jsonl": jsonl_path,
                "snapshot": snapshot_path,
                "count": len(scored),
                "session_id": reported_session,
            },
            indent=2,
        )
    )
    return 0



def cmd_analyze_frontier(args: argparse.Namespace) -> int:
    scores = infer_records(args.runs)
    if not (0.0 <= float(args.compute_matched_token_tolerance) < 1.0):
        raise ValueError("--compute-matched-token-tolerance must be in [0,1)")
    session_id = ""
    if bool(getattr(args, "require_single_session", True)):
        session_id = _require_single_session(scores, source_label=f"scores input ({args.runs})")
    analysis_rows = scores
    if args.exclude_invalid:
        analysis_rows = [r for r in scores if bool(r.get("valid_for_primary", False))]

    points, extended = frontier_points(scores, exclude_invalid=args.exclude_invalid)
    shifts = base_vs_instruct_shift(analysis_rows)
    bestn = best_of_n_at_budget(analysis_rows, token_budget=args.token_budget)

    primary_method = "restlessness_best"
    if not any(str(r.get("method", "")) == primary_method for r in analysis_rows):
        primary_method = "restlessness_loop"
    paired = paired_method_deltas(
        analysis_rows,
        method_a=primary_method,
        method_b="one_shot",
        paired_by=args.paired_by,
        exclude_invalid=args.exclude_invalid,
    )
    compute_matched = compute_matched_summary(
        analysis_rows,
        k=args.compute_matched_k,
        exclude_invalid=args.exclude_invalid,
        paired_by=args.compute_matched_by,
        token_tolerance=args.compute_matched_token_tolerance,
    )
    efficiency = efficiency_summary(analysis_rows, exclude_invalid=args.exclude_invalid)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = save_frontier_plot(points, out_dir / "frontier.png")
    payload = {
        "frontier_points": [p.to_dict() for p in points],
        "frontier_summary": extended,
        "base_vs_instruct_shift": shifts,
        "best_of_n": bestn,
        "paired_deltas": paired,
        "compute_matched": compute_matched,
        "efficiency_summary": efficiency,
        "exclude_invalid": bool(args.exclude_invalid),
        "paired_by": args.paired_by,
        "compute_matched_by": args.compute_matched_by,
        "compute_matched_token_tolerance": args.compute_matched_token_tolerance,
        "session_id": session_id,
        "plot": png_path,
    }
    out_json = out_dir / "frontier_analysis.json"
    save_analysis_artifact(payload, out_json)
    print(
        json.dumps(
            {
                "analysis": str(out_json),
                "plot": png_path,
                "point_count": len(points),
                "session_id": session_id,
            },
            indent=2,
        )
    )
    return 0



def cmd_audit_homogeneity(args: argparse.Namespace) -> int:
    runs = infer_records(args.runs)
    rows = homogeneity_audit_from_runs(runs, by_task=args.by_task)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "homogeneity_audit.json"
    write_json(out_json, {"rows": rows})
    snapshot = snapshot_tabular(rows, out_dir / "homogeneity_audit_snapshot.parquet")

    print(json.dumps({"audit": str(out_json), "snapshot": snapshot, "count": len(rows)}, indent=2))
    return 0


def cmd_analyze_samplers(args: argparse.Namespace) -> int:
    scores = infer_records(args.scores)
    session_id = ""
    if bool(getattr(args, "require_single_session", True)):
        session_id = _require_single_session(scores, source_label=f"scores input ({args.scores})")
    payload = sampler_profile_analysis(
        scores,
        baseline_profile=str(args.baseline_profile),
        exclude_invalid=bool(args.exclude_invalid),
    )
    payload["session_id"] = session_id
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "sampler_analysis.json"
    write_json(out_json, payload)

    report_path = out_dir / "SAMPLER_ANALYSIS_REPORT.md"
    lines = [
        "# Phase 4 Sampler Analysis Report",
        "",
        f"- Baseline sampler profile: `{args.baseline_profile}`",
        f"- Session: `{session_id or 'unknown'}`",
        f"- Exclude invalid: `{bool(args.exclude_invalid)}`",
        "",
        "## Sampler Means",
    ]
    for row in payload.get("sampler_summary", []):
        lines.append(
            "- {profile}: n={n} objective={obj:.4f} novelty={nov:.4f} appropriateness={app:.4f} "
            "valid={valid:.2%} json={json_rate:.2%}".format(
                profile=row.get("sampler_profile", "unknown"),
                n=int(row.get("n_primary_valid", 0)),
                obj=float(row.get("objective_mean", 0.0)),
                nov=float(row.get("novelty_mean", 0.0)),
                app=float(row.get("appropriateness_mean", 0.0)),
                valid=float(row.get("primary_valid_rate", 0.0)),
                json_rate=float(row.get("json_valid_rate", 0.0)),
            )
        )
    lines.extend(["", "## Paired Deltas vs Baseline"])
    for row in payload.get("paired_deltas_vs_baseline", []):
        lines.append(
            "- {profile}: pairs={n} obj_delta={mean:+.4f} [{low:+.4f},{high:+.4f}] "
            "nov_delta={nov:+.4f} app_delta={app:+.4f}".format(
                profile=row.get("sampler_profile", "unknown"),
                n=int(row.get("n_pairs", 0)),
                mean=float(row.get("objective_delta_mean", 0.0)),
                low=float(row.get("objective_delta_ci_low", 0.0)),
                high=float(row.get("objective_delta_ci_high", 0.0)),
                nov=float(row.get("novelty_delta_mean", 0.0)),
                app=float(row.get("appropriateness_delta_mean", 0.0)),
            )
        )
    lines.extend(["", "## Pareto Profiles"])
    for row in payload.get("pareto_profiles", []):
        lines.append(
            "- {profile}: novelty={nov:.4f} appropriateness={app:.4f} objective={obj:.4f}".format(
                profile=row.get("sampler_profile", "unknown"),
                nov=float(row.get("novelty_mean", 0.0)),
                app=float(row.get("appropriateness_mean", 0.0)),
                obj=float(row.get("objective_mean", 0.0)),
            )
        )
    lines.extend(["", f"Raw JSON: `{out_json}`", ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "analysis": str(out_json),
                "report": str(report_path),
                "profile_count": len(payload.get("sampler_summary", [])),
                "session_id": session_id,
            },
            indent=2,
        )
    )
    return 0



def cmd_prepare_human_slice(args: argparse.Namespace) -> int:
    scores = infer_records(args.scores)
    rows = stratified_human_slice(scores, n=args.n, seed=args.seed)
    out = Path(args.output)
    write_json(out, {"rows": rows})
    print(json.dumps({"output": str(out), "count": len(rows)}, indent=2))
    return 0



def cmd_eval_human(args: argparse.Namespace) -> int:
    payload = infer_records(args.ratings)
    if payload and isinstance(payload[0], dict) and "rows" in payload[0]:
        ratings = payload[0]["rows"]
    else:
        ratings = payload
    result = evaluate_human_calibration(ratings, target_corr=args.target_corr)
    out = Path(args.output)
    write_json(out, result)
    print(json.dumps({"output": str(out), **result}, indent=2))
    return 0



def cmd_compare_backends(args: argparse.Namespace) -> int:
    local_payload = infer_records(args.local_frontier)
    cuda_payload = infer_records(args.cuda_frontier)

    local_rows = local_payload[0].get("frontier_summary", []) if local_payload and "frontier_summary" in local_payload[0] else local_payload
    cuda_rows = cuda_payload[0].get("frontier_summary", []) if cuda_payload and "frontier_summary" in cuda_payload[0] else cuda_payload

    result = compare_backend_trend(local_rows, cuda_rows)
    out = Path(args.output)
    write_json(out, result)
    print(json.dumps({"output": str(out), **result}, indent=2))
    return 0



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="creativeai", description="Creativity-under-constraints research harness")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate one run")
    gen.add_argument("--task", choices=["dat", "cdat", "aut"], required=True)
    gen.add_argument(
        "--method",
        choices=[
            "one_shot",
            "best_of_k_one_shot",
            "naive_multiturn",
            "dialogue_multiturn",
            "brainstorm_then_select",
            "restlessness_loop",
            "restlessness_best",
            "restlessness_triggered",
            "restlessness_last_iter",
            "restlessness_adaptive",
        ],
        required=True,
    )
    gen.add_argument("--model", required=True)
    gen.add_argument("--backend", default="mock", help="mock | llama_cpp")
    gen.add_argument("--model-path", default=None)
    gen.add_argument("--n-gpu-layers", type=int, default=-1)
    gen.add_argument("--n-ctx", type=int, default=4096)
    gen.add_argument("--n-threads", type=int, default=0)
    gen.add_argument("--n-batch", type=int, default=512)
    gen.add_argument("--n-ubatch", type=int, default=512)
    gen.add_argument("--n-threads-batch", type=int, default=0)
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--temperature", type=float, required=True)
    gen.add_argument("--top-p", type=float, default=0.9)
    gen.add_argument("--sampler-profile", choices=sampler_profile_names(), default="manual")
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--min-p", type=float, default=0.0)
    gen.add_argument("--typical-p", type=float, default=0.0)
    gen.add_argument("--repeat-penalty", type=float, default=1.0)
    gen.add_argument("--frequency-penalty", type=float, default=0.0)
    gen.add_argument("--presence-penalty", type=float, default=0.0)
    gen.add_argument("--mirostat-mode", type=int, default=0)
    gen.add_argument("--mirostat-tau", type=float, default=0.0)
    gen.add_argument("--mirostat-eta", type=float, default=0.0)
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--quantization", default="q4_k_m")
    gen.add_argument("--strict-json", action=argparse.BooleanOptionalAction, default=True)
    gen.add_argument("--max-retries", type=int, default=2)
    gen.add_argument("--prompt-mode", choices=["completion", "chat", "auto"], default="auto")
    gen.add_argument("--grammar-mode", choices=["auto", "word_list", "idea_list"], default="auto")
    gen.add_argument("--stop", default="")
    gen.add_argument("--restlessness-k", type=int, default=3)
    gen.add_argument("--best-of-k", type=int, default=4)
    gen.add_argument("--adaptive-stop-delta", type=float, default=0.0)
    gen.add_argument("--adaptive-min-iters", type=int, default=1)
    gen.add_argument("--trigger-objective", type=float, default=0.0)
    gen.add_argument("--token-budget-per-prompt", type=int, default=0)
    gen.add_argument("--compute-tag", default="")
    gen.add_argument("--stage", choices=["micro", "main", "confirm", "aux", "phase4"], default="main")
    gen.add_argument("--cue", default=None)
    gen.add_argument("--object", dest="object_name", default=None)
    gen.add_argument("--context", default=None)
    gen.add_argument("--session-id", default="")
    gen.add_argument("--output-dir", default="outputs/runs")
    gen.set_defaults(func=cmd_generate)

    grid = sub.add_parser("generate-grid", help="Generate full experiment grid")
    grid.add_argument("--tasks", default="dat,cdat,aut")
    grid.add_argument(
        "--methods",
        default="one_shot,best_of_k_one_shot,restlessness_best",
    )
    grid.add_argument("--models", default="gemma-2-2b,gemma-2-2b-it,qwen2.5-3b,qwen2.5-3b-instruct,mistral-7b-v0.3,mistral-7b-instruct-v0.3")
    grid.add_argument("--backend", default="mock")
    grid.add_argument("--model-path", default=None)
    grid.add_argument("--model-path-map", default=None)
    grid.add_argument("--n-gpu-layers", type=int, default=-1)
    grid.add_argument("--n-ctx", type=int, default=4096)
    grid.add_argument("--n-threads", type=int, default=0)
    grid.add_argument("--n-batch", type=int, default=512)
    grid.add_argument("--n-ubatch", type=int, default=512)
    grid.add_argument("--n-threads-batch", type=int, default=0)
    grid.add_argument("--temperatures", default="0.2,0.7,1.0,1.3")
    grid.add_argument("--seeds", default="11,37,73,101,149")
    grid.add_argument("--top-p", type=float, default=0.9)
    grid.add_argument("--sampler-profile", choices=sampler_profile_names(), default="manual")
    grid.add_argument("--sampler-profiles", default="manual")
    grid.add_argument("--top-k", type=int, default=0)
    grid.add_argument("--min-p", type=float, default=0.0)
    grid.add_argument("--typical-p", type=float, default=0.0)
    grid.add_argument("--repeat-penalty", type=float, default=1.0)
    grid.add_argument("--frequency-penalty", type=float, default=0.0)
    grid.add_argument("--presence-penalty", type=float, default=0.0)
    grid.add_argument("--mirostat-mode", type=int, default=0)
    grid.add_argument("--mirostat-tau", type=float, default=0.0)
    grid.add_argument("--mirostat-eta", type=float, default=0.0)
    grid.add_argument("--max-tokens", type=int, default=512)
    grid.add_argument("--quantization", default="q4_k_m")
    grid.add_argument("--strict-json", action=argparse.BooleanOptionalAction, default=True)
    grid.add_argument("--max-retries", type=int, default=2)
    grid.add_argument("--prompt-mode", choices=["completion", "chat", "auto"], default="auto")
    grid.add_argument("--grammar-mode", choices=["auto", "word_list", "idea_list"], default="auto")
    grid.add_argument("--stop", default="")
    grid.add_argument("--restlessness-k", type=int, default=3)
    grid.add_argument("--best-of-k", type=int, default=4)
    grid.add_argument("--adaptive-stop-delta", type=float, default=0.0)
    grid.add_argument("--adaptive-min-iters", type=int, default=1)
    grid.add_argument("--trigger-objective", type=float, default=0.0)
    grid.add_argument("--token-budget-per-prompt", type=int, default=0)
    grid.add_argument("--compute-tag", default="")
    grid.add_argument("--stage", choices=["micro", "main", "confirm", "aux", "phase4"], default="main")
    grid.add_argument("--token-budget-cap", type=int, default=0)
    grid.add_argument("--dat-repeats", type=int, default=1)
    grid.add_argument("--cue-offset", type=int, default=0)
    grid.add_argument("--aut-offset", type=int, default=0)
    grid.add_argument("--limit-cues", type=int, default=0)
    grid.add_argument("--limit-aut", type=int, default=0)
    grid.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    grid.add_argument("--progress-every", type=int, default=10)
    grid.add_argument("--health-window", type=int, default=20)
    grid.add_argument("--health-min-json", type=float, default=0.95)
    grid.add_argument("--health-min-valid", type=float, default=0.90)
    grid.add_argument("--health-min-samples", type=int, default=20)
    grid.add_argument("--health-action", choices=["quarantine_cell", "stop_run", "retry_once"], default="quarantine_cell")
    grid.add_argument("--health-events", default=None)
    grid.add_argument("--session-id", default="")
    grid.add_argument("--output-dir", default="outputs/runs")
    grid.add_argument("--append-runs", action=argparse.BooleanOptionalAction, default=False)
    grid.set_defaults(func=cmd_generate_grid)

    score = sub.add_parser("score", help="Score generated runs")
    score.add_argument("--input", required=True, help="runs.jsonl or JSON run file")
    score.add_argument("--output-dir", default="outputs/scores")
    score.add_argument("--require-single-session", action=argparse.BooleanOptionalAction, default=True)
    score.add_argument("--append-scores", action=argparse.BooleanOptionalAction, default=False)
    score.set_defaults(func=cmd_score)

    frontier = sub.add_parser("analyze-frontier", help="Analyze novelty-appropriateness frontier")
    frontier.add_argument("--runs", required=True, help="scores.jsonl")
    frontier.add_argument("--token-budget", type=int, default=800)
    frontier.add_argument("--compute-matched-k", type=int, default=4)
    frontier.add_argument("--paired-by", choices=["prompt", "task"], default="prompt")
    frontier.add_argument("--compute-matched-by", choices=["prompt", "task"], default="prompt")
    frontier.add_argument("--compute-matched-token-tolerance", type=float, default=0.25)
    frontier.add_argument("--exclude-invalid", action=argparse.BooleanOptionalAction, default=True)
    frontier.add_argument("--require-single-session", action=argparse.BooleanOptionalAction, default=True)
    frontier.add_argument("--output-dir", default="outputs/analysis")
    frontier.set_defaults(func=cmd_analyze_frontier)

    homo = sub.add_parser("audit-homogeneity", help="Run population-level homogeneity audit")
    homo.add_argument("--runs", required=True, help="runs.jsonl")
    homo.add_argument("--by-task", action="store_true")
    homo.add_argument("--output-dir", default="outputs/analysis")
    homo.set_defaults(func=cmd_audit_homogeneity)

    samplers = sub.add_parser("analyze-samplers", help="Analyze decoding sampler profiles")
    samplers.add_argument("--scores", required=True, help="scores.jsonl")
    samplers.add_argument("--baseline-profile", default="default_nucleus")
    samplers.add_argument("--exclude-invalid", action=argparse.BooleanOptionalAction, default=True)
    samplers.add_argument("--require-single-session", action=argparse.BooleanOptionalAction, default=True)
    samplers.add_argument("--output-dir", default="outputs/analysis")
    samplers.set_defaults(func=cmd_analyze_samplers)

    hslice = sub.add_parser("prepare-human-slice", help="Create stratified sample for human rating")
    hslice.add_argument("--scores", required=True)
    hslice.add_argument("--n", type=int, default=180)
    hslice.add_argument("--seed", type=int, default=19)
    hslice.add_argument("--output", default="outputs/calibration/human_slice.json")
    hslice.set_defaults(func=cmd_prepare_human_slice)

    heval = sub.add_parser("eval-human", help="Evaluate human calibration file")
    heval.add_argument("--ratings", required=True)
    heval.add_argument("--target-corr", type=float, default=0.45)
    heval.add_argument("--output", default="outputs/calibration/calibration_report.json")
    heval.set_defaults(func=cmd_eval_human)

    cbe = sub.add_parser("compare-backends", help="Compare MPS vs CUDA trend stability")
    cbe.add_argument("--local-frontier", required=True)
    cbe.add_argument("--cuda-frontier", required=True)
    cbe.add_argument("--output", default="outputs/analysis/backend_parity.json")
    cbe.set_defaults(func=cmd_compare_backends)

    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
