from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from creativeai.io_utils import (
    append_jsonl,
    build_manifest,
    ensure_dir,
    snapshot_tabular,
    token_count_from_list,
    token_count_text,
    write_json,
)
from creativeai.methods import MethodRunner
from creativeai.model_backend import ModelAdapter
from creativeai.schemas import GenerationConfig, RunRecord, TaskSpec, utc_now_iso
from creativeai.validators import validate_output



def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def _compute_group_id(task_spec: TaskSpec, config: GenerationConfig, metadata: dict[str, Any]) -> str:
    task_id = task_spec.task_id
    if task_id == "cdat":
        prompt_key = f"cue={metadata.get('cue', '')}"
    elif task_id == "aut":
        prompt_key = f"object={metadata.get('object', '')}|context={metadata.get('context', '')}"
    elif task_id == "dat":
        prompt_key = f"dat_prompt_id={metadata.get('dat_prompt_id', 0)}"
    else:
        prompt_key = "default"
    compute_tag = config.compute_tag.strip() if isinstance(config.compute_tag, str) else ""
    if not compute_tag:
        compute_tag = "default"
    return (
        f"{config.model_id}|{task_id}|{prompt_key}|seed={config.seed}|temp={config.temperature}|"
        f"top_p={config.top_p}|tag={compute_tag}"
    )



def generate_run(
    task_spec: TaskSpec,
    method_runner: MethodRunner,
    model: ModelAdapter,
    config: GenerationConfig,
    output_dir: str | Path,
    session_id: str = "",
    extra_metadata: dict[str, Any] | None = None,
    phase3_stage: str = "",
) -> RunRecord:
    run_id = _new_run_id()
    result = method_runner.run(task_spec, config, model)
    validity = validate_output(task_spec.task_id, result.output)
    validity["json_valid"] = bool(result.json_valid)
    if not result.json_valid:
        validity["valid"] = False
        problems = list(validity.get("problems", []))
        if "non_json_output" not in problems:
            problems.append("non_json_output")
        validity["problems"] = sorted(set(problems))

    manifest = build_manifest(
        run_id=run_id,
        quantization=config.quantization,
        backend=config.backend,
        model_hash=getattr(model, "model_hash", "unknown"),
        cwd=Path.cwd(),
        session_id=session_id,
        extra={"task_id": task_spec.task_id, "method": method_runner.method_name},
    )

    metadata = dict(task_spec.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    effective_calls = int(getattr(result, "generation_calls", 1))
    tokens_in = int(getattr(result, "tokens_in", 0) or token_count_text(result.prompt))
    tokens_out = int(getattr(result, "tokens_out", 0) or token_count_from_list(result.output))
    tokens_total = max(0, tokens_in + tokens_out)
    compute_group_id = _compute_group_id(task_spec, config, metadata)

    metadata["generation_calls"] = effective_calls
    metadata["effective_calls"] = effective_calls
    metadata["parse_mode"] = str(getattr(result, "parse_mode", "unknown"))
    metadata["session_id"] = session_id
    metadata["tokens_in"] = tokens_in
    metadata["tokens_out"] = tokens_out
    metadata["tokens_total"] = tokens_total
    metadata["compute_group_id"] = compute_group_id
    metadata["phase3_stage"] = phase3_stage

    record = RunRecord(
        run_id=run_id,
        task_id=task_spec.task_id,
        method=method_runner.method_name,
        model_id=config.model_id,
        config=asdict(config),
        prompt=result.prompt,
        output=result.output,
        raw_trace=result.raw_trace,
        validity_flags=validity,
        timestamp_utc=utc_now_iso(),
        token_count=token_count_from_list(result.output),
        manifest=asdict(manifest),
        session_id=session_id,
        parse_mode=str(getattr(result, "parse_mode", "unknown")),
        json_valid=bool(getattr(result, "json_valid", False)),
        retry_count=int(getattr(result, "retry_count", 0)),
        candidate_objectives=[float(x) for x in getattr(result, "candidate_objectives", [])],
        metadata=metadata,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        tokens_total=tokens_total,
        effective_calls=effective_calls,
        compute_group_id=compute_group_id,
        phase3_stage=phase3_stage,
    )

    out_dir = ensure_dir(output_dir)
    run_json = out_dir / f"{run_id}.json"
    write_json(run_json, record.to_dict())
    append_jsonl(out_dir / "runs.jsonl", record.to_dict())
    return record



def save_score_records(records: list[dict[str, Any]], output_dir: str | Path, append: bool = False) -> tuple[str, str]:
    out_dir = ensure_dir(output_dir)
    jsonl_path = out_dir / "scores.jsonl"
    if jsonl_path.exists() and not append:
        jsonl_path.unlink()
    for rec in records:
        append_jsonl(jsonl_path, rec)
    snapshot_path = snapshot_tabular(records, out_dir / "scores.parquet")
    return str(jsonl_path), snapshot_path



def save_analysis_artifact(payload: dict[str, Any], output_path: str | Path) -> str:
    write_json(output_path, payload)
    return str(output_path)
