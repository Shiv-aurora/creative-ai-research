#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
CLI_CMD=("$PYTHON_BIN" -m creativeai.cli)
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

RUNS_DIR="${1:-outputs/phase2_preflight/runs}"
SUMMARY_PATH="${2:-outputs/phase2_preflight/preflight_summary.json}"

BACKEND="${BACKEND:-llama_cpp}"
MODEL_PATH_MAP="${MODEL_PATH_MAP:-model_paths.downloaded.json}"
MODELS="${MODELS:-gemma-2b,gemma-2b-it,qwen2.5-3b,qwen2.5-3b-instruct}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-256}"
QUANTIZATION="${QUANTIZATION:-q4_k_m}"
STRICT_JSON="${STRICT_JSON:-true}"
MAX_RETRIES="${MAX_RETRIES:-4}"
PROMPT_MODE="${PROMPT_MODE:-auto}"
GRAMMAR_MODE="${GRAMMAR_MODE:-auto}"
RESTLESSNESS_K="${RESTLESSNESS_K:-3}"
BEST_OF_K="${BEST_OF_K:-4}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
N_CTX="${N_CTX:-24576}"
N_THREADS="${N_THREADS:-0}"
N_BATCH="${N_BATCH:-3072}"
N_UBATCH="${N_UBATCH:-1536}"
N_THREADS_BATCH="${N_THREADS_BATCH:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
HEALTH_WINDOW="${HEALTH_WINDOW:-20}"
HEALTH_MIN_JSON="${HEALTH_MIN_JSON:-0.95}"
HEALTH_MIN_VALID="${HEALTH_MIN_VALID:-0.90}"
HEALTH_MIN_SAMPLES="${HEALTH_MIN_SAMPLES:-20}"
HEALTH_ACTION="${HEALTH_ACTION:-quarantine_cell}"
HEALTH_EVENTS="${HEALTH_EVENTS:-$RUNS_DIR/health_events.jsonl}"

PREFLIGHT_SEEDS="${PREFLIGHT_SEEDS:-11}"
PREFLIGHT_TEMPERATURES="${PREFLIGHT_TEMPERATURES:-0.7}"
PREFLIGHT_LIMIT_CUES="${PREFLIGHT_LIMIT_CUES:-6}"
PREFLIGHT_LIMIT_AUT="${PREFLIGHT_LIMIT_AUT:-6}"
PREFLIGHT_JSON_MIN="${PREFLIGHT_JSON_MIN:-0.95}"
PREFLIGHT_VALID_MIN="${PREFLIGHT_VALID_MIN:-0.90}"

mkdir -p "$RUNS_DIR" "$(dirname "$SUMMARY_PATH")"

GEN_CMD=(
  "${CLI_CMD[@]}" generate-grid
  --backend "$BACKEND"
  --tasks "cdat,aut"
  --methods "one_shot"
  --models "$MODELS"
  --temperatures "$PREFLIGHT_TEMPERATURES"
  --seeds "$PREFLIGHT_SEEDS"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
  --quantization "$QUANTIZATION"
  --max-retries "$MAX_RETRIES"
  --prompt-mode "$PROMPT_MODE"
  --grammar-mode "$GRAMMAR_MODE"
  --restlessness-k "$RESTLESSNESS_K"
  --best-of-k "$BEST_OF_K"
  --limit-cues "$PREFLIGHT_LIMIT_CUES"
  --limit-aut "$PREFLIGHT_LIMIT_AUT"
  --n-gpu-layers "$N_GPU_LAYERS"
  --n-ctx "$N_CTX"
  --n-threads "$N_THREADS"
  --n-batch "$N_BATCH"
  --n-ubatch "$N_UBATCH"
  --n-threads-batch "$N_THREADS_BATCH"
  --progress-every "$PROGRESS_EVERY"
  --health-window "$HEALTH_WINDOW"
  --health-min-json "$HEALTH_MIN_JSON"
  --health-min-valid "$HEALTH_MIN_VALID"
  --health-min-samples "$HEALTH_MIN_SAMPLES"
  --health-action "$HEALTH_ACTION"
  --health-events "$HEALTH_EVENTS"
  --model-path-map "$MODEL_PATH_MAP"
  --output-dir "$RUNS_DIR"
  --no-append-runs
)
if [[ "$STRICT_JSON" == "true" ]]; then
  GEN_CMD+=(--strict-json)
else
  GEN_CMD+=(--no-strict-json)
fi
if [[ "$SHOW_PROGRESS" == "true" ]]; then
  GEN_CMD+=(--progress)
else
  GEN_CMD+=(--no-progress)
fi

"${GEN_CMD[@]}"

"$PYTHON_BIN" - "$RUNS_DIR/runs.jsonl" "$SUMMARY_PATH" "$PREFLIGHT_JSON_MIN" "$PREFLIGHT_VALID_MIN" <<'PY'
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

runs_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
json_min = float(sys.argv[3])
valid_min = float(sys.argv[4])

if not runs_path.exists():
    raise SystemExit(f"missing runs file: {runs_path}")

rows = []
with runs_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

by_model: dict[str, list[dict]] = defaultdict(list)
for row in rows:
    by_model[str(row.get("model_id", "unknown-model"))].append(row)

summary_models = []
passed_models = []
failed_models = []
for model_id, model_rows in sorted(by_model.items()):
    total = len(model_rows)
    json_valid = 0
    valid = 0
    for row in model_rows:
        flags = row.get("validity_flags", {})
        is_json = bool(row.get("json_valid", flags.get("json_valid", False)))
        is_valid = bool(flags.get("valid", False))
        if is_json:
            json_valid += 1
        if is_valid:
            valid += 1
    json_rate = (json_valid / total) if total else 0.0
    valid_rate = (valid / total) if total else 0.0
    passed = total > 0 and json_rate >= json_min and valid_rate >= valid_min
    reasons = []
    if total == 0:
        reasons.append("no_samples")
    if json_rate < json_min:
        reasons.append(f"json_valid_below_{json_min:.2f}")
    if valid_rate < valid_min:
        reasons.append(f"valid_below_{valid_min:.2f}")

    item = {
        "model_id": model_id,
        "samples": total,
        "json_valid_rate": json_rate,
        "valid_rate": valid_rate,
        "pass": passed,
        "reason": ",".join(reasons) if reasons else "ok",
    }
    summary_models.append(item)
    if passed:
        passed_models.append(model_id)
    else:
        failed_models.append(model_id)

payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "runs_path": str(runs_path),
    "thresholds": {"json_valid_min": json_min, "valid_min": valid_min},
    "models": summary_models,
    "passed_models": passed_models,
    "failed_models": failed_models,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=True)
print(str(summary_path))
PY

cat <<EOF
Phase 2 preflight complete.
Runs: $RUNS_DIR
Summary: $SUMMARY_PATH
EOF
