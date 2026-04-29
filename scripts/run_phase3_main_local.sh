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

RUNS_DIR="${1:-outputs/phase3_v3/runs}"
SCORES_DIR="${2:-outputs/phase3_v3/scores}"
ANALYSIS_DIR="${3:-outputs/phase3_v3/analysis}"

BACKEND="${BACKEND:-llama_cpp}"
MODEL_PATH_MAP="${MODEL_PATH_MAP:-model_paths.downloaded.json}"
CORE_MODELS="${CORE_MODELS-gemma-2b,gemma-2b-it,qwen2.5-3b,qwen2.5-3b-instruct}"
CONFIRM_MODELS="${CONFIRM_MODELS-mistral-7b-v0.3,mistral-7b-instruct-v0.3}"

MICRO_METHODS="${MICRO_METHODS:-one_shot,best_of_k_one_shot,restlessness_best,restlessness_adaptive}"
MAIN_METHODS="${MAIN_METHODS:-one_shot,best_of_k_one_shot,restlessness_best}"
AUX_METHODS="${AUX_METHODS:-one_shot,restlessness_best}"

TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-256}"
QUANTIZATION="${QUANTIZATION:-q4_k_m}"
STRICT_JSON="${STRICT_JSON:-true}"
MAX_RETRIES="${MAX_RETRIES:-4}"
PROMPT_MODE="${PROMPT_MODE:-auto}"
GRAMMAR_MODE="${GRAMMAR_MODE:-auto}"
STOP="${STOP:-}"
RESTLESSNESS_K="${RESTLESSNESS_K:-3}"
BEST_OF_K="${BEST_OF_K:-4}"
ADAPTIVE_STOP_DELTA="${ADAPTIVE_STOP_DELTA:-0.015}"
ADAPTIVE_MIN_ITERS="${ADAPTIVE_MIN_ITERS:-1}"
TOKEN_BUDGET_PER_PROMPT="${TOKEN_BUDGET_PER_PROMPT:-0}"

N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
N_CTX="${N_CTX:-16384}"
N_THREADS="${N_THREADS:-0}"
N_BATCH="${N_BATCH:-2048}"
N_UBATCH="${N_UBATCH:-1024}"
N_THREADS_BATCH="${N_THREADS_BATCH:-0}"

SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
HEALTH_WINDOW="${HEALTH_WINDOW:-20}"
HEALTH_MIN_JSON="${HEALTH_MIN_JSON:-0.95}"
HEALTH_MIN_VALID="${HEALTH_MIN_VALID:-0.90}"
HEALTH_MIN_SAMPLES="${HEALTH_MIN_SAMPLES:-20}"
HEALTH_ACTION="${HEALTH_ACTION:-quarantine_cell}"

SESSION_ID="${SESSION_ID:-phase3-$(date +%s)}"
CREATIVEAI_EMBEDDING_BACKEND="${CREATIVEAI_EMBEDDING_BACKEND:-sentence_transformer}"
CREATIVEAI_REQUIRE_SEMANTIC="${CREATIVEAI_REQUIRE_SEMANTIC:-true}"
export CREATIVEAI_EMBEDDING_BACKEND
export CREATIVEAI_REQUIRE_SEMANTIC

MICRO_JSON_MIN="${MICRO_JSON_MIN:-0.95}"
MICRO_VALID_MIN="${MICRO_VALID_MIN:-0.90}"
MICRO_ALLOW_COMPUTE_GATE_FAIL="${MICRO_ALLOW_COMPUTE_GATE_FAIL:-true}"

MICRO_SEEDS="${MICRO_SEEDS:-11,37}"
MICRO_TEMPS="${MICRO_TEMPS:-0.7}"
MAIN_SEEDS="${MAIN_SEEDS:-11,37}"
MAIN_TEMPS="${MAIN_TEMPS:-0.7,1.0}"
CONFIRM_SEEDS="${CONFIRM_SEEDS:-11,37}"
CONFIRM_TEMPS="${CONFIRM_TEMPS:-0.7}"
DAT_SEEDS="${DAT_SEEDS:-11}"
DAT_TEMPS="${DAT_TEMPS:-0.7}"

MICRO_LIMIT_CUES="${MICRO_LIMIT_CUES:-4}"
MICRO_LIMIT_AUT="${MICRO_LIMIT_AUT:-4}"
MICRO_CUE_OFFSET="${MICRO_CUE_OFFSET:-0}"
MICRO_AUT_OFFSET="${MICRO_AUT_OFFSET:-0}"
MAIN_LIMIT_CUES="${MAIN_LIMIT_CUES:-6}"
MAIN_LIMIT_AUT="${MAIN_LIMIT_AUT:-6}"
MAIN_CUE_OFFSET="${MAIN_CUE_OFFSET:-4}"
MAIN_AUT_OFFSET="${MAIN_AUT_OFFSET:-4}"
CONFIRM_LIMIT_CUES="${CONFIRM_LIMIT_CUES:-4}"
CONFIRM_LIMIT_AUT="${CONFIRM_LIMIT_AUT:-4}"
CONFIRM_CUE_OFFSET="${CONFIRM_CUE_OFFSET:-10}"
CONFIRM_AUT_OFFSET="${CONFIRM_AUT_OFFSET:-10}"
DAT_REPEATS="${DAT_REPEATS:-4}"

PHASE2_RUNS_PATH="${PHASE2_RUNS_PATH:-outputs/phase2_micro_lab_20260226_170010/runs/runs.jsonl}"
TOKEN_BUDGET_CAP_TOTAL="${TOKEN_BUDGET_CAP_TOTAL:-0}"

MICRO_ANALYSIS_DIR="$ANALYSIS_DIR/phase3a_micro"
FINAL_ANALYSIS_DIR="$ANALYSIS_DIR/phase3_full"
MICRO_GATE_SUMMARY="$ANALYSIS_DIR/phase3a_micro_gate_summary.json"

mkdir -p "$RUNS_DIR" "$SCORES_DIR" "$ANALYSIS_DIR" "$MICRO_ANALYSIS_DIR" "$FINAL_ANALYSIS_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
fmt_dur() {
  local sec="$1"
  printf '%02d:%02d:%02d' $((sec/3600)) $(((sec%3600)/60)) $((sec%60))
}
log() { echo "[$(ts)] $*"; }
run_stage() {
  local name="$1"
  shift
  local t0
  t0=$(date +%s)
  log "START ${name}"
  "$@"
  local t1
  t1=$(date +%s)
  log "DONE  ${name} (duration=$(fmt_dur $((t1-t0))))"
}

require_semantic="$(printf '%s' "${CREATIVEAI_REQUIRE_SEMANTIC}" | tr '[:upper:]' '[:lower:]')"
if [[ "$require_semantic" == "1" || "$require_semantic" == "true" || "$require_semantic" == "yes" || "$require_semantic" == "on" ]]; then
  if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
missing = [name for name in ("torch", "sentence_transformers") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required semantic embedding deps: " + ", ".join(missing))
PY
  then
    echo "Install deps first: pip install torch sentence-transformers" >&2
    exit 1
  fi
fi

if [[ "$TOKEN_BUDGET_CAP_TOTAL" == "auto" ]]; then
  if [[ -f "$PHASE2_RUNS_PATH" ]]; then
    TOKEN_BUDGET_CAP_TOTAL="$($PYTHON_BIN - "$PHASE2_RUNS_PATH" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
rows = []
with p.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

total = 0
for r in rows:
    meta = r.get('metadata', {}) if isinstance(r.get('metadata', {}), dict) else {}
    t = int(meta.get('tokens_total', 0) or r.get('tokens_total', 0) or meta.get('token_count', 0) or r.get('token_count', 0) or 0)
    total += max(0, t)
print(int(total * 1.3) if total > 0 else 0)
PY
)"
  else
    TOKEN_BUDGET_CAP_TOTAL="0"
  fi
fi

current_token_total() {
  if [[ ! -f "$RUNS_DIR/runs.jsonl" ]]; then
    echo 0
    return
  fi
  "$PYTHON_BIN" - "$RUNS_DIR/runs.jsonl" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
total = 0
with p.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        meta = r.get('metadata', {}) if isinstance(r.get('metadata', {}), dict) else {}
        t = int(meta.get('tokens_total', 0) or r.get('tokens_total', 0) or meta.get('token_count', 0) or r.get('token_count', 0) or 0)
        total += max(0, t)
print(total)
PY
}

remaining_token_cap() {
  if [[ "$TOKEN_BUDGET_CAP_TOTAL" -le 0 ]]; then
    echo 0
    return
  fi
  local used
  used="$(current_token_total)"
  local rem=$((TOKEN_BUDGET_CAP_TOTAL - used))
  if [[ "$rem" -lt 0 ]]; then
    rem=0
  fi
  echo "$rem"
}

run_generate_stage() {
  local stage="$1"
  local models="$2"
  local methods="$3"
  local tasks="$4"
  local seeds="$5"
  local temps="$6"
  local limit_cues="$7"
  local limit_aut="$8"
  local cue_offset="$9"
  local aut_offset="${10}"
  local dat_repeats="${11}"
  local append_runs="${12}"

  local health_events="$RUNS_DIR/health_events_${stage}.jsonl"
  local rem_cap
  rem_cap="$(remaining_token_cap)"
  if [[ "$TOKEN_BUDGET_CAP_TOTAL" -gt 0 && "$rem_cap" -le 0 ]]; then
    log "TOKEN BUDGET EXHAUSTED before stage=$stage (cap=$TOKEN_BUDGET_CAP_TOTAL)"
    return 2
  fi

  local cmd=(
    "${CLI_CMD[@]}" generate-grid
    --backend "$BACKEND"
    --model-path-map "$MODEL_PATH_MAP"
    --tasks "$tasks"
    --methods "$methods"
    --models "$models"
    --temperatures "$temps"
    --seeds "$seeds"
    --top-p "$TOP_P"
    --max-tokens "$MAX_TOKENS"
    --quantization "$QUANTIZATION"
    --max-retries "$MAX_RETRIES"
    --prompt-mode "$PROMPT_MODE"
    --grammar-mode "$GRAMMAR_MODE"
    --stop "$STOP"
    --restlessness-k "$RESTLESSNESS_K"
    --best-of-k "$BEST_OF_K"
    --adaptive-stop-delta "$ADAPTIVE_STOP_DELTA"
    --adaptive-min-iters "$ADAPTIVE_MIN_ITERS"
    --token-budget-per-prompt "$TOKEN_BUDGET_PER_PROMPT"
    --compute-tag "$stage"
    --stage "$stage"
    --limit-cues "$limit_cues"
    --limit-aut "$limit_aut"
    --cue-offset "$cue_offset"
    --aut-offset "$aut_offset"
    --dat-repeats "$dat_repeats"
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
    --health-events "$health_events"
    --session-id "$SESSION_ID"
    --output-dir "$RUNS_DIR"
  )

  if [[ "$STRICT_JSON" == "true" ]]; then
    cmd+=(--strict-json)
  else
    cmd+=(--no-strict-json)
  fi
  if [[ "$SHOW_PROGRESS" == "true" ]]; then
    cmd+=(--progress)
  else
    cmd+=(--no-progress)
  fi
  if [[ "$append_runs" == "true" ]]; then
    cmd+=(--append-runs)
  else
    cmd+=(--no-append-runs)
  fi
  if [[ "$TOKEN_BUDGET_CAP_TOTAL" -gt 0 ]]; then
    cmd+=(--token-budget-cap "$rem_cap")
  fi

  run_stage "generate-grid-${stage}" "${cmd[@]}"
}

run_generate_stage "micro" "$CORE_MODELS" "$MICRO_METHODS" "cdat,aut" "$MICRO_SEEDS" "$MICRO_TEMPS" "$MICRO_LIMIT_CUES" "$MICRO_LIMIT_AUT" "$MICRO_CUE_OFFSET" "$MICRO_AUT_OFFSET" "1" "false"

run_stage "score-micro" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR/micro" --require-single-session --no-append-scores
run_stage "analyze-micro" "${CLI_CMD[@]}" analyze-frontier --runs "$SCORES_DIR/micro/scores.jsonl" --require-single-session --exclude-invalid --paired-by prompt --compute-matched-by prompt --compute-matched-k "$BEST_OF_K" --compute-matched-token-tolerance 0.25 --output-dir "$MICRO_ANALYSIS_DIR"

run_stage "micro-gate" "$PYTHON_BIN" - "$RUNS_DIR/runs.jsonl" "$MICRO_ANALYSIS_DIR/frontier_analysis.json" "$MICRO_GATE_SUMMARY" "$MICRO_JSON_MIN" "$MICRO_VALID_MIN" "$MICRO_ALLOW_COMPUTE_GATE_FAIL" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

runs_path = Path(sys.argv[1])
frontier_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
json_min = float(sys.argv[4])
valid_min = float(sys.argv[5])
allow_compute_fail = str(sys.argv[6]).strip().lower() in {"1", "true", "yes", "on"}

rows = []
with runs_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

micro = [r for r in rows if str(r.get("phase3_stage", r.get("metadata", {}).get("phase3_stage", ""))) == "micro"]
by_cell = defaultdict(list)
for r in micro:
    by_cell[(str(r.get("model_id", "")), str(r.get("task_id", "")))].append(r)

cell_rows = []
all_quality_pass = True
for (model_id, task_id), bucket in sorted(by_cell.items()):
    n = len(bucket)
    json_valid = 0
    valid = 0
    for r in bucket:
        flags = r.get("validity_flags", {}) if isinstance(r.get("validity_flags", {}), dict) else {}
        is_json = bool(r.get("json_valid", flags.get("json_valid", False)))
        is_valid = bool(flags.get("valid", False))
        json_valid += 1 if is_json else 0
        valid += 1 if is_valid else 0
    json_rate = (json_valid / n) if n else 0.0
    valid_rate = (valid / n) if n else 0.0
    passed = n > 0 and json_rate >= json_min and valid_rate >= valid_min
    all_quality_pass = all_quality_pass and passed
    cell_rows.append({
        "model_id": model_id,
        "task_id": task_id,
        "n": n,
        "json_valid_rate": json_rate,
        "valid_rate": valid_rate,
        "pass": passed,
    })

frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
primary = frontier.get("compute_matched", {}).get("primary_comparison", {})
task_summary = primary.get("task_summary", [])
ci_pass_tasks = [
    t for t in task_summary
    if str(t.get("task_id", "")) in {"cdat", "aut"}
    and int(t.get("n", 0)) > 0
    and float(t.get("ci_low", 0.0)) > 0.0
]
compute_pass = len(ci_pass_tasks) >= 1

payload = {
    "status": "ok" if (all_quality_pass and compute_pass) else "fail",
    "quality_thresholds": {"json_min": json_min, "valid_min": valid_min},
    "quality_by_model_task": cell_rows,
    "compute_gate": {
        "n_pairs": int(primary.get("n_pairs", 0)),
        "mean_delta": float(primary.get("mean_delta", 0.0)),
        "ci_low": float(primary.get("ci_low", 0.0)),
        "ci_high": float(primary.get("ci_high", 0.0)),
        "ci_pass_task_count": len(ci_pass_tasks),
        "ci_pass_tasks": ci_pass_tasks,
    },
    "pass": bool(all_quality_pass and compute_pass),
}

# model selection: require all task rows for that model to pass quality gate
model_ok = {}
for row in cell_rows:
    model_ok.setdefault(row["model_id"], True)
    model_ok[row["model_id"]] = model_ok[row["model_id"]] and bool(row["pass"])
passed_models = sorted([m for m, ok in model_ok.items() if ok])
failed_models = sorted([m for m, ok in model_ok.items() if not ok])
payload["passed_models"] = passed_models
payload["failed_models"] = failed_models

if not allow_compute_fail:
    payload["pass"] = bool(payload["pass"])
else:
    payload["pass"] = bool(all_quality_pass and len(passed_models) > 0)
    payload["compute_gate"]["warning_only"] = True
payload["status"] = "ok" if payload["pass"] else "fail"

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(out_path)
if not payload["pass"]:
    raise SystemExit(3)
PY

ACTIVE_CORE_MODELS="$($PYTHON_BIN - "$MICRO_GATE_SUMMARY" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
obj=json.loads(p.read_text())
print(",".join(obj.get("passed_models", [])))
PY
)"
if [[ -z "$ACTIVE_CORE_MODELS" ]]; then
  echo "No core models passed micro quality gate. See: $MICRO_GATE_SUMMARY" >&2
  exit 1
fi
log "MICRO GATE selected core models: $ACTIVE_CORE_MODELS"

run_generate_stage "main" "$ACTIVE_CORE_MODELS" "$MAIN_METHODS" "cdat,aut" "$MAIN_SEEDS" "$MAIN_TEMPS" "$MAIN_LIMIT_CUES" "$MAIN_LIMIT_AUT" "$MAIN_CUE_OFFSET" "$MAIN_AUT_OFFSET" "1" "true"
if [[ -n "$CONFIRM_MODELS" ]]; then
  run_generate_stage "confirm" "$CONFIRM_MODELS" "$MAIN_METHODS" "cdat,aut" "$CONFIRM_SEEDS" "$CONFIRM_TEMPS" "$CONFIRM_LIMIT_CUES" "$CONFIRM_LIMIT_AUT" "$CONFIRM_CUE_OFFSET" "$CONFIRM_AUT_OFFSET" "1" "true"
fi
run_generate_stage "aux" "$ACTIVE_CORE_MODELS" "$AUX_METHODS" "dat" "$DAT_SEEDS" "$DAT_TEMPS" "0" "0" "0" "0" "$DAT_REPEATS" "true"

run_stage "score-full" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR/full" --require-single-session --no-append-scores
run_stage "analyze-full" "${CLI_CMD[@]}" analyze-frontier --runs "$SCORES_DIR/full/scores.jsonl" --require-single-session --exclude-invalid --paired-by prompt --compute-matched-by prompt --compute-matched-k "$BEST_OF_K" --compute-matched-token-tolerance 0.25 --output-dir "$FINAL_ANALYSIS_DIR"
run_stage "audit-homogeneity-full" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --output-dir "$FINAL_ANALYSIS_DIR"
run_stage "audit-homogeneity-by-task" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --by-task --output-dir "$FINAL_ANALYSIS_DIR/task_stratified"

run_stage "phase3-count-summary" "$PYTHON_BIN" - "$RUNS_DIR/runs.jsonl" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

rows = []
with Path(sys.argv[1]).open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

stage_counts = Counter(str(r.get('phase3_stage', r.get('metadata', {}).get('phase3_stage', ''))) for r in rows)
print(json.dumps({
    'total_runs': len(rows),
    'stage_counts': dict(stage_counts),
    'expected_total_runs': 960,
}, indent=2))
PY

cat <<EOF2
Phase 3 v3 local pipeline complete.
Session: $SESSION_ID
Runs: $RUNS_DIR
Scores: $SCORES_DIR/full
Micro gate: $MICRO_GATE_SUMMARY
Final analysis: $FINAL_ANALYSIS_DIR
EOF2
