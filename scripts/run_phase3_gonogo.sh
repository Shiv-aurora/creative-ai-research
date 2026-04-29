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

RUNS_DIR="${1:-outputs/phase3_gonogo/runs}"
SCORES_DIR="${2:-outputs/phase3_gonogo/scores}"
ANALYSIS_DIR="${3:-outputs/phase3_gonogo/analysis}"

DECISION_JSON="${DECISION_JSON:-$ANALYSIS_DIR/go_nogo_decision.json}"
DECISION_MD="${DECISION_MD:-$ANALYSIS_DIR/GO_NOGO_REPORT.md}"

BACKEND="${BACKEND:-llama_cpp}"
MODEL_PATH_MAP="${MODEL_PATH_MAP:-model_paths.downloaded.json}"
TASKS="${TASKS:-cdat,aut}"
METHODS="${METHODS:-one_shot,best_of_k_one_shot,restlessness_best}"
MODELS="${MODELS:-qwen2.5-3b,qwen2.5-3b-instruct,mistral-7b-instruct-v0.3}"
TEMPERATURES="${TEMPERATURES:-0.7}"
SEEDS="${SEEDS:-11,37,73}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-224}"
TOKEN_BUDGET_PER_PROMPT="${TOKEN_BUDGET_PER_PROMPT:-224}"
QUANTIZATION="${QUANTIZATION:-q4_k_m}"
STRICT_JSON="${STRICT_JSON:-true}"
MAX_RETRIES="${MAX_RETRIES:-4}"
PROMPT_MODE="${PROMPT_MODE:-auto}"
GRAMMAR_MODE="${GRAMMAR_MODE:-auto}"
STOP="${STOP:-}"
RESTLESSNESS_K="${RESTLESSNESS_K:-3}"
BEST_OF_K="${BEST_OF_K:-4}"
LIMIT_CUES="${LIMIT_CUES:-10}"
LIMIT_AUT="${LIMIT_AUT:-10}"
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
HEALTH_EVENTS="${HEALTH_EVENTS:-$RUNS_DIR/health_events.jsonl}"
SESSION_ID="${SESSION_ID:-gonogo-$(date +%s)}"
APPEND_RUNS="${APPEND_RUNS:-false}"
APPEND_SCORES="${APPEND_SCORES:-false}"

COMPUTE_MATCHED_BY="${COMPUTE_MATCHED_BY:-prompt}"
COMPUTE_MATCHED_TOKEN_TOLERANCE="${COMPUTE_MATCHED_TOKEN_TOLERANCE:-0.30}"

GO_MIN_JSON="${GO_MIN_JSON:-0.95}"
GO_MIN_VALID="${GO_MIN_VALID:-0.90}"
GO_MIN_MATCHED_PAIRS="${GO_MIN_MATCHED_PAIRS:-24}"
GO_MIN_TASK_PAIRS="${GO_MIN_TASK_PAIRS:-8}"
GO_TASKS="${GO_TASKS:-cdat,aut}"
EXIT_ON_NO_GO="${EXIT_ON_NO_GO:-false}"

CREATIVEAI_EMBEDDING_BACKEND="${CREATIVEAI_EMBEDDING_BACKEND:-sentence_transformer}"
CREATIVEAI_REQUIRE_SEMANTIC="${CREATIVEAI_REQUIRE_SEMANTIC:-true}"
export CREATIVEAI_EMBEDDING_BACKEND
export CREATIVEAI_REQUIRE_SEMANTIC

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

mkdir -p "$RUNS_DIR" "$SCORES_DIR" "$ANALYSIS_DIR"

GEN_CMD=(
  "${CLI_CMD[@]}" generate-grid
  --backend "$BACKEND"
  --tasks "$TASKS"
  --methods "$METHODS"
  --models "$MODELS"
  --temperatures "$TEMPERATURES"
  --seeds "$SEEDS"
  --top-p "$TOP_P"
  --max-tokens "$MAX_TOKENS"
  --token-budget-per-prompt "$TOKEN_BUDGET_PER_PROMPT"
  --quantization "$QUANTIZATION"
  --max-retries "$MAX_RETRIES"
  --prompt-mode "$PROMPT_MODE"
  --grammar-mode "$GRAMMAR_MODE"
  --stop "$STOP"
  --restlessness-k "$RESTLESSNESS_K"
  --best-of-k "$BEST_OF_K"
  --compute-tag "go_nogo"
  --stage "main"
  --limit-cues "$LIMIT_CUES"
  --limit-aut "$LIMIT_AUT"
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
  --session-id "$SESSION_ID"
  --output-dir "$RUNS_DIR"
)
if [[ -n "$MODEL_PATH_MAP" ]]; then
  GEN_CMD+=(--model-path-map "$MODEL_PATH_MAP")
fi
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
if [[ "$APPEND_RUNS" == "true" ]]; then
  GEN_CMD+=(--append-runs)
else
  GEN_CMD+=(--no-append-runs)
fi

run_stage "generate-grid-go-nogo" "${GEN_CMD[@]}"

if [[ "$APPEND_SCORES" == "true" ]]; then
  run_stage "score-go-nogo" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --append-scores
else
  run_stage "score-go-nogo" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --no-append-scores
fi

run_stage "analyze-go-nogo" "${CLI_CMD[@]}" analyze-frontier \
  --runs "$SCORES_DIR/scores.jsonl" \
  --require-single-session \
  --exclude-invalid \
  --paired-by prompt \
  --compute-matched-by "$COMPUTE_MATCHED_BY" \
  --compute-matched-k "$BEST_OF_K" \
  --compute-matched-token-tolerance "$COMPUTE_MATCHED_TOKEN_TOLERANCE" \
  --output-dir "$ANALYSIS_DIR"

run_stage "audit-homogeneity-go-nogo" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --output-dir "$ANALYSIS_DIR"
run_stage "audit-homogeneity-go-nogo-by-task" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --by-task --output-dir "$ANALYSIS_DIR/task_stratified"

run_stage "go-nogo-decision" "$PYTHON_BIN" - \
  "$RUNS_DIR/runs.jsonl" \
  "$ANALYSIS_DIR/frontier_analysis.json" \
  "$DECISION_JSON" \
  "$DECISION_MD" \
  "$GO_MIN_JSON" \
  "$GO_MIN_VALID" \
  "$GO_MIN_MATCHED_PAIRS" \
  "$GO_MIN_TASK_PAIRS" \
  "$GO_TASKS" <<'PY'
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

runs_path = Path(sys.argv[1])
frontier_path = Path(sys.argv[2])
decision_json = Path(sys.argv[3])
decision_md = Path(sys.argv[4])
go_min_json = float(sys.argv[5])
go_min_valid = float(sys.argv[6])
go_min_pairs = int(sys.argv[7])
go_min_task_pairs = int(sys.argv[8])
go_tasks = [x.strip().lower() for x in str(sys.argv[9]).split(",") if x.strip()]

runs = []
with runs_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            runs.append(json.loads(line))

frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
compute = frontier.get("compute_matched", {}).get("primary_comparison", {})
task_summary = compute.get("task_summary", [])

by_cell = defaultdict(list)
for r in runs:
    model_id = str(r.get("model_id", ""))
    task_id = str(r.get("task_id", ""))
    by_cell[(model_id, task_id)].append(r)

quality_rows = []
quality_pass = True
for (model_id, task_id), bucket in sorted(by_cell.items()):
    n = len(bucket)
    json_ok = 0
    valid_ok = 0
    for r in bucket:
        flags = r.get("validity_flags", {}) if isinstance(r.get("validity_flags", {}), dict) else {}
        is_json = bool(r.get("json_valid", flags.get("json_valid", False)))
        is_valid = bool(flags.get("valid", False))
        json_ok += 1 if is_json else 0
        valid_ok += 1 if is_valid else 0
    json_rate = (json_ok / n) if n else 0.0
    valid_rate = (valid_ok / n) if n else 0.0
    passed = (n > 0) and (json_rate >= go_min_json) and (valid_rate >= go_min_valid)
    quality_pass = quality_pass and passed
    quality_rows.append(
        {
            "model_id": model_id,
            "task_id": task_id,
            "n": n,
            "json_valid_rate": json_rate,
            "valid_rate": valid_rate,
            "pass": passed,
        }
    )

n_pairs = int(compute.get("n_pairs", 0))
matched_pairs = int(compute.get("matched_pairs", n_pairs))
unmatched_pairs = int(compute.get("unmatched_pairs", 0))
total_pair_slots = matched_pairs + unmatched_pairs
match_rate = (matched_pairs / total_pair_slots) if total_pair_slots > 0 else 0.0

ci_pass_tasks = []
for row in task_summary:
    task = str(row.get("task_id", "")).lower()
    if task not in go_tasks:
        continue
    n = int(row.get("n", 0))
    ci_low = float(row.get("ci_low", 0.0))
    if n >= go_min_task_pairs and ci_low > 0.0:
        ci_pass_tasks.append(row)

pair_count_pass = n_pairs >= go_min_pairs
task_ci_pass = len(ci_pass_tasks) >= 1
go = bool(quality_pass and pair_count_pass and task_ci_pass)

reasons = []
if not quality_pass:
    reasons.append("quality_gate_failed")
if not pair_count_pass:
    reasons.append("insufficient_compute_matched_pairs")
if not task_ci_pass:
    reasons.append("no_task_family_with_positive_ci")

payload = {
    "status": "GO" if go else "NO_GO",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "thresholds": {
        "go_min_json": go_min_json,
        "go_min_valid": go_min_valid,
        "go_min_pairs": go_min_pairs,
        "go_min_task_pairs": go_min_task_pairs,
        "go_tasks": go_tasks,
    },
    "quality_gate": {
        "pass": quality_pass,
        "by_model_task": quality_rows,
    },
    "compute_gate": {
        "pass": pair_count_pass and task_ci_pass,
        "n_pairs": n_pairs,
        "mean_delta": float(compute.get("mean_delta", 0.0)),
        "ci_low": float(compute.get("ci_low", 0.0)),
        "ci_high": float(compute.get("ci_high", 0.0)),
        "wins": int(compute.get("wins", 0)),
        "matched_pairs": matched_pairs,
        "unmatched_pairs": unmatched_pairs,
        "match_rate": match_rate,
        "ci_pass_tasks": ci_pass_tasks,
        "task_summary": task_summary,
    },
    "decision": {
        "go": go,
        "reasons": reasons,
        "next_action": (
            "Scale to claim-grade run with same frozen settings and add CUDA confirm."
            if go
            else "Pivot: keep reliability+frontier claim, do not claim compute-matched superiority yet."
        ),
    },
}

decision_json.parent.mkdir(parents=True, exist_ok=True)
decision_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

md = []
md.append("# Go/No-Go Decision")
md.append("")
md.append(f"- Status: **{payload['status']}**")
md.append(f"- Generated: `{payload['generated_at_utc']}`")
md.append("")
md.append("## Compute-Matched Primary Comparison")
md.append(f"- n_pairs: `{payload['compute_gate']['n_pairs']}`")
md.append(f"- mean_delta: `{payload['compute_gate']['mean_delta']:.6f}`")
md.append(f"- 95% CI: `[{payload['compute_gate']['ci_low']:.6f}, {payload['compute_gate']['ci_high']:.6f}]`")
md.append(f"- matched/unmatched: `{matched_pairs}/{unmatched_pairs}` (match_rate=`{match_rate:.3f}`)")
md.append("")
md.append("## Quality Gate")
md.append(f"- pass: `{payload['quality_gate']['pass']}`")
for row in quality_rows:
    md.append(
        f"- {row['model_id']} | {row['task_id']} -> n={row['n']} json={row['json_valid_rate']:.3f} valid={row['valid_rate']:.3f} pass={row['pass']}"
    )
md.append("")
md.append("## Decision")
if reasons:
    md.append("- failure_reasons: " + ", ".join(reasons))
else:
    md.append("- failure_reasons: none")
md.append(f"- next_action: {payload['decision']['next_action']}")
md.append("")
md.append(f"Raw JSON: `{decision_json}`")

decision_md.write_text("\n".join(md) + "\n", encoding="utf-8")
print(str(decision_json))
print(str(decision_md))
PY

cat <<EOF
Phase 3 compute-matched go/no-go run complete.
Session: $SESSION_ID
Runs: $RUNS_DIR
Scores: $SCORES_DIR
Analysis: $ANALYSIS_DIR
Decision JSON: $DECISION_JSON
Decision report: $DECISION_MD
EOF

if [[ "$EXIT_ON_NO_GO" == "true" ]]; then
  status="$("$PYTHON_BIN" - "$DECISION_JSON" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("status", "NO_GO"))
PY
)"
  if [[ "$status" != "GO" ]]; then
    echo "Go/no-go gate returned NO_GO." >&2
    exit 4
  fi
fi
