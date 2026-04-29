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

RUNS_DIR="${1:-outputs/runs}"
SCORES_DIR="${2:-outputs/scores}"
ANALYSIS_DIR="${3:-outputs/analysis}"
BACKEND="${BACKEND:-mock}"
MODEL_PATH_MAP="${MODEL_PATH_MAP:-}"
TASKS="${TASKS:-dat,cdat,aut}"
METHODS="${METHODS:-one_shot,best_of_k_one_shot,restlessness_best}"
MODELS="${MODELS:-gemma-2-2b,gemma-2-2b-it,qwen2.5-3b,qwen2.5-3b-instruct,mistral-7b-v0.3,mistral-7b-instruct-v0.3}"
TEMPERATURES="${TEMPERATURES:-0.2,0.7,1.0,1.3}"
SEEDS="${SEEDS:-11,37,73,101,149}"
MAX_TOKENS="${MAX_TOKENS:-512}"
LIMIT_CUES="${LIMIT_CUES:-120}"
LIMIT_AUT="${LIMIT_AUT:-90}"
RESTLESSNESS_K="${RESTLESSNESS_K:-3}"
BEST_OF_K="${BEST_OF_K:-4}"
TOP_P="${TOP_P:-0.9}"
QUANTIZATION="${QUANTIZATION:-q4_k_m}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
STRICT_JSON="${STRICT_JSON:-true}"
MAX_RETRIES="${MAX_RETRIES:-4}"
PROMPT_MODE="${PROMPT_MODE:-auto}"
GRAMMAR_MODE="${GRAMMAR_MODE:-auto}"
STOP="${STOP:-}"
N_CTX="${N_CTX:-16384}"
N_THREADS="${N_THREADS:-0}"
N_BATCH="${N_BATCH:-2048}"
N_UBATCH="${N_UBATCH:-1024}"
N_THREADS_BATCH="${N_THREADS_BATCH:-0}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
HEALTH_WINDOW="${HEALTH_WINDOW:-20}"
HEALTH_MIN_JSON="${HEALTH_MIN_JSON:-0.95}"
HEALTH_MIN_VALID="${HEALTH_MIN_VALID:-0.90}"
HEALTH_MIN_SAMPLES="${HEALTH_MIN_SAMPLES:-20}"
HEALTH_ACTION="${HEALTH_ACTION:-quarantine_cell}"
HEALTH_EVENTS="${HEALTH_EVENTS:-$RUNS_DIR/health_events.jsonl}"
APPEND_RUNS="${APPEND_RUNS:-false}"
APPEND_SCORES="${APPEND_SCORES:-false}"
SESSION_ID="${SESSION_ID:-}"
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

GEN_CMD=("${CLI_CMD[@]}" generate-grid \
  --backend "$BACKEND" \
  --tasks "$TASKS" \
  --methods "$METHODS" \
  --models "$MODELS" \
  --temperatures "$TEMPERATURES" \
  --seeds "$SEEDS" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --quantization "$QUANTIZATION" \
  --max-retries "$MAX_RETRIES" \
  --prompt-mode "$PROMPT_MODE" \
  --grammar-mode "$GRAMMAR_MODE" \
  --stop "$STOP" \
  --restlessness-k "$RESTLESSNESS_K" \
  --best-of-k "$BEST_OF_K" \
  --limit-cues "$LIMIT_CUES" \
  --limit-aut "$LIMIT_AUT" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --n-ctx "$N_CTX" \
  --n-threads "$N_THREADS" \
  --n-batch "$N_BATCH" \
  --n-ubatch "$N_UBATCH" \
  --n-threads-batch "$N_THREADS_BATCH" \
  --progress-every "$PROGRESS_EVERY" \
  --health-window "$HEALTH_WINDOW" \
  --health-min-json "$HEALTH_MIN_JSON" \
  --health-min-valid "$HEALTH_MIN_VALID" \
  --health-min-samples "$HEALTH_MIN_SAMPLES" \
  --health-action "$HEALTH_ACTION" \
  --health-events "$HEALTH_EVENTS" \
  --output-dir "$RUNS_DIR")

if [[ -n "$MODEL_PATH_MAP" ]]; then
  GEN_CMD+=(--model-path-map "$MODEL_PATH_MAP")
fi
if [[ -n "$SESSION_ID" ]]; then
  GEN_CMD+=(--session-id "$SESSION_ID")
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

run_stage "generate-grid" "${GEN_CMD[@]}"

if [[ "$APPEND_SCORES" == "true" ]]; then
  run_stage "score" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --append-scores
else
  run_stage "score" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --no-append-scores
fi
run_stage "analyze-frontier" "${CLI_CMD[@]}" analyze-frontier --runs "$SCORES_DIR/scores.jsonl" --require-single-session --output-dir "$ANALYSIS_DIR"
run_stage "audit-homogeneity" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --output-dir "$ANALYSIS_DIR"

cat <<EOF
Phase 1 local pipeline complete.
Runs: $RUNS_DIR
Scores: $SCORES_DIR
Analysis: $ANALYSIS_DIR
EOF
