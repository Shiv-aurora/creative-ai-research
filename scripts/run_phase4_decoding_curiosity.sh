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

RUNS_DIR="${1:-outputs/phase4_decoding_curiosity/runs}"
SCORES_DIR="${2:-outputs/phase4_decoding_curiosity/scores}"
ANALYSIS_DIR="${3:-outputs/phase4_decoding_curiosity/analysis}"

BACKEND="${BACKEND:-llama_cpp}"
MODEL_PATH_MAP="${MODEL_PATH_MAP:-model_paths.downloaded.json}"
TASKS="${TASKS:-cdat,aut}"
METHODS="${METHODS:-one_shot}"
MODELS="${MODELS:-qwen2.5-3b-instruct,qwen2.5-3b}"
TEMPERATURES="${TEMPERATURES:-0.7}"
SAMPLER_PROFILES="${SAMPLER_PROFILES:-low_temp,default_nucleus,high_temp,spread_topk_minp,anti_repetition,mirostat}"
BASELINE_SAMPLER_PROFILE="${BASELINE_SAMPLER_PROFILE:-default_nucleus}"
SEEDS="${SEEDS:-11,37,73}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-0}"
MIN_P="${MIN_P:-0}"
TYPICAL_P="${TYPICAL_P:-0}"
REPEAT_PENALTY="${REPEAT_PENALTY:-1.0}"
FREQUENCY_PENALTY="${FREQUENCY_PENALTY:-0.0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
MIROSTAT_MODE="${MIROSTAT_MODE:-0}"
MIROSTAT_TAU="${MIROSTAT_TAU:-0}"
MIROSTAT_ETA="${MIROSTAT_ETA:-0}"
MAX_TOKENS="${MAX_TOKENS:-224}"
TOKEN_BUDGET_PER_PROMPT="${TOKEN_BUDGET_PER_PROMPT:-224}"
QUANTIZATION="${QUANTIZATION:-q4_k_m}"
STRICT_JSON="${STRICT_JSON:-true}"
MAX_RETRIES="${MAX_RETRIES:-4}"
PROMPT_MODE="${PROMPT_MODE:-auto}"
GRAMMAR_MODE="${GRAMMAR_MODE:-auto}"
STOP="${STOP:-}"
LIMIT_CUES="${LIMIT_CUES:-8}"
LIMIT_AUT="${LIMIT_AUT:-8}"
CUE_OFFSET="${CUE_OFFSET:-0}"
AUT_OFFSET="${AUT_OFFSET:-0}"
DAT_REPEATS="${DAT_REPEATS:-1}"
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
SESSION_ID="${SESSION_ID:-phase4-decoding-$(date +%s)}"
APPEND_RUNS="${APPEND_RUNS:-false}"
APPEND_SCORES="${APPEND_SCORES:-false}"

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
  --sampler-profiles "$SAMPLER_PROFILES"
  --seeds "$SEEDS"
  --top-p "$TOP_P"
  --top-k "$TOP_K"
  --min-p "$MIN_P"
  --typical-p "$TYPICAL_P"
  --repeat-penalty "$REPEAT_PENALTY"
  --frequency-penalty "$FREQUENCY_PENALTY"
  --presence-penalty "$PRESENCE_PENALTY"
  --mirostat-mode "$MIROSTAT_MODE"
  --mirostat-tau "$MIROSTAT_TAU"
  --mirostat-eta "$MIROSTAT_ETA"
  --max-tokens "$MAX_TOKENS"
  --token-budget-per-prompt "$TOKEN_BUDGET_PER_PROMPT"
  --quantization "$QUANTIZATION"
  --max-retries "$MAX_RETRIES"
  --prompt-mode "$PROMPT_MODE"
  --grammar-mode "$GRAMMAR_MODE"
  --stop "$STOP"
  --compute-tag "phase4_decoding"
  --stage "phase4"
  --cue-offset "$CUE_OFFSET"
  --aut-offset "$AUT_OFFSET"
  --limit-cues "$LIMIT_CUES"
  --limit-aut "$LIMIT_AUT"
  --dat-repeats "$DAT_REPEATS"
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

run_stage "generate-grid-phase4-decoding" "${GEN_CMD[@]}"

if [[ "$APPEND_SCORES" == "true" ]]; then
  run_stage "score-phase4-decoding" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --append-scores
else
  run_stage "score-phase4-decoding" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --no-append-scores
fi

run_stage "analyze-frontier-phase4-decoding" "${CLI_CMD[@]}" analyze-frontier \
  --runs "$SCORES_DIR/scores.jsonl" \
  --require-single-session \
  --exclude-invalid \
  --paired-by prompt \
  --compute-matched-by prompt \
  --compute-matched-k 1 \
  --compute-matched-token-tolerance 0.20 \
  --output-dir "$ANALYSIS_DIR"

run_stage "analyze-samplers-phase4-decoding" "${CLI_CMD[@]}" analyze-samplers \
  --scores "$SCORES_DIR/scores.jsonl" \
  --baseline-profile "$BASELINE_SAMPLER_PROFILE" \
  --require-single-session \
  --output-dir "$ANALYSIS_DIR"

run_stage "audit-homogeneity-phase4-decoding" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --output-dir "$ANALYSIS_DIR"
run_stage "audit-homogeneity-phase4-decoding-by-task" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --by-task --output-dir "$ANALYSIS_DIR/task_stratified"

cat <<EOF2
Phase 4 decoding curiosity pipeline complete.
Session: $SESSION_ID
Runs: $RUNS_DIR
Scores: $SCORES_DIR
Analysis: $ANALYSIS_DIR
Experiment manifest: $RUNS_DIR/experiment_manifest.json
EOF2
