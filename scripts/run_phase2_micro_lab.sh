#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUNS_DIR="${1:-outputs/phase2_micro_lab/runs}"
SCORES_DIR="${2:-outputs/phase2_micro_lab/scores}"
ANALYSIS_DIR="${3:-outputs/phase2_micro_lab/analysis}"
PREFLIGHT_DIR="${4:-outputs/phase2_micro_lab/preflight}"

PREFLIGHT_RUNS_DIR="${PREFLIGHT_RUNS_DIR:-$PREFLIGHT_DIR/runs}"
PREFLIGHT_SUMMARY="${PREFLIGHT_SUMMARY:-$PREFLIGHT_DIR/preflight_summary.json}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-false}"

export BACKEND="${BACKEND:-llama_cpp}"
export MODEL_PATH_MAP="${MODEL_PATH_MAP:-model_paths.downloaded.json}"
export TASKS="${TASKS:-cdat,aut}"
export METHODS="${METHODS:-one_shot,best_of_k_one_shot,restlessness_best,restlessness_last_iter}"
export CANDIDATE_MODELS="${CANDIDATE_MODELS:-gemma-2b,gemma-2b-it,qwen2.5-3b,qwen2.5-3b-instruct}"
export FALLBACK_MODELS="${FALLBACK_MODELS:-gemma-2b,qwen2.5-3b,qwen2.5-3b-instruct}"
export TEMPERATURES="${TEMPERATURES:-0.7}"
export SEEDS="${SEEDS:-11,37}"
export TOP_P="${TOP_P:-0.9}"
export MAX_TOKENS="${MAX_TOKENS:-256}"
export QUANTIZATION="${QUANTIZATION:-q4_k_m}"
export STRICT_JSON="${STRICT_JSON:-true}"
export MAX_RETRIES="${MAX_RETRIES:-4}"
export PROMPT_MODE="${PROMPT_MODE:-auto}"
export GRAMMAR_MODE="${GRAMMAR_MODE:-auto}"
export STOP="${STOP:-}"
export RESTLESSNESS_K="${RESTLESSNESS_K:-3}"
export BEST_OF_K="${BEST_OF_K:-4}"
export LIMIT_CUES="${LIMIT_CUES:-12}"
export LIMIT_AUT="${LIMIT_AUT:-12}"
export N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
export N_CTX="${N_CTX:-24576}"
export N_THREADS="${N_THREADS:-0}"
export N_BATCH="${N_BATCH:-3072}"
export N_UBATCH="${N_UBATCH:-1536}"
export N_THREADS_BATCH="${N_THREADS_BATCH:-0}"
export SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-1}"
export APPEND_RUNS="${APPEND_RUNS:-false}"
export APPEND_SCORES="${APPEND_SCORES:-false}"
export HEALTH_WINDOW="${HEALTH_WINDOW:-20}"
export HEALTH_MIN_JSON="${HEALTH_MIN_JSON:-0.95}"
export HEALTH_MIN_VALID="${HEALTH_MIN_VALID:-0.90}"
export HEALTH_MIN_SAMPLES="${HEALTH_MIN_SAMPLES:-20}"
export HEALTH_ACTION="${HEALTH_ACTION:-quarantine_cell}"
export HEALTH_EVENTS="${HEALTH_EVENTS:-$RUNS_DIR/health_events.jsonl}"

selected_models="$FALLBACK_MODELS"
if [[ "$SKIP_PREFLIGHT" != "true" ]]; then
  MODELS="$CANDIDATE_MODELS" "$(dirname "$0")/run_phase2_preflight.sh" "$PREFLIGHT_RUNS_DIR" "$PREFLIGHT_SUMMARY"
  selected_models="$("$PYTHON_BIN" - "$PREFLIGHT_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
models = [m for m in payload.get("passed_models", []) if isinstance(m, str) and m.strip()]
print(",".join(models))
PY
)"
fi

if [[ -z "$selected_models" ]]; then
  echo "No models passed preflight. Check: $PREFLIGHT_SUMMARY" >&2
  exit 1
fi

export MODELS="$selected_models"
"$(dirname "$0")/run_phase1_local.sh" "$RUNS_DIR" "$SCORES_DIR" "$ANALYSIS_DIR"

cat <<EOF
Phase 2 micro-lab complete.
Selected models: $MODELS
Runs: $RUNS_DIR
Scores: $SCORES_DIR
Analysis: $ANALYSIS_DIR
Preflight summary: $PREFLIGHT_SUMMARY
EOF
