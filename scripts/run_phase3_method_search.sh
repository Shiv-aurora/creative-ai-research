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

RUNS_DIR="${1:-outputs/phase3_method_search/runs}"
SCORES_DIR="${2:-outputs/phase3_method_search/scores}"
ANALYSIS_DIR="${3:-outputs/phase3_method_search/analysis}"

DECISION_JSON="${DECISION_JSON:-$ANALYSIS_DIR/method_search_decision.json}"
DECISION_MD="${DECISION_MD:-$ANALYSIS_DIR/METHOD_SEARCH_REPORT.md}"

BACKEND="${BACKEND:-llama_cpp}"
MODEL_PATH_MAP="${MODEL_PATH_MAP:-model_paths.downloaded.json}"
TASKS="${TASKS:-cdat,aut}"
METHODS="${METHODS:-one_shot,naive_multiturn,restlessness_best,restlessness_triggered,restlessness_adaptive}"
MODELS="${MODELS:-qwen2.5-3b,qwen2.5-3b-instruct}"
TEMPERATURES="${TEMPERATURES:-0.7}"
SEEDS="${SEEDS:-11,37}"
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
TRIGGER_OBJECTIVE="${TRIGGER_OBJECTIVE:-0.64}"
ADAPTIVE_STOP_DELTA="${ADAPTIVE_STOP_DELTA:-0.015}"
ADAPTIVE_MIN_ITERS="${ADAPTIVE_MIN_ITERS:-1}"
LIMIT_CUES="${LIMIT_CUES:-3}"
LIMIT_AUT="${LIMIT_AUT:-3}"
CUE_OFFSET="${CUE_OFFSET:-0}"
AUT_OFFSET="${AUT_OFFSET:-0}"
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
SESSION_ID="${SESSION_ID:-method-search-$(date +%s)}"
APPEND_RUNS="${APPEND_RUNS:-false}"
APPEND_SCORES="${APPEND_SCORES:-false}"

COMPUTE_MATCHED_BY="${COMPUTE_MATCHED_BY:-prompt}"
COMPUTE_MATCHED_TOKEN_TOLERANCE="${COMPUTE_MATCHED_TOKEN_TOLERANCE:-0.30}"

BASELINE_METHOD="${BASELINE_METHOD:-naive_multiturn}"
ANCHOR_METHOD="${ANCHOR_METHOD:-restlessness_best}"
CANDIDATE_METHODS="${CANDIDATE_METHODS:-restlessness_triggered,restlessness_adaptive,restlessness_best}"
TARGET_TOKEN_REDUCTION="${TARGET_TOKEN_REDUCTION:-0.50}"
QUALITY_MARGIN="${QUALITY_MARGIN:--0.01}"
MIN_PAIRED_SAMPLES="${MIN_PAIRED_SAMPLES:-16}"
EXIT_ON_NO_CANDIDATE="${EXIT_ON_NO_CANDIDATE:-false}"

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
  --trigger-objective "$TRIGGER_OBJECTIVE"
  --adaptive-stop-delta "$ADAPTIVE_STOP_DELTA"
  --adaptive-min-iters "$ADAPTIVE_MIN_ITERS"
  --compute-tag "method_search"
  --stage "main"
  --cue-offset "$CUE_OFFSET"
  --aut-offset "$AUT_OFFSET"
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

run_stage "generate-grid-method-search" "${GEN_CMD[@]}"

if [[ "$APPEND_SCORES" == "true" ]]; then
  run_stage "score-method-search" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --append-scores
else
  run_stage "score-method-search" "${CLI_CMD[@]}" score --input "$RUNS_DIR/runs.jsonl" --output-dir "$SCORES_DIR" --require-single-session --no-append-scores
fi

run_stage "analyze-method-search" "${CLI_CMD[@]}" analyze-frontier \
  --runs "$SCORES_DIR/scores.jsonl" \
  --require-single-session \
  --exclude-invalid \
  --paired-by prompt \
  --compute-matched-by "$COMPUTE_MATCHED_BY" \
  --compute-matched-k "$BEST_OF_K" \
  --compute-matched-token-tolerance "$COMPUTE_MATCHED_TOKEN_TOLERANCE" \
  --output-dir "$ANALYSIS_DIR"

run_stage "audit-homogeneity-method-search" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --output-dir "$ANALYSIS_DIR"
run_stage "audit-homogeneity-method-search-by-task" "${CLI_CMD[@]}" audit-homogeneity --runs "$RUNS_DIR/runs.jsonl" --by-task --output-dir "$ANALYSIS_DIR/task_stratified"

run_stage "method-search-decision" "$PYTHON_BIN" - \
  "$SCORES_DIR/scores.jsonl" \
  "$DECISION_JSON" \
  "$DECISION_MD" \
  "$BASELINE_METHOD" \
  "$ANCHOR_METHOD" \
  "$CANDIDATE_METHODS" \
  "$TARGET_TOKEN_REDUCTION" \
  "$QUALITY_MARGIN" \
  "$MIN_PAIRED_SAMPLES" <<'PY'
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random

scores_path = Path(sys.argv[1])
decision_json = Path(sys.argv[2])
decision_md = Path(sys.argv[3])
baseline_method = str(sys.argv[4]).strip()
anchor_method = str(sys.argv[5]).strip()
candidate_methods = [m.strip() for m in str(sys.argv[6]).split(",") if m.strip()]
target_token_reduction = float(sys.argv[7])
quality_margin = float(sys.argv[8])
min_pairs = int(sys.argv[9])


def bootstrap_mean_ci(values, n_boot=1000, seed=7):
    if not values:
        return 0.0, 0.0, 0.0
    rng = Random(seed)
    n = len(values)
    means = []
    for _ in range(max(100, n_boot)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(values) / n
    low_idx = max(0, int(0.025 * len(means)) - 1)
    high_idx = min(len(means) - 1, int(0.975 * len(means)) - 1)
    return means[low_idx], mean, means[high_idx]


def objective(row):
    n = float(row.get("novelty", 0.0))
    a = float(row.get("appropriateness", 0.0))
    return math.sqrt(max(0.0, n) * max(0.0, a))


rows = []
with scores_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

rows = [r for r in rows if str(r.get("task_id", "")).lower() in {"cdat", "aut"}]
valid_rows = [r for r in rows if bool(r.get("valid_for_primary", False))]

by_method = defaultdict(list)
for r in valid_rows:
    by_method[str(r.get("method", ""))].append(r)

method_summary = []
for method, bucket in sorted(by_method.items()):
    objs = [objective(r) for r in bucket]
    toks = [int(r.get("metadata", {}).get("tokens_total", 0) or 0) for r in bucket]
    calls = [int(r.get("metadata", {}).get("effective_calls", r.get("metadata", {}).get("generation_calls", 1) or 1)) for r in bucket]
    method_summary.append(
        {
            "method": method,
            "n": len(bucket),
            "objective_mean": (sum(objs) / len(objs)) if objs else 0.0,
            "tokens_mean": (sum(toks) / len(toks)) if toks else 0.0,
            "calls_mean": (sum(calls) / len(calls)) if calls else 0.0,
        }
    )

def pair_map(method_name):
    out = {}
    for r in by_method.get(method_name, []):
        meta = r.get("metadata", {}) if isinstance(r.get("metadata", {}), dict) else {}
        key = (
            str(r.get("model_id", "")),
            str(r.get("task_id", "")),
            str(meta.get("compute_group_id", "")),
        )
        out[key] = r
    return out

baseline_map = pair_map(baseline_method)
anchor_map = pair_map(anchor_method)

candidate_rows = []
for method in candidate_methods:
    m_map = pair_map(method)
    common = sorted(set(baseline_map.keys()) & set(m_map.keys()))
    deltas = []
    token_reductions = []
    call_reductions = []
    for key in common:
        b = baseline_map[key]
        m = m_map[key]
        b_obj = objective(b)
        m_obj = objective(m)
        deltas.append(m_obj - b_obj)

        b_tok = int(b.get("metadata", {}).get("tokens_total", 0) or 0)
        m_tok = int(m.get("metadata", {}).get("tokens_total", 0) or 0)
        if b_tok > 0:
            token_reductions.append(1.0 - (m_tok / b_tok))

        b_call = int(b.get("metadata", {}).get("effective_calls", b.get("metadata", {}).get("generation_calls", 1) or 1))
        m_call = int(m.get("metadata", {}).get("effective_calls", m.get("metadata", {}).get("generation_calls", 1) or 1))
        if b_call > 0:
            call_reductions.append(1.0 - (m_call / b_call))

    d_low, d_mean, d_high = bootstrap_mean_ci(deltas)
    t_low, t_mean, t_high = bootstrap_mean_ci(token_reductions)
    c_low, c_mean, c_high = bootstrap_mean_ci(call_reductions)

    anchor_common = sorted(set(anchor_map.keys()) & set(m_map.keys()))
    delta_anchor = []
    for key in anchor_common:
        a = anchor_map[key]
        m = m_map[key]
        delta_anchor.append(objective(m) - objective(a))
    a_low, a_mean, a_high = bootstrap_mean_ci(delta_anchor)

    passes = (
        len(common) >= min_pairs
        and t_mean >= target_token_reduction
        and d_low >= quality_margin
    )
    candidate_rows.append(
        {
            "method": method,
            "n_pairs_vs_baseline": len(common),
            "objective_delta_vs_baseline": {"ci_low": d_low, "mean": d_mean, "ci_high": d_high},
            "token_reduction_vs_baseline": {"ci_low": t_low, "mean": t_mean, "ci_high": t_high},
            "call_reduction_vs_baseline": {"ci_low": c_low, "mean": c_mean, "ci_high": c_high},
            "objective_delta_vs_anchor": {"ci_low": a_low, "mean": a_mean, "ci_high": a_high},
            "passes_gate": passes,
        }
    )

passing = [r for r in candidate_rows if r.get("passes_gate")]
recommended = None
if passing:
    passing = sorted(
        passing,
        key=lambda x: (
            x["objective_delta_vs_baseline"]["mean"],
            x["token_reduction_vs_baseline"]["mean"],
        ),
        reverse=True,
    )
    recommended = passing[0]["method"]

payload = {
    "status": "GO" if recommended else "NO_GO",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "baseline_method": baseline_method,
    "anchor_method": anchor_method,
    "thresholds": {
        "target_token_reduction": target_token_reduction,
        "quality_margin": quality_margin,
        "min_pairs": min_pairs,
    },
    "method_summary": method_summary,
    "candidate_results": candidate_rows,
    "recommended_method": recommended,
    "next_action": (
        f"Promote `{recommended}` to next larger run with anchor `{anchor_method}` retained."
        if recommended
        else "No candidate met gate; tune trigger threshold/iterations and rerun micro-lab."
    ),
}

decision_json.parent.mkdir(parents=True, exist_ok=True)
decision_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

md = []
md.append("# Method Search Report")
md.append("")
md.append(f"- Status: **{payload['status']}**")
md.append(f"- Baseline: `{baseline_method}` (unstructured multi-turn)")
md.append(f"- Anchor: `{anchor_method}` (current working method)")
md.append("")
md.append("## Method Means")
for row in method_summary:
    md.append(
        f"- {row['method']}: n={row['n']} objective={row['objective_mean']:.4f} "
        f"tokens={row['tokens_mean']:.1f} calls={row['calls_mean']:.2f}"
    )
md.append("")
md.append("## Candidate Gates")
for row in candidate_rows:
    od = row["objective_delta_vs_baseline"]
    tr = row["token_reduction_vs_baseline"]
    cr = row["call_reduction_vs_baseline"]
    md.append(
        f"- {row['method']}: pairs={row['n_pairs_vs_baseline']} "
        f"obj_delta={od['mean']:.4f} [{od['ci_low']:.4f},{od['ci_high']:.4f}] "
        f"token_reduction={tr['mean']:.3f} [{tr['ci_low']:.3f},{tr['ci_high']:.3f}] "
        f"call_reduction={cr['mean']:.3f} [{cr['ci_low']:.3f},{cr['ci_high']:.3f}] "
        f"pass={row['passes_gate']}"
    )
md.append("")
if recommended:
    md.append(f"## Recommendation\n- Promote `{recommended}` to next stage.")
else:
    md.append("## Recommendation\n- No method passed gate; retune and rerun micro-lab.")
md.append("")
md.append(f"Raw JSON: `{decision_json}`")
decision_md.write_text("\n".join(md) + "\n", encoding="utf-8")

print(str(decision_json))
print(str(decision_md))
PY

cat <<EOF
Phase 3 method-search micro-lab complete.
Session: $SESSION_ID
Runs: $RUNS_DIR
Scores: $SCORES_DIR
Analysis: $ANALYSIS_DIR
Decision JSON: $DECISION_JSON
Decision report: $DECISION_MD
EOF

if [[ "$EXIT_ON_NO_CANDIDATE" == "true" ]]; then
  status="$("$PYTHON_BIN" - "$DECISION_JSON" <<'PY'
import json,sys
obj=json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(obj.get("status", "NO_GO"))
PY
)"
  if [[ "$status" != "GO" ]]; then
    echo "Method search returned NO_GO." >&2
    exit 5
  fi
fi
