#!/bin/bash
#SBATCH --job-name=creativeai_phase5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sarora01@villanova.edu

# Phase 5 — CUDA confirm of Phase-4 sampler-profile findings.
# Axis: sampler profiles (internal decoding). Method: one_shot.
# No internet calls — all deps and models pre-staged by stage_augie.sh.
set -euo pipefail

echo "Job $SLURM_JOB_ID started $(date)"
echo "Node: $SLURMD_NODENAME"

# ── CUDA runtime libraries ────────────────────────────────────────────────────
# llama-cpp-python cu121 wheel links against libcudart.so.12 / libcublas.so.12.
# Augie has CUDA 13.1 only, so we point .so.12 symlinks at .so.13 in ~/creative-ai-libs.
# libcuda.so.1 (driver) is only on gpu001; this block is a no-op on the login node.
module load cuda 2>/dev/null || module load cuda/13 2>/dev/null || true

export LD_LIBRARY_PATH="$HOME/creative-ai-libs:/usr/local/cuda-13.1/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f "$HOME/creative-ai-venv/bin/activate" ]; then
    source "$HOME/creative-ai-venv/bin/activate"
elif [ -f "$HOME/polca_venv/bin/activate" ]; then
    source "$HOME/polca_venv/bin/activate"
else
    echo "ERROR: no venv found at ~/creative-ai-venv or ~/polca_venv" >&2
    exit 1
fi
echo "Python: $(which python3)  ($(python3 --version))"

# Verify llama_cpp imports before wasting queue time
python3 -c "import llama_cpp; print('llama_cpp OK:', llama_cpp.__file__)"

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO="$HOME/creative-ai"
MODEL_PATH_MAP="$REPO/model_paths.augie.json"
OUTPUT_ROOT="$REPO/outputs/phase5_cuda_confirm"
RUNS_DIR="$OUTPUT_ROOT/runs"
SCORES_DIR="$OUTPUT_ROOT/scores"
ANALYSIS_DIR="$OUTPUT_ROOT/analysis"

if [ ! -f "$MODEL_PATH_MAP" ]; then
    echo "ERROR: $MODEL_PATH_MAP not found. Run stage_augie.sh on the login node first." >&2
    exit 1
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found." >&2
fi

cd "$REPO"
mkdir -p "$RUNS_DIR" "$SCORES_DIR" "$ANALYSIS_DIR"

echo ""
echo "=== Phase 5 — internal decoding / sampler profiles on CUDA ==="
echo "  Repo:      $REPO"
echo "  Model map: $MODEL_PATH_MAP"
echo "  Output:    $OUTPUT_ROOT"
echo "  Grid:      6 models × 5 profiles × 3 tasks × 2 seeds"
echo ""

# ── 1. generate-grid ──────────────────────────────────────────────────────────
python3 -m creativeai.cli generate-grid \
    --backend llama_cpp \
    --model-path-map "$MODEL_PATH_MAP" \
    --tasks cdat,aut,dat \
    --methods one_shot \
    --sampler-profiles default_nucleus,anti_repetition,spread_topk_minp,mirostat,high_temp \
    --models gemma-2b,gemma-2b-it,qwen2.5-3b,qwen2.5-3b-instruct,mistral-7b-v0.3,mistral-7b-instruct-v0.3 \
    --temperatures 0.7 \
    --seeds 11,37 \
    --top-p 0.9 \
    --max-tokens 256 \
    --quantization q4_k_m \
    --strict-json \
    --max-retries 2 \
    --prompt-mode auto \
    --grammar-mode auto \
    --dat-repeats 6 \
    --limit-cues 20 \
    --limit-aut 15 \
    --compute-tag phase5_cuda_confirm \
    --stage confirm \
    --n-gpu-layers -1 \
    --n-ctx 16384 \
    --n-threads 16 \
    --n-batch 2048 \
    --n-ubatch 1024 \
    --n-threads-batch 16 \
    --progress \
    --progress-every 10 \
    --health-window 20 \
    --health-min-json 0.95 \
    --health-min-valid 0.90 \
    --health-min-samples 20 \
    --health-action quarantine_cell \
    --health-events "$RUNS_DIR/health_events.jsonl" \
    --output-dir "$RUNS_DIR" \
    --no-append-runs

echo ""
echo "=== 2. Score ==="
python3 -m creativeai.cli score \
    --input "$RUNS_DIR/runs.jsonl" \
    --output-dir "$SCORES_DIR" \
    --require-single-session \
    --no-append-scores

echo ""
echo "=== 3. Frontier analysis ==="
python3 -m creativeai.cli analyze-frontier \
    --runs "$SCORES_DIR/scores.jsonl" \
    --require-single-session \
    --exclude-invalid \
    --paired-by prompt \
    --compute-matched-by prompt \
    --compute-matched-k 4 \
    --compute-matched-token-tolerance 0.25 \
    --token-budget 256 \
    --output-dir "$ANALYSIS_DIR"

echo ""
echo "=== 4. Sampler profile analysis (primary result) ==="
python3 -m creativeai.cli analyze-samplers \
    --scores "$SCORES_DIR/scores.jsonl" \
    --baseline-profile default_nucleus \
    --exclude-invalid \
    --require-single-session \
    --output-dir "$ANALYSIS_DIR"

echo ""
echo "=== 5. Homogeneity audit (pooled) ==="
python3 -m creativeai.cli audit-homogeneity \
    --runs "$RUNS_DIR/runs.jsonl" \
    --output-dir "$ANALYSIS_DIR"

echo ""
echo "=== 6. Homogeneity audit (task-stratified) ==="
python3 -m creativeai.cli audit-homogeneity \
    --runs "$RUNS_DIR/runs.jsonl" \
    --by-task \
    --output-dir "$ANALYSIS_DIR/task_stratified"

# ── 7. Compare-backends (vs Phase 3 MPS frontier) ────────────────────────────
PHASE3_FRONTIER=$(ls "$REPO/outputs/phase3_main"*/analysis/frontier_analysis.json 2>/dev/null | head -1 || true)
if [ -n "$PHASE3_FRONTIER" ]; then
    echo ""
    echo "=== 7. Compare backends (CUDA vs MPS) ==="
    python3 -m creativeai.cli compare-backends \
        --local-frontier "$PHASE3_FRONTIER" \
        --cuda-frontier "$ANALYSIS_DIR/frontier_analysis.json" \
        --output "$ANALYSIS_DIR/backend_parity.json"
    echo "  -> $ANALYSIS_DIR/backend_parity.json"
else
    echo ""
    echo "NOTE: No phase3_main* frontier on Augie — skipping compare-backends."
    echo "      To run later:"
    echo "  python3 -m creativeai.cli compare-backends \\"
    echo "    --local-frontier outputs/phase3_main_XXXX/analysis/frontier_analysis.json \\"
    echo "    --cuda-frontier $ANALYSIS_DIR/frontier_analysis.json \\"
    echo "    --output $ANALYSIS_DIR/backend_parity.json"
fi

echo ""
echo "=== Phase 5 complete: $(date) ==="
echo "Key outputs:"
echo "  Sampler analysis: $ANALYSIS_DIR/sampler_analysis.json"
echo "  Frontier:         $ANALYSIS_DIR/frontier_analysis.json"
echo "  Homogeneity:      $ANALYSIS_DIR/homogeneity_audit.json"
echo "  Output root:      $OUTPUT_ROOT"
