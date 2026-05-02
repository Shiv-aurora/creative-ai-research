#!/bin/bash
#SBATCH --job-name=creativeai_phase6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sarora01@villanova.edu

# Phase 6 — Ablation + Combined Internal/External experiment.
# Part A: Isolate which component of anti_repetition drives the creativity gain.
# Part B: Test if anti_repetition compounds with restlessness_best.
# Models: Qwen2.5-3B-Instruct + Mistral-7B-Instruct (two families, instruct only).
set -euo pipefail

echo "Job $SLURM_JOB_ID started $(date) on $SLURMD_NODENAME"

# ── Venv + CUDA ───────────────────────────────────────────────────────────────
source "$HOME/polca_venv/bin/activate"
export LD_LIBRARY_PATH="$HOME/creative-ai-libs:/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}"
python3 -c "import llama_cpp; print('llama_cpp OK')"

REPO="$HOME/creative-ai"
MODEL_PATH_MAP="$REPO/model_paths.augie.json"
OUTPUT_ROOT="$REPO/outputs/phase6_ablation"
cd "$REPO"

# ── Part A: Ablation — isolate the mechanism ──────────────────────────────────
echo ""
echo "=== Part A: Ablation (6 profiles × 2 models × 2 tasks × 3 seeds) ==="
ABLATION_OUT="$OUTPUT_ROOT/ablation"

python3 -m creativeai.cli generate-grid \
    --backend llama_cpp \
    --model-path-map "$MODEL_PATH_MAP" \
    --tasks cdat,aut \
    --methods one_shot \
    --sampler-profiles default_nucleus,anti_repetition,ablation_temp_only,ablation_repeat_only,ablation_penalties_only,anti_repetition_strong \
    --models qwen2.5-3b-instruct,mistral-7b-instruct-v0.3 \
    --temperatures 0.7 \
    --seeds 11,37,73 \
    --max-tokens 256 \
    --quantization q4_k_m \
    --strict-json --max-retries 2 \
    --prompt-mode auto --grammar-mode auto \
    --limit-cues 16 --limit-aut 12 \
    --n-gpu-layers -1 --n-ctx 16384 --n-threads 16 \
    --n-batch 2048 --n-ubatch 1024 --n-threads-batch 16 \
    --compute-tag phase6_ablation --stage ablation \
    --progress --progress-every 10 \
    --health-window 20 --health-min-json 0.95 --health-min-valid 0.90 \
    --health-min-samples 20 --health-action quarantine_cell \
    --health-events "$ABLATION_OUT/runs/health_events.jsonl" \
    --output-dir "$ABLATION_OUT/runs" \
    --no-append-runs

python3 -m creativeai.cli score \
    --input "$ABLATION_OUT/runs/runs.jsonl" \
    --output-dir "$ABLATION_OUT/scores" \
    --require-single-session --no-append-scores

python3 -m creativeai.cli analyze-samplers \
    --scores "$ABLATION_OUT/scores/scores.jsonl" \
    --baseline-profile default_nucleus \
    --exclude-invalid --require-single-session \
    --output-dir "$ABLATION_OUT/analysis"

python3 -m creativeai.cli analyze-frontier \
    --runs "$ABLATION_OUT/scores/scores.jsonl" \
    --require-single-session --exclude-invalid \
    --paired-by prompt --compute-matched-by prompt \
    --compute-matched-k 4 --token-budget 256 \
    --output-dir "$ABLATION_OUT/analysis"

# ── Part B: Combined internal + external ──────────────────────────────────────
echo ""
echo "=== Part B: Combined (2 profiles × 2 methods × 2 models × 2 tasks × 2 seeds) ==="
COMBINED_OUT="$OUTPUT_ROOT/combined"

python3 -m creativeai.cli generate-grid \
    --backend llama_cpp \
    --model-path-map "$MODEL_PATH_MAP" \
    --tasks cdat,aut \
    --methods one_shot,restlessness_best \
    --sampler-profiles default_nucleus,anti_repetition \
    --models qwen2.5-3b-instruct,mistral-7b-instruct-v0.3 \
    --temperatures 0.7 \
    --seeds 11,37 \
    --max-tokens 256 \
    --quantization q4_k_m \
    --strict-json --max-retries 2 \
    --prompt-mode auto --grammar-mode auto \
    --restlessness-k 3 --best-of-k 4 \
    --limit-cues 16 --limit-aut 12 \
    --n-gpu-layers -1 --n-ctx 16384 --n-threads 16 \
    --n-batch 2048 --n-ubatch 1024 --n-threads-batch 16 \
    --compute-tag phase6_combined --stage combined \
    --progress --progress-every 10 \
    --health-window 20 --health-min-json 0.95 --health-min-valid 0.90 \
    --health-min-samples 20 --health-action quarantine_cell \
    --health-events "$COMBINED_OUT/runs/health_events.jsonl" \
    --output-dir "$COMBINED_OUT/runs" \
    --no-append-runs

python3 -m creativeai.cli score \
    --input "$COMBINED_OUT/runs/runs.jsonl" \
    --output-dir "$COMBINED_OUT/scores" \
    --require-single-session --no-append-scores

python3 -m creativeai.cli analyze-samplers \
    --scores "$COMBINED_OUT/scores/scores.jsonl" \
    --baseline-profile default_nucleus \
    --exclude-invalid --require-single-session \
    --output-dir "$COMBINED_OUT/analysis"

python3 -m creativeai.cli analyze-frontier \
    --runs "$COMBINED_OUT/scores/scores.jsonl" \
    --require-single-session --exclude-invalid \
    --paired-by prompt --compute-matched-by prompt \
    --compute-matched-k 4 --token-budget 256 \
    --output-dir "$COMBINED_OUT/analysis"

echo ""
echo "=== Phase 6 complete: $(date) ==="
echo "Ablation analysis:  $ABLATION_OUT/analysis/sampler_analysis.json"
echo "Combined analysis:  $COMBINED_OUT/analysis/sampler_analysis.json"
