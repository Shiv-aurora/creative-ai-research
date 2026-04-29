# CreativeAI: Creativity Under Constraints

This repository implements a local-first research harness for studying LLM creativity as a novelty-appropriateness frontier.

## Implemented Capabilities

- Reproducible run records with immutable manifests (`run_id`, git hash, model hash, backend, quantization, timestamp).
- Tasks: `dat`, `cdat`, `aut`.
- Methods: `one_shot`, `best_of_k_one_shot`, `restlessness_best`, `restlessness_last_iter`, `restlessness_adaptive`, `brainstorm_then_select`.
- Scoring: novelty, appropriateness, usefulness proxy, failure flags.
- Analysis: frontier plots with bootstrap CIs, base-vs-instruct shift, best-of-N under token budget, paired prompt-level deltas, compute-matched summaries, efficiency tables.
- Homogeneity audit: nearest-neighbor similarity, compactness, self-BLEU, diversity index (pooled or task-stratified).
- Human calibration helpers: stratified rating slice + calibration gate metrics.
- Cross-backend trend comparison utility (MPS vs CUDA runs).
- Strict JSON generation mode with retry budget and parse provenance tracking.
- Runtime health gate + cell quarantine + session lineage checks.
- Token/call accounting on every run for compute-matched analysis.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Optional for llama.cpp local models:

```bash
pip install '.[llama,data]'
```

Optional for semantic embeddings in scoring:

```bash
pip install '.[semantic]'
```

If editable installs are required for local development and your environment skips hidden `.pth` files, use module mode:

```bash
python -m creativeai.cli --help
```

## Quick Start (No Model Download Needed)

Run with deterministic mock backend:

```bash
creativeai generate \
  --task cdat \
  --method restlessness_best \
  --model gemma-2-2b \
  --backend mock \
  --seed 11 \
  --temperature 0.7 \
  --strict-json \
  --max-retries 2 \
  --cue forest \
  --output-dir outputs/runs

creativeai score \
  --input outputs/runs/runs.jsonl \
  --output-dir outputs/scores

creativeai analyze-frontier \
  --runs outputs/scores/scores.jsonl \
  --exclude-invalid \
  --paired-by prompt \
  --compute-matched-k 4 \
  --compute-matched-by prompt \
  --output-dir outputs/analysis

creativeai audit-homogeneity \
  --runs outputs/runs/runs.jsonl \
  --by-task \
  --output-dir outputs/analysis
```

## Full Grid (Plan Defaults)

```bash
creativeai generate-grid \
  --backend mock \
  --tasks dat,cdat,aut \
  --methods one_shot,best_of_k_one_shot,restlessness_best,restlessness_last_iter \
  --models gemma-2-2b,gemma-2-2b-it,qwen2.5-3b,qwen2.5-3b-instruct,mistral-7b-v0.3,mistral-7b-instruct-v0.3 \
  --temperatures 0.2,0.7,1.0,1.3 \
  --seeds 11,37,73,101,149 \
  --strict-json \
  --max-retries 2 \
  --output-dir outputs/runs
```

To run with `llama_cpp`, pass `--backend llama_cpp` and `--model-path` (or `--model-path-map`).

## Real MPS Run With Downloaded GGUFs

If you have downloaded local GGUFs and created `model_paths.downloaded.json`, run:

```bash
source .venv/bin/activate
scripts/run_phase1_real_mps.sh
```

For a fast real smoke batch:

```bash
source .venv/bin/activate
creativeai generate-grid \
  --backend llama_cpp \
  --model-path-map model_paths.downloaded.json \
  --n-gpu-layers -1 \
  --tasks dat,cdat,aut \
  --methods one_shot,best_of_k_one_shot,restlessness_best \
  --models gemma-2b-it,qwen2.5-3b-instruct \
  --temperatures 0.7 \
  --seeds 11 \
  --strict-json \
  --limit-cues 8 \
  --limit-aut 6 \
  --output-dir outputs/phase1_real_mini/runs
```

Phase 3 staged local run (micro-gate -> main -> confirm -> DAT aux, target 960 runs):

```bash
source .venv/bin/activate
caffeinate -dimsu env BACKEND=llama_cpp MODEL_PATH_MAP=model_paths.downloaded.json \
  scripts/run_phase3_main_local.sh \
  outputs/phase3_v3/runs \
  outputs/phase3_v3/scores \
  outputs/phase3_v3/analysis
```

## CLI Reference

- `creativeai generate`
- `creativeai generate-grid`
- `creativeai score`
- `creativeai analyze-frontier`
- `creativeai audit-homogeneity`
- `creativeai prepare-human-slice`
- `creativeai eval-human`
- `creativeai compare-backends`

## Output Contract

- `outputs/runs/*.json`: single run artifact.
- `outputs/runs/runs.jsonl`: append-only run log.
- `outputs/scores/scores.jsonl`: append-only score log.
- `outputs/analysis/frontier_analysis.json`: frontier + shift + best-of-N summary.
- `outputs/analysis/frontier.png`: frontier plot.
- `outputs/analysis/homogeneity_audit.json`: homogeneity summary.

## Notes

- Word-level noun checking is format-level (single-token pattern), not linguistic POS tagging.
- Semantic embedding scoring prefers `sentence-transformers` when available; otherwise it falls back to deterministic hash embeddings.
- Usefulness is a conservative proxy and must be calibrated with human ratings before strong claims.
- Mock backend is for pipeline validation only; use `llama_cpp` backend for real experiments.
