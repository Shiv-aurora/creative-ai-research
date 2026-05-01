# North Star — CreativeAI Research

**Read this at the start of every session. It is an assignment brief, not full docs.**

---

## What this project is

A research harness measuring how LLMs trade **novelty** against **appropriateness** on open-ended creativity tasks. The core claim under investigation: decoding strategy (sampler, method, temperature) moves outputs on a novelty–appropriateness frontier in ways that are separable from generic quality and model size.

**This is empirical ML research**, not a product. Every run is logged with immutable metadata so results are traceable and replicable.

---

## The experiment pipeline (one run cycle)

```
generate-grid  →  score  →  analyze-frontier  →  audit-homogeneity  →  compare-backends
```

1. **generate-grid** — run a Cartesian grid (models × methods × tasks × seeds) and write `runs.jsonl`
2. **score** — attach novelty/appropriateness/usefulness scores → `scores.jsonl`
3. **analyze-frontier** — frontier plots, paired deltas, compute-matched summaries → `frontier_analysis.json`
4. **audit-homogeneity** — diversity/similarity audit (self-BLEU, NN-similarity) → `homogeneity_audit.json`
5. **compare-backends** — check directional parity between MPS (M1 local) and CUDA (A100) runs

---

## Tasks

| ID | Name | What it probes |
|----|------|----------------|
| `dat` | Divergent Associative Task | semantic distance between word pairs |
| `cdat` | Constrained DAT | same but under lexical constraint |
| `aut` | Alternative Uses Task | unusual uses for an object |

---

## Methods (two generations)

**External** — multiple LLM calls; prompting does the creative work:
| ID | Description |
|----|-------------|
| `one_shot` | single sample (baseline) |
| `best_of_k_one_shot` | k independent samples, pick best by score |
| `restlessness_best` | iterative refinement, keep best |
| `restlessness_last_iter` | iterative refinement, keep final iteration |
| `restlessness_adaptive` | restlessness with adaptive stopping |
| `brainstorm_then_select` | brainstorm list, then select |

**Internal** — single call; decoding parameters do the creative work (Phase 4 onward focus):
| Profile | Key knobs |
|---------|-----------|
| `default_nucleus` | temp=0.7, top_p=0.9, top_k=40 — baseline |
| `anti_repetition` | temp=0.9, repeat_penalty=1.15, freq_pen=0.2, pres_pen=0.1 |
| `spread_topk_minp` | temp=1.0, top_k=80, min_p=0.05, repeat_penalty=1.05 |
| `mirostat` | entropy-targeting sampler, mode=2, tau=5.0 |
| `high_temp` | temp=1.1, top_p=0.95, top_k=0 |

---

## Models (all Q4_K_M GGUF via llama.cpp)

- gemma-2b, gemma-2b-it
- qwen2.5-3b, qwen2.5-3b-instruct
- mistral-7b-v0.3, mistral-7b-instruct-v0.3

Base vs. instruct pairs are intentional — the shift is part of the hypothesis.

---

## Experimental phases

| Phase | Config | Description | Status |
|-------|--------|-------------|--------|
| 1 | `phase1_local.json` | Mock + real weights smoke test | Done |
| 2 | `phase2_micro_lab.json` | Micro-lab calibration run | Done |
| 3 | `phase3_main_local.json` | Main frontier grid on M1/MPS | Done; frontier is the MPS baseline |
| 4 | `phase4_decoding_curiosity.json` | Sampler profile study (anti_repetition, spread_topk_minp, …) | Done locally |
| 5 | `phase5_cuda_confirm.json` | **Current target** — CUDA confirm of Phase-4 sampler-profile findings across all 6 models. Axis: sampler profiles (internal), method: `one_shot`. Compare-backends vs Phase 3 MPS frontier. |

**Phase 3 result (MPS baseline):** `anti_repetition` and `spread_topk_minp` profiles showed paired objective delta ~+0.04 vs default nucleus, with bootstrap CIs that don't include zero. That's the bar to replicate and extend on CUDA.

---

## Key empirical result so far

From Phase 4 confirm run on M1:
- `anti_repetition`: paired delta +0.0395, CI [+0.0141, +0.0651]
- `spread_topk_minp`: +0.0371, CI [+0.0150, +0.0610]

Both versus `default_nucleus` baseline. These are the findings Phase 5 must confirm hold on CUDA before any paper-level claims.

---

## Active goal (as of 2026-05-01)

Run **Phase 5** on **Augie HPC** (Villanova A100, SLURM). This replaces the Colab plan.

The deliverable: `outputs/phase5_cuda_confirm/analysis/frontier_analysis.json` + `backend_parity.json` comparing CUDA vs the Phase 3 MPS frontier.

See `scripts/run_phase5_augie.sh` (SLURM job) and `scripts/stage_augie.sh` (login-node staging).

---

## Augie HPC facts (memorize before touching anything)

- **Login**: `ssh sarora01@augie.villanova.edu` (VPN required: Villanova GlobalProtect)
- **Scheduler**: SLURM, partition `gpu`, single A100 40 GB on `gpu001`
- **CRITICAL**: gpu001 has **NO internet**. All models and pip installs must be staged from the login node before submitting.
- **Home is NFS**: files written on login node are visible inside the job. Stage to `~/creative-ai/` and `~/creative-ai-models/`.
- **Get code there**: `scp` or `rsync` from laptop — git pull on Augie fails (no HTTPS creds).
- **Email**: `sarora01@villanova.edu` on job END/FAIL.
- **Venv to reuse**: `~/polca_venv/` has torch/pandas/pytest but NOT llama-cpp-python. Install llama-cpp-python into it (or a new venv) on the login node before submitting.

---

## Repo layout (source only)

```
creativeai/          # Python package — all core logic
  cli.py             # CLI entry point for all commands
  pipeline.py        # generate_run / scoring loop
  methods.py         # one_shot, restlessness, best_of_k, brainstorm
  analysis.py        # frontier, homogeneity, compare-backends
  model_backend.py   # llama_cpp / mock / transformers adapters
  decoding.py        # sampler profiles
  scoring.py         # novelty, appropriateness, usefulness
configs/             # JSON grids per phase
scripts/             # shell/Python drivers per phase
outputs/             # all run artifacts (gitignored for large runs)
models/gguf/         # local GGUF weights (not in git)
```

---

## CLI cheat-sheet

```bash
# Phase 5 — internal/sampler-profile grid (one_shot, decoding does the work)
creativeai generate-grid --backend llama_cpp --model-path-map model_paths.json \
  --tasks cdat,aut,dat --methods one_shot \
  --sampler-profiles default_nucleus,anti_repetition,spread_topk_minp,mirostat,high_temp \
  --models gemma-2b,gemma-2b-it,qwen2.5-3b,qwen2.5-3b-instruct,mistral-7b-v0.3,mistral-7b-instruct-v0.3 \
  --temperatures 0.7 --seeds 11,37 --strict-json --n-gpu-layers -1 --output-dir outputs/phase5/runs

creativeai score --input outputs/phase5/runs/runs.jsonl --output-dir outputs/phase5/scores
creativeai analyze-samplers --scores outputs/phase5/scores/scores.jsonl \
  --baseline-profile default_nucleus --output-dir outputs/phase5/analysis
creativeai analyze-frontier --runs outputs/phase5/scores/scores.jsonl --paired-by prompt \
  --compute-matched-k 4 --token-budget 256 --output-dir outputs/phase5/analysis
creativeai audit-homogeneity --runs outputs/phase5/runs/runs.jsonl --output-dir outputs/phase5/analysis
creativeai compare-backends \
  --local-frontier outputs/phase3_main_*/analysis/frontier_analysis.json \
  --cuda-frontier outputs/phase5/analysis/frontier_analysis.json \
  --output outputs/phase5/analysis/backend_parity.json
```

---

## Do not do

- Do not pip install anything inside a SLURM job (no internet on gpu001)
- Do not assume `model_paths.downloaded.json` is present on Augie — it must be generated during staging
- Do not use the mock backend for empirical claims
- Do not compare backends without `--require-single-session` or `--exclude-invalid` — dirty data contaminates the frontier
