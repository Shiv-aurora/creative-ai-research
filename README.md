# CreativeAI

**Creativity under constraints** — a local-first research harness for studying how large language models trade **novelty** against **appropriateness** when generation is bounded by task structure, decoding, and compute.

This repository is **ongoing research** by **Shivam Arora**. The codebase was recently made public while the full experimental stack (scripts, phase configs, and reproducibility paths) is still being exercised and hardened. Treat interfaces, default grids, and auxiliary scripts as **live**: they may change as runs complete and analysis tightens. Issues and reproducibility reports are welcome.

---

## Why this project exists

Creativity in humans and machines is rarely “more random = more creative.” Psychometric and cognitive work typically treats creative products as lying on a **frontier**: outputs should be **original** relative to a reference class *and* **appropriate** to the goal — novelty without fit is not the same construct. For LLMs, that tension is operational: temperature, best-of-N, iterative refinement, and instruction style all move points on a novelty–appropriateness plane in ways that are easy to confound with generic fluency or length.

**CreativeAI** is a deliberately **instrumented** harness: every run records immutable metadata (identifiers, git revision, model and backend fingerprints, quantization, timestamps), supports multiple **tasks** and **generation methods**, attaches **automatic scores** (with explicit failure modes and calibration hooks), and ships **analysis** utilities so comparisons can be made under **matched compute** and **paired prompts** where possible. The goal is not a single leaderboard score but **traceable** evidence about *where* on the frontier different stacks sit — and how homogeneous “creative” outputs are across seeds and methods.

---

## What is implemented today

- **Run ledger**: reproducible run records with immutable manifests (`run_id`, git hash, model hash, backend, quantization, timestamp).
- **Tasks**: `dat`, `cdat`, `aut` (divergent-style probes with different scoring geometry).
- **Methods**: `one_shot`, `best_of_k_one_shot`, `restlessness_best`, `restlessness_last_iter`, `restlessness_adaptive`, `brainstorm_then_select`.
- **Scoring**: novelty, appropriateness, a conservative usefulness proxy, and structured failure flags.
- **Analysis**: frontier plots with bootstrap confidence intervals, base-vs-instruct shifts, best-of-N under token budget, paired prompt-level deltas, compute-matched summaries, efficiency tables.
- **Homogeneity audit**: nearest-neighbor similarity, compactness, self-BLEU, diversity index (pooled or task-stratified).
- **Human calibration**: stratified rating slices and calibration gate metrics.
- **Cross-backend comparison**: e.g. MPS vs CUDA trend checks.
- **Strict JSON generation**: retry budget with parse provenance.
- **Operational guards**: runtime health gate, cell quarantine, session lineage checks.
- **Accounting**: token and call counts on every run for compute-matched analysis.

For broader literature positioning and open questions, see `deep-research-report.md` (research notes; not a substitute for cited papers in any eventual write-up).

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Optional dependencies for local **llama.cpp** models and data tooling:

```bash
pip install '.[llama,data]'
```

Optional **semantic** embeddings in scoring:

```bash
pip install '.[semantic]'
```

If your environment skips hidden `.pth` files for editable installs, prefer module invocation:

```bash
python -m creativeai.cli --help
```

---

## Quick start (no model download)

The **mock** backend validates the full pipeline without weights:

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

---

## Full grid (plan defaults)

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

For **llama.cpp**, supply `--backend llama_cpp` and `--model-path` or `--model-path-map`.

---

## Real local runs (Metal / GGUF)

With downloaded GGUF weights and `model_paths.downloaded.json` (see `model_paths.example.json`):

```bash
source .venv/bin/activate
scripts/run_phase1_real_mps.sh
```

Fast smoke batch on real weights:

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

Staged Phase 3 local driver (micro-gate → main → confirm → DAT aux; large run target):

```bash
source .venv/bin/activate
caffeinate -dimsu env BACKEND=llama_cpp MODEL_PATH_MAP=model_paths.downloaded.json \
  scripts/run_phase3_main_local.sh \
  outputs/phase3_v3/runs \
  outputs/phase3_v3/scores \
  outputs/phase3_v3/analysis
```

Additional phase scripts and Colab-oriented runners live under `scripts/` and `configs/`; they are part of the same research program and may be updated as batches complete.

---

## CLI reference

| Command | Role |
|--------|------|
| `creativeai generate` | Single configuration run |
| `creativeai generate-grid` | Cartesian product over tasks, methods, models, temperatures, seeds |
| `creativeai score` | Score a `runs.jsonl` stream |
| `creativeai analyze-frontier` | Frontier, shift, best-of-N, paired and compute-matched summaries |
| `creativeai audit-homogeneity` | Diversity / similarity audit |
| `creativeai prepare-human-slice` | Stratified slice for human rating |
| `creativeai eval-human` | Human rating evaluation helpers |
| `creativeai compare-backends` | Cross-backend trend comparison |

Use `creativeai <command> --help` for full flags.

---

## Output contract

| Path | Contents |
|------|----------|
| `outputs/runs/*.json` | Single-run artifact |
| `outputs/runs/runs.jsonl` | Append-only run log |
| `outputs/scores/scores.jsonl` | Append-only score log |
| `outputs/analysis/frontier_analysis.json` | Frontier, shift, best-of-N summary |
| `outputs/analysis/frontier.png` | Frontier plot |
| `outputs/analysis/homogeneity_audit.json` | Homogeneity summary |

---

## Limitations (read before strong claims)

- **Word-level noun checks** in some paths are format-level (single-token pattern), not full linguistic POS tagging.
- **Semantic scoring** prefers `sentence-transformers` when installed; otherwise it may fall back to deterministic hash embeddings — check your environment before interpreting semantic distance magnitudes.
- **Usefulness** is a conservative automatic proxy; it is **not** a substitute for human creativity judgments until calibrated (see human slice tooling above).
- **Mock backend** is for pipeline validation only; empirical claims require real backends and documented model revisions.

---

## License and attribution

If you use this code or derived artifacts, please cite the repository and name the maintainer when appropriate: **Shivam Arora**, *CreativeAI* (ongoing research). A formal paper citation will be added when available.

---

## Contact

**Shivam Arora** — ongoing work; reach out via GitHub issues for bugs, reproducibility, or collaboration aligned with the research goals above.
