# CreativeAI

**Creativity under constraints** — a research harness for studying how large language models trade **novelty** against **appropriateness** when generation is bounded by task structure, decoding strategy, and compute.

This repository documents ongoing research by **Shivam Arora** (Villanova University). The central question: *can internal decoding parameters — sampler settings that control how a model samples from its own distribution — reliably shift LLM outputs toward the creative frontier, without extra inference calls?*

---

## Core finding (Phase 4–5, confirmed)

> **`anti_repetition` and `spread_topk_minp` sampler profiles produce a statistically reliable improvement in the composite creativity objective over `default_nucleus` baseline, but the effect is concentrated in base models and is model-family-dependent. Instruction-tuned Mistral is actively harmed.**

| Profile | Paired Δ (Phase 5, all 6 models) | 95% CI | Replicated? |
|---|---|---|---|
| anti_repetition | +0.030 | [+0.019, +0.043] | ✓ |
| spread_topk_minp | +0.019 | [+0.006, +0.031] | ✓ |
| mirostat | +0.008 | [−0.007, +0.021] | ✗ |
| high_temp | +0.003 | [−0.010, +0.014] | ✗ |

Per-model breakdown (Phase 5):

| Model | anti_repetition Δ | CI above zero? |
|---|---|---|
| gemma-2b | +0.096 | ✓ |
| gemma-2b-it | +0.037 | ✓ |
| qwen2.5-3b | +0.050 | ✓ |
| qwen2.5-3b-instruct | +0.015 | ✗ |
| mistral-7b-v0.3 | +0.026 | ✗ |
| **mistral-7b-instruct-v0.3** | **−0.033** | **✗ (hurts)** |

Base models: mean Δ = +0.058. Instruct models: mean Δ = +0.006. The aggregate positive result masks this heterogeneity. A single sampler profile is not model-agnostic.

---

## Why this matters

Creativity in humans and machines is rarely "more random = more creative." Psychometric work treats creative products as lying on a **frontier**: outputs must be **original** *and* **appropriate** — novelty without fit is not the same construct. For LLMs, that tension is operational: temperature, best-of-N, iterative refinement, and sampling strategy all move points on a novelty–appropriateness plane in ways that confound with generic fluency or length.

**CreativeAI** studies two classes of approach:

- **External methods** — repeated LLM calls that guide generation from outside (restlessness, best-of-k, brainstorm-then-select). These improve scores but multiply token cost.
- **Internal methods** — sampler parameters that reshape the token distribution during a *single* forward pass (temperature, top-k, min-p, repeat/frequency/presence penalties). These are cost-neutral but their effect on genuine creative quality — not just lexical diversity — is not well understood.

The research question: are "internal" methods a legitimate path to more creative outputs, or do they produce lexically varied but semantically equivalent text that fools automatic novelty metrics?

---

## Experimental arc

| Phase | What ran | Key result |
|---|---|---|
| 1 | Mock + real weights smoke test | Pipeline validated |
| 2 | Micro-lab calibration (CDAT/AUT, Qwen, mock backend) | Scoring and health gates calibrated |
| 3 | Main frontier grid on M1/MPS (restlessness, best-of-k, 6 models) | `restlessness_best` objective ~0.63 at ~1200 tokens/run; `one_shot` at ~0.60 with 228 tokens |
| 4 | Decoding curiosity (sampler profiles, Qwen only, MPS) | `anti_repetition` +0.040, `spread_topk_minp` +0.037 vs `default_nucleus`. CIs above zero. |
| 5 | CUDA confirm (all 6 models, Augie A100) | Aggregate replicates. Per-model reveals effect is base-model-dominant; Mistral-instruct hurt. |
| **6** | **Ablation + combined** (in design) | Which component of anti_repetition drives the effect? Does it compound with restlessness? |

---

## Open questions driving Phase 6

**1. What drives `anti_repetition`?**

The profile has three components: elevated temperature (0.7→0.9), repeat_penalty (1.15), and frequency+presence penalties (0.2/0.1). Phase 6 isolates each via dedicated ablation profiles (`ablation_temp_only`, `ablation_repeat_only`, `ablation_penalties_only`). If penalty-driven distributional pressure is the mechanism, that has implications for how the profile should be tuned per model family.

**2. Why does Mistral-instruct score lower under anti_repetition?**

RLHF alignment enforces strong distributional priors. The frequency/presence penalties may push Mistral-instruct into low-probability regions that its alignment training specifically disfavors — producing fluent but off-task or incoherent outputs that score poorly on appropriateness. If this is the mechanism, a milder or temperature-only profile might recover the gain without triggering the regression.

**3. Does internal + external compound?**

Phase 3 showed `restlessness_best` achieves ~0.63 at high token cost. Phase 5 showed `anti_repetition` achieves ~0.51 at 122 tokens. If restlessness already explores enough of the distribution, applying anti_repetition inside restlessness adds nothing. If restlessness is constrained by the token distribution quality per call, anti_repetition could lift each call and compound the effect. Phase 6 Part B tests this directly.

**4. Are novelty scores semantically valid?**

Phase 4–5 novelty was computed with hash embeddings (token-level heuristic), not semantic embeddings (`sentence-transformers`). Re-scoring Phase 5 with semantic embeddings is underway. If the sampler profile effect collapses under semantic scoring, the gains may reflect lexical diversity rather than genuine creative originality.

---

## Limitations (read before strong claims)

- **Hash embeddings in Phase 4–5.** Novelty scores were computed with `metric_backend=hash`, not semantic embeddings. Absolute novelty values are not semantically meaningful. Semantic re-scoring is pending.
- **No human calibration yet.** Appropriateness and usefulness are automatic proxies. The `prepare-human-slice` / `eval-human` pipeline exists but has not been run. Without human anchor points, claims about appropriateness magnitude are provisional.
- **Word-level POS checks** in CDAT appropriateness are pattern-based, not full linguistic analysis.
- **Aggregate masks heterogeneity.** Summary statistics over all 6 models obscure model-family effects that are large in magnitude (Gemma +0.096 vs Mistral-instruct −0.033 under the same profile).
- **Compute environment not controlled across phases.** Phase 3 ran on MPS; Phase 4–5 on CUDA A100. Cross-phase comparisons are directional only.

---

## What is implemented

- **Run ledger**: reproducible run records with immutable manifests (`run_id`, git hash, model hash, backend, quantization, timestamps).
- **Tasks**: `dat`, `cdat`, `aut`.
- **Methods**: `one_shot`, `best_of_k_one_shot`, `restlessness_best`, `restlessness_last_iter`, `restlessness_adaptive`, `brainstorm_then_select`.
- **Sampler profiles**: `default_nucleus`, `anti_repetition`, `spread_topk_minp`, `mirostat`, `high_temp`, `low_temp` + Phase 6 ablation profiles.
- **Scoring**: novelty, appropriateness, usefulness proxy; `frontier_objective = sqrt(novelty × appropriateness)`.
- **Analysis**: frontier plots, bootstrap CIs, base-vs-instruct shifts, best-of-N under token budget, paired prompt-level deltas, compute-matched summaries, efficiency tables.
- **Sampler analysis**: per-profile means, paired deltas vs baseline, Pareto summary.
- **Per-model breakdown**: `scripts/analyze_per_model.py` — family-level and base/instruct splits.
- **Homogeneity audit**: nearest-neighbor similarity, compactness, self-BLEU, diversity index.
- **Human calibration**: stratified rating slices (`prepare-human-slice`) and calibration gate (`eval-human`).
- **Cross-backend comparison**: `compare-backends` for MPS vs CUDA parity checks.
- **Operational guards**: runtime health gate, cell quarantine, session lineage.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[llama,data,semantic]'   # includes sentence-transformers
```

---

## Quick start

```bash
# Validate pipeline (no weights needed)
creativeai generate --task cdat --method one_shot --model gemma-2b \
  --backend mock --seed 11 --sampler-profile anti_repetition \
  --strict-json --cue forest --output-dir outputs/test/runs

creativeai score --input outputs/test/runs/runs.jsonl --output-dir outputs/test/scores
creativeai analyze-samplers --scores outputs/test/scores/scores.jsonl \
  --baseline-profile default_nucleus --output-dir outputs/test/analysis
```

---

## Re-score with semantic embeddings

```bash
CREATIVEAI_EMBEDDING_BACKEND=semantic \
  creativeai score \
  --input outputs/phase5_cuda_confirm/runs/runs.jsonl \
  --output-dir outputs/phase5_semantic/scores \
  --no-append-scores

CREATIVEAI_EMBEDDING_BACKEND=semantic \
  creativeai analyze-samplers \
  --scores outputs/phase5_semantic/scores/scores.jsonl \
  --baseline-profile default_nucleus \
  --output-dir outputs/phase5_semantic/analysis
```

---

## CLI reference

| Command | Role |
|---|---|
| `creativeai generate` | Single configuration run |
| `creativeai generate-grid` | Cartesian product over tasks, methods, models, temperatures, seeds, sampler profiles |
| `creativeai score` | Score a `runs.jsonl` stream |
| `creativeai analyze-frontier` | Frontier, shift, best-of-N, paired and compute-matched summaries |
| `creativeai analyze-samplers` | Per-profile means and paired deltas vs baseline |
| `creativeai audit-homogeneity` | Diversity / similarity audit |
| `creativeai compare-backends` | Cross-backend trend comparison |
| `creativeai prepare-human-slice` | Stratified slice for human rating |
| `creativeai eval-human` | Human rating evaluation |

---

## Output contract

| Path | Contents |
|---|---|
| `outputs/runs/runs.jsonl` | Append-only run log |
| `outputs/scores/scores.jsonl` | Append-only score log |
| `outputs/analysis/frontier_analysis.json` | Frontier, shift, best-of-N summary |
| `outputs/analysis/sampler_analysis.json` | Per-profile means and paired deltas |
| `outputs/analysis/per_model_sampler.json` | Per-model sampler breakdown |
| `outputs/analysis/homogeneity_audit.json` | Homogeneity summary |

---

## Repo layout

```
creativeai/           Python package — all core logic
  cli.py              CLI entry point
  pipeline.py         generate_run / scoring loop
  methods.py          one_shot, restlessness, best_of_k, brainstorm
  analysis.py         frontier, homogeneity, compare-backends, sampler analysis
  decoding.py         sampler profiles (incl. Phase 6 ablation profiles)
  scoring.py          novelty, appropriateness, usefulness, frontier_objective
  model_backend.py    llama_cpp / mock / transformers adapters
configs/              JSON grids per phase
scripts/              Shell/Python drivers per phase
  analyze_per_model.py   Per-model sampler breakdown
  stage_augie.sh         Laptop→Augie staging
  run_phase5_augie.sh    Phase 5 SLURM job
  run_phase6_augie.sh    Phase 6 SLURM job (ablation + combined)
north-star.md         Quick project brief for AI agents
outputs/              All run artifacts (large runs gitignored)
```

---

## Contact

**Shivam Arora**, Villanova University. Reach out via GitHub issues for bugs, reproducibility questions, or collaboration.
