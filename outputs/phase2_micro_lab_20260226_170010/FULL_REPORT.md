# Full Report: Phase-2 Micro-Lab v2 (768 Runs)

## Executive Summary

- Completed **768/768** runs on local MPS via `llama_cpp` across **4 model variants**, **4 inference-time methods**, **2 task families**, and **2 seeds**.
- Preflight passed for all primary models before the main run: [preflight_summary.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/preflight/preflight_summary.json)
- Reliability was strong in the hardened pipeline:
  - **JSON-valid rate: 100%**
  - **Primary-valid rate: 98.18%**
  - **No health-gate quarantine events**
- Main scientific result:
  - `restlessness_best` beat `one_shot` on the paired frontier objective by **+0.0333** with **95% CI [0.0238, 0.0436]** over **96 prompt-matched pairs**.
  - This improvement appeared on both task families: **AUT +0.0354**, **CDAT +0.0313**.
- Critical caveat:
  - Under a **compute-matched** comparison, `restlessness_best` did **not** beat `best_of_k_one_shot`.
  - Compute-matched delta was **-0.0009** with **95% CI [-0.0088, 0.0078]** over **32 pairs**.
- Stronger interpretation of this run:
  - The data support that **inference-time search helps relative to one-shot**.
  - The data do **not** support a strong claim that **restlessness is better than a compute-matched search baseline**.
- Model-level pattern:
  - The **Qwen instruct** model was strongest overall.
  - Instruction-tuned variants outperformed base variants across both families in this micro-lab.
- Diversity / homogeneity result:
  - `restlessness_best` did **not** collapse population diversity in this run.
  - In pooled homogeneity, it was actually the **best** method by diversity index.

## Experiment Configuration

- Run window (UTC): `2026-02-26T22:05:25.000527+00:00` to `2026-02-27T03:31:55.131032+00:00`
- Duration: **5.44 hours**
- Main session id: `sess-1772143513-5be22633`
- Models: `gemma-2b`, `gemma-2b-it`, `qwen2.5-3b`, `qwen2.5-3b-instruct`
- Methods: `one_shot`, `best_of_k_one_shot`, `restlessness_best`, `restlessness_last_iter`
- Tasks: `cdat`, `aut`
- Temperature(s): `[0.7]`
- Seed(s): `[11, 37]`
- Prompt counts per cell: `12 CDAT + 12 AUT = 24`
- Backend: `llama_cpp` on local MPS
- Strict JSON: enabled
- Health gate: enabled with `window=20`, `json>=0.95`, `valid>=0.90`, action=`quarantine_cell`

## Coverage Check

Every model-method-task cell completed at full prompt coverage.

| Model | Method | CDAT n | AUT n | Total |
|---|---:|---:|---:|---:|
| gemma-2b | one_shot | 24 | 24 | 48 |
| gemma-2b | best_of_k_one_shot | 24 | 24 | 48 |
| gemma-2b | restlessness_best | 24 | 24 | 48 |
| gemma-2b | restlessness_last_iter | 24 | 24 | 48 |
| gemma-2b-it | one_shot | 24 | 24 | 48 |
| gemma-2b-it | best_of_k_one_shot | 24 | 24 | 48 |
| gemma-2b-it | restlessness_best | 24 | 24 | 48 |
| gemma-2b-it | restlessness_last_iter | 24 | 24 | 48 |
| qwen2.5-3b | one_shot | 24 | 24 | 48 |
| qwen2.5-3b | best_of_k_one_shot | 24 | 24 | 48 |
| qwen2.5-3b | restlessness_best | 24 | 24 | 48 |
| qwen2.5-3b | restlessness_last_iter | 24 | 24 | 48 |
| qwen2.5-3b-instruct | one_shot | 24 | 24 | 48 |
| qwen2.5-3b-instruct | best_of_k_one_shot | 24 | 24 | 48 |
| qwen2.5-3b-instruct | restlessness_best | 24 | 24 | 48 |
| qwen2.5-3b-instruct | restlessness_last_iter | 24 | 24 | 48 |

## Reliability And Validity Check

### Pipeline Health

- Preflight pass rate: **4/4 models passed**
- Main-run health events: **0**
- Quarantined cells: **0**
- JSON-valid outputs: **768/768 = 100.00%**
- Primary-valid outputs: **754/768 = 98.18%**

### Validity Losses

- Problems were concentrated in **schema failures**, not parse failures.
- Observed problem type: **`duplicate_items` only**
- Total duplicate-item failures: **14**
- Residual invalids were concentrated in the Gemma pair; both Qwen variants were fully primary-valid.

Primary-valid rate by method:

| Method | Valid / Total | Rate |
|---|---:|---:|
| one_shot | 185 / 192 | 96.35% |
| best_of_k_one_shot | 187 / 192 | 97.40% |
| restlessness_best | 191 / 192 | 99.48% |
| restlessness_last_iter | 191 / 192 | 99.48% |

Primary-valid rate by model:

| Model | Valid / Total | Rate |
|---|---:|---:|
| gemma-2b | 184 / 192 | 95.83% |
| gemma-2b-it | 186 / 192 | 96.88% |
| qwen2.5-3b | 192 / 192 | 100.00% |
| qwen2.5-3b-instruct | 192 / 192 | 100.00% |

Interpretation: the hardening cycle worked. The remaining quality issue is now narrow and diagnosable rather than systemic.

## Primary Result: `restlessness_best` vs `one_shot`

Prompt-paired comparison from [frontier_analysis.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/frontier_analysis.json):

- Matched pairs: **96**
- Mean delta (restlessness - one-shot): **+0.0333**
- 95% CI: **[+0.0238, +0.0436]**
- Wins for `restlessness_best`: **63 / 96**

Task-level deltas:

| Task | Pair Count | Mean Delta | 95% CI | Wins |
|---|---:|---:|---:|---:|
| AUT | 48 | +0.0354 | [0.0207, 0.0506] | 30 |
| CDAT | 48 | +0.0313 | [0.0208, 0.0441] | 33 |

Interpretation: `restlessness_best` is clearly better than `one_shot` in this micro-lab on the aggregate frontier objective.

## Compute-Matched Result: `restlessness_best` vs `best_of_k_one_shot`

This is the critical fairness test.

- Matched pairs: **32**
- Mean delta (restlessness - best-of-k): **-0.0009**
- 95% CI: **[-0.0088, +0.0078]**
- Wins for `restlessness_best`: **12 / 32**

Task-level deltas:

| Task | Pair Count | Mean Delta | 95% CI | Wins |
|---|---:|---:|---:|---:|
| AUT | 25 | -0.0002 | [-0.0105, 0.0095] | 10 |
| CDAT | 7 | -0.0033 | [-0.0131, 0.0067] | 2 |

Interpretation: the micro-lab does **not** support the claim that restlessness has a method-specific advantage beyond compute budget. At this scale, it looks **competitive with** but **not superior to** a compute-matched search baseline.

## Baseline Search Result: `best_of_k_one_shot` vs `one_shot`

- Matched pairs: **47**
- Mean delta (best-of-k - one-shot): **+0.0284**
- 95% CI: **[+0.0173, +0.0420]**
- Wins for `best_of_k_one_shot`: **35 / 47**

Task-level deltas:

| Task | Pair Count | Mean Delta | 95% CI | Wins |
|---|---:|---:|---:|---:|
| AUT | 30 | +0.0358 | [0.0190, 0.0552] | 22 |
| CDAT | 17 | +0.0155 | [0.0089, 0.0223] | 13 |

Interpretation: a simple search baseline already provides a robust gain over one-shot. This strengthens the conclusion that the main effect in this run is **search**, not necessarily **restlessness-specific structure**.

## Frontier Summary By Model Family

Best method by `(model, task)` objective mean:

| Model | Task | Best Method | Objective | Novelty | Appropriateness |
|---|---|---|---:|---:|---:|
| gemma-2b | AUT | restlessness_best | 0.4983 | 0.4451 | 0.5791 |
| gemma-2b | CDAT | best_of_k_one_shot | 0.6213 | 0.6999 | 0.5543 |
| gemma-2b-it | AUT | restlessness_best | 0.6469 | 0.7090 | 0.6014 |
| gemma-2b-it | CDAT | restlessness_best | 0.6384 | 0.6366 | 0.6469 |
| qwen2.5-3b | AUT | best_of_k_one_shot | 0.6354 | 0.5401 | 0.7540 |
| qwen2.5-3b | CDAT | restlessness_best | 0.6477 | 0.6657 | 0.6340 |
| qwen2.5-3b-instruct | AUT | best_of_k_one_shot | 0.6761 | 0.6436 | 0.7174 |
| qwen2.5-3b-instruct | CDAT | best_of_k_one_shot | 0.6618 | 0.7160 | 0.6132 |

Top frontier cells overall:

1. `qwen2.5-3b-instruct / best_of_k_one_shot / AUT` = **0.6761**
2. `qwen2.5-3b-instruct / restlessness_best / AUT` = **0.6736**
3. `qwen2.5-3b-instruct / best_of_k_one_shot / CDAT` = **0.6618**
4. `qwen2.5-3b-instruct / one_shot / AUT` = **0.6586**
5. `qwen2.5-3b-instruct / restlessness_best / CDAT` = **0.6577**

Interpretation: the Qwen instruct model dominates this micro-lab. The Gemma instruct model is respectable, especially under `restlessness_best`, but clearly behind Qwen instruct.

## Base vs Instruct Shift

From [frontier_analysis.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/frontier_analysis.json):

| Family | Task | Delta Instruct - Base |
|---|---|---:|
| gemma-2b | CDAT | +0.0241 |
| gemma-2b | AUT | +0.1376 |
| qwen2.5-3b | CDAT | +0.0100 |
| qwen2.5-3b | AUT | +0.0470 |

Interpretation: in this run, instruction tuning helped rather than hurt the frontier objective, especially on AUT. This is opposite to a naive “alignment always suppresses creativity” story.

## Homogeneity Audit

Pooled homogeneity from [homogeneity_audit.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/homogeneity_audit.json):

| Method | Diversity Index | NN Similarity | Compactness | Self-BLEU |
|---|---:|---:|---:|---:|
| one_shot | 0.3275 | 0.8527 | 0.5522 | 0.5097 |
| best_of_k_one_shot | 0.3115 | 0.8593 | 0.5886 | 0.5260 |
| restlessness_best | 0.3539 | 0.8344 | 0.5782 | 0.4429 |
| restlessness_last_iter | 0.3289 | 0.8465 | 0.5672 | 0.5048 |

Task-stratified homogeneity from [by_task/homogeneity_audit.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/by_task/homogeneity_audit.json):

| Method | Task | Diversity Index |
|---|---|---:|
| one_shot | CDAT | 0.2975 |
| best_of_k_one_shot | CDAT | 0.2604 |
| restlessness_best | CDAT | 0.3278 |
| restlessness_last_iter | CDAT | 0.2974 |
| one_shot | AUT | 0.3259 |
| best_of_k_one_shot | AUT | 0.3234 |
| restlessness_best | AUT | 0.3427 |
| restlessness_last_iter | AUT | 0.3218 |

Interpretation:

- `best_of_k_one_shot` is the worst method on homogeneity in this run.
- `restlessness_best` is the best method on diversity index both pooled and by task.
- This directly weakens the hypothesis that creativity gains here are merely homogenization in disguise.

## What The Results Mean In Human Terms

1. The hardened experiment stack worked.
   - This matters because the previous recovery plan was about reliability first.
   - The run completed cleanly, strict JSON held, and bad cells never needed quarantine.

2. Search helps.
   - Both `best_of_k_one_shot` and `restlessness_best` beat `one_shot`.
   - So the system is finding real gain from extra inference-time exploration.

3. Restlessness is promising, but not yet uniquely justified.
   - It improves over one-shot.
   - But once you compare fairly against compute-matched `best_of_k_one_shot`, the advantage disappears.
   - That means the paper cannot honestly claim: “restlessness is better than simpler search” based on this run alone.

4. The strongest model result is not subtle.
   - `qwen2.5-3b-instruct` is the best overall model in this micro-lab.
   - If the next stage needs a reduced confirmatory set, this model should stay in it.

5. Instruction tuning did not destroy constrained creativity here.
   - In this run, instruct variants beat their base siblings on both task families.
   - The effect is especially strong for Gemma on AUT.

6. Diversity collapse is not the hidden failure mode here.
   - The homogeneity audit actually favors `restlessness_best` over `best_of_k_one_shot`.
   - So at least in this micro-lab, restlessness does not buy gains by making everything more same-y.

## Decision

### What We Can Claim Now

- The local-first hardened pipeline is operational and reliable.
- Inference-time search improves the novelty-appropriateness frontier over one-shot.
- `restlessness_best` is a viable search-style method.
- `qwen2.5-3b-instruct` is the strongest model in the tested set.
- Homogeneity is not obviously worsened by `restlessness_best` in this micro-lab.

### What We Cannot Claim Yet

- We cannot claim `restlessness_best` is better than a compute-matched `best_of_k_one_shot` baseline.
- We cannot claim publication-safe generality from this micro-lab alone.
- We cannot yet make strong external claims about human-rated creativity/usefulness without calibration.
- We cannot yet make hardware-stable claims without CUDA confirmation.

## Publication Readiness

Current recommendation: **do not publish this as a main algorithm paper yet**.

This run is strong enough for:

- an internal memo
- a pilot report
- a methods-and-evaluation note
- a foundation for the larger confirmatory local study

This run is not yet strong enough for a claim of:

- “restlessness is a better creativity method than standard search”

The current evidence supports a narrower and more defensible framing:

- **Creativity under constraints benefits from inference-time search; compute matching is essential; restlessness is competitive and diversity-safe but not yet superior to simple search.**

## Recommended Next Steps

1. Run the larger Phase 3 local study with the hardened stack unchanged.
2. Keep `qwen2.5-3b-instruct` and `qwen2.5-3b` as mandatory models.
3. Keep the Gemma pair for contrast, but treat them as secondary evidence rather than strongest-model evidence.
4. In the next stage, center the primary comparison on:
   - `one_shot`
   - `best_of_k_one_shot`
   - `restlessness_best`
5. Make the compute-matched test a headline analysis, not an appendix analysis.
6. Run human calibration before any external claim about usefulness/appropriateness quality.
7. Run CUDA confirmation on a frozen subset before any claim of cross-hardware robustness.

## Artifact Index

- Full runs: [runs.jsonl](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/runs/runs.jsonl)
- Scores: [scores.jsonl](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/scores/scores.jsonl)
- Frontier analysis: [frontier_analysis.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/frontier_analysis.json)
- Frontier plot: [frontier.png](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/frontier.png)
- Pooled homogeneity: [homogeneity_audit.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/homogeneity_audit.json)
- By-task homogeneity: [by_task/homogeneity_audit.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/analysis/by_task/homogeneity_audit.json)
- Preflight summary: [preflight_summary.json](/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase2_micro_lab_20260226_170010/preflight/preflight_summary.json)
