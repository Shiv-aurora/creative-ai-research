# Full Report: Phase-1 Real MPS Batch (468 Runs)

## Executive Summary

- Completed **468/468** runs on local MPS (`llama_cpp`) across 6 model variants, 2 methods, and 3 task families.
- Main finding: `restlessness_loop` outperformed `one_shot` in **18/18** matched model-task cells (losses: 0).
- Average objective improvement across matched cells: **0.0501**.
- Strongest average gains are in CDAT-like conditional creativity, suggesting better novelty-under-constraints behavior.
- Caveat: population-level homogeneity worsened slightly for `restlessness_loop` in this full run, so diversity tradeoff must be reported.

## Experiment Configuration

- Run window (UTC): `2026-02-25T20:15:09.291277+00:00` to `2026-02-26T01:40:13.349786+00:00` (duration: `5:25:04.058509`).
- Models (6): `gemma-2b, gemma-2b-it, mistral-7b-instruct-v0.3, mistral-7b-v0.3, qwen2.5-3b, qwen2.5-3b-instruct`
- Methods: `one_shot, restlessness_loop`
- Tasks: `aut, cdat, dat`
- Temperature(s): `[0.7]`; seed(s): `[11]`; backend(s): `['llama_cpp']`; max_tokens: `[256]`
- Dataset limits used in this batch: DAT=1 prompt per cell, CDAT=20 cues per cell, AUT=18 prompts per cell.

## Coverage Check

| Model | Method | DAT n | CDAT n | AUT n | Total |
|---|---:|---:|---:|---:|---:|
| gemma-2b | one_shot | 1 | 20 | 18 | 39 |
| gemma-2b | restlessness_loop | 1 | 20 | 18 | 39 |
| gemma-2b-it | one_shot | 1 | 20 | 18 | 39 |
| gemma-2b-it | restlessness_loop | 1 | 20 | 18 | 39 |
| mistral-7b-instruct-v0.3 | one_shot | 1 | 20 | 18 | 39 |
| mistral-7b-instruct-v0.3 | restlessness_loop | 1 | 20 | 18 | 39 |
| mistral-7b-v0.3 | one_shot | 1 | 20 | 18 | 39 |
| mistral-7b-v0.3 | restlessness_loop | 1 | 20 | 18 | 39 |
| qwen2.5-3b | one_shot | 1 | 20 | 18 | 39 |
| qwen2.5-3b | restlessness_loop | 1 | 20 | 18 | 39 |
| qwen2.5-3b-instruct | one_shot | 1 | 20 | 18 | 39 |
| qwen2.5-3b-instruct | restlessness_loop | 1 | 20 | 18 | 39 |

## Primary Result: Method Comparison (`restlessness_loop` vs `one_shot`)

- Matched cells compared: **18**
- Wins for `restlessness_loop`: **18**
- Losses for `restlessness_loop`: **0**

### Task-Level Mean Deltas (Restlessness - One-shot)

| Task | Pair Count | Δ Objective | Δ Novelty | Δ Appropriateness |
|---|---:|---:|---:|---:|
| dat | 6 | 0.0247 | 0.0468 | 0.0000 |
| cdat | 6 | 0.0873 | -0.0213 | 0.0657 |
| aut | 6 | 0.0384 | 0.0770 | 0.0013 |

### Cell-Level Deltas

| Model | Task | One-shot Obj | Restlessness Obj | Δ Obj | Δ Nov | Δ App | n(one/rest) |
|---|---|---:|---:|---:|---:|---:|---:|
| gemma-2b | aut | 0.6388 | 0.6836 | 0.0448 | 0.0500 | 0.0363 | 18/18 |
| gemma-2b | cdat | 0.2536 | 0.3734 | 0.1198 | -0.0126 | 0.0826 | 20/20 |
| gemma-2b | dat | 0.9415 | 0.9555 | 0.0140 | 0.0265 | 0.0000 | 1/1 |
| gemma-2b-it | aut | 0.6500 | 0.6788 | 0.0288 | 0.0353 | 0.0212 | 18/18 |
| gemma-2b-it | cdat | 0.3769 | 0.4987 | 0.1219 | -0.0837 | 0.1508 | 20/20 |
| gemma-2b-it | dat | 0.9808 | 0.9952 | 0.0144 | 0.0285 | 0.0000 | 1/1 |
| mistral-7b-instruct-v0.3 | aut | 0.6391 | 0.6826 | 0.0435 | 0.1273 | -0.0387 | 18/18 |
| mistral-7b-instruct-v0.3 | cdat | 0.3264 | 0.3416 | 0.0152 | -0.0010 | 0.0063 | 20/20 |
| mistral-7b-instruct-v0.3 | dat | 0.9868 | 0.9928 | 0.0060 | 0.0119 | 0.0000 | 1/1 |
| mistral-7b-v0.3 | aut | 0.6370 | 0.6819 | 0.0449 | 0.0838 | 0.0142 | 18/18 |
| mistral-7b-v0.3 | cdat | 0.3102 | 0.3804 | 0.0702 | 0.0027 | 0.0434 | 20/20 |
| mistral-7b-v0.3 | dat | 0.9896 | 0.9943 | 0.0047 | 0.0093 | 0.0000 | 1/1 |
| qwen2.5-3b | aut | 0.6378 | 0.6662 | 0.0284 | 0.0659 | -0.0081 | 18/18 |
| qwen2.5-3b | cdat | 0.2366 | 0.3766 | 0.1400 | -0.0359 | 0.0899 | 20/20 |
| qwen2.5-3b | dat | 0.8856 | 0.9895 | 0.1039 | 0.1947 | 0.0000 | 1/1 |
| qwen2.5-3b-instruct | aut | 0.6469 | 0.6869 | 0.0400 | 0.0996 | -0.0170 | 18/18 |
| qwen2.5-3b-instruct | cdat | 0.2109 | 0.2673 | 0.0564 | 0.0026 | 0.0212 | 20/20 |
| qwen2.5-3b-instruct | dat | 0.9918 | 0.9969 | 0.0051 | 0.0102 | 0.0000 | 1/1 |

## Method Means by Task (Across Models)

| Task | One-shot Mean Obj | Restlessness Mean Obj |
|---|---:|---:|
| dat | 0.9627 | 0.9874 |
| cdat | 0.2857 | 0.3730 |
| aut | 0.6416 | 0.6800 |

## Base vs Instruct Shift (Instruct - Base)

| Family | Task | Δ Objective (Instruct-Base) |
|---|---|---:|
| gemma-2b | aut | 0.0032 |
| gemma-2b | cdat | 0.1243 |
| gemma-2b | dat | 0.0395 |
| mistral-7b-v0.3 | aut | 0.0014 |
| mistral-7b-v0.3 | cdat | -0.0113 |
| mistral-7b-v0.3 | dat | -0.0021 |
| qwen2.5-3b | aut | 0.0148 |
| qwen2.5-3b | cdat | -0.0675 |
| qwen2.5-3b | dat | 0.0568 |

## Homogeneity Audit

| Method | Sample Count | Diversity Index | Nearest-Neighbor Sim | Compactness | Self-BLEU |
|---|---:|---:|---:|---:|---:|
| one_shot | 234 | 0.5067 | 0.6507 | 0.5553 | 0.2555 |
| restlessness_loop | 234 | 0.4898 | 0.6665 | 0.5706 | 0.2747 |

Interpretation: In this full batch, `restlessness_loop` improves objective but slightly worsens aggregate homogeneity metrics (lower diversity index), indicating a measurable tradeoff to report.

## Top and Bottom Frontier Cells

### Top 5 by Objective

| Model | Method | Task | Objective | n |
|---|---|---|---:|---:|
| qwen2.5-3b-instruct | restlessness_loop | dat | 0.9969 | 1 |
| gemma-2b-it | restlessness_loop | dat | 0.9952 | 1 |
| mistral-7b-v0.3 | restlessness_loop | dat | 0.9943 | 1 |
| mistral-7b-instruct-v0.3 | restlessness_loop | dat | 0.9928 | 1 |
| qwen2.5-3b-instruct | one_shot | dat | 0.9918 | 1 |

### Bottom 5 by Objective

| Model | Method | Task | Objective | n |
|---|---|---|---:|---:|
| qwen2.5-3b-instruct | one_shot | cdat | 0.2109 | 20 |
| qwen2.5-3b | one_shot | cdat | 0.2366 | 20 |
| gemma-2b | one_shot | cdat | 0.2536 | 20 |
| qwen2.5-3b-instruct | restlessness_loop | cdat | 0.2673 | 20 |
| mistral-7b-v0.3 | one_shot | cdat | 0.3102 | 20 |

## Human Interpretation

1. The core hypothesis is supported: inference-time restlessness improves creativity-under-constraints objective consistently across model-task cells in this run.
2. The strongest evidence is in CDAT-style conditional tasks, where objective gains are driven mainly by appropriateness gains.
3. AUT also improves on objective, largely via novelty gains, with near-flat appropriateness.
4. The approach is not free: homogeneity trends show a slight downside at population level, so "more creative" is not automatically "more diverse across samples."

## Limitations

1. DAT cells have n=1 per model-method cell in this run; treat DAT conclusions as directional only.
2. Metrics are automatic proxies; external or human calibration is still needed before strong claim language.
3. This report covers MPS-only execution; CUDA parity still pending for universality claims.

## Recommended Next Steps

1. Run a CUDA/A100 replication on a frozen subset to test trend stability (same models/methods/seeds/prompts).
2. Add a targeted human calibration slice for CDAT/AUT outputs to validate appropriateness/usefulness proxies.
3. Tune restlessness constraints to reduce homogeneity regression while retaining objective gains.

## Artifact Index

- Runs: `outputs/phase1_real_batch1/runs/runs.jsonl`
- Scores: `outputs/phase1_real_batch1/scores/scores.jsonl`
- Frontier JSON: `outputs/phase1_real_batch1/analysis/frontier_analysis.json`
- Frontier plot: `outputs/phase1_real_batch1/analysis/frontier.png`
- Homogeneity JSON: `outputs/phase1_real_batch1/analysis/homogeneity_audit.json`
