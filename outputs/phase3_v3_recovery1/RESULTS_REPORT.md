# Phase 3 v3 Recovery1 Results Report

## Run Identity
- Session: `phase3-1772324612`
- Runs completed: `376`
- Stage counts: `micro=128`, `main=144`, `confirm=96`, `aux=8`
- Runs file: `/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase3_v3_recovery1/runs/runs.jsonl`
- Scores file: `/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase3_v3_recovery1/scores/full/scores.jsonl`
- Analysis: `/Users/shivamarora/Documents/Programs/CreativeAI/outputs/phase3_v3_recovery1/analysis/phase3_full/frontier_analysis.json`

## Reliability / Data Quality
- JSON validity: `100%` (`376/376`)
- Primary-valid rate: `99.73%` (`375/376`)
- Invalid records: `1` (duplicate ideas in one AUT sample)
- Quarantined cells: `0`
- Health gate events: `0`

## Core Findings

### 1. Restlessness vs One-Shot (not compute-matched)
- Paired delta (objective): `+0.02095`
- 95% CI: `[+0.01461, +0.02724]`
- Pairs: `116`
- Interpretation: strong positive signal that restlessness beats one-shot when compute is not matched.

### 2. Compute-matched: Restlessness vs Best-of-k (primary fairness test)
- Paired delta (objective): `+0.00447`
- 95% CI: `[-0.00233, +0.01060]`
- Pairs: `36` matched, `75` unmatched under token tolerance
- Task-level:
  - AUT: CI crosses 0
  - CDAT: CI crosses 0
- Interpretation: no statistically secure advantage over compute-matched control in this run.

### 3. Base vs Instruct Shift
- Instruct variants are generally higher objective on AUT and CDAT for both Qwen and Mistral families.
- Direction is consistent with stronger appropriateness/constraint-following behavior.

### 4. Homogeneity (CDAT + AUT)
- No diversity collapse for `restlessness_best` vs `one_shot` on headline tasks.
- In pooled/task-stratified audits, `restlessness_best` is comparable or better on diversity index for CDAT/AUT.
- DAT aux slice is too small (`n=4`) and unstable; keep DAT auxiliary only.

## Acceptance Criteria Check
1. **Primary claim (restlessness beats one_shot and best_of_k with CI excluding 0): FAIL**
- Pass vs one_shot: yes
- Pass vs best_of_k (compute-matched): no (CI includes 0)

2. **Reliability invalid rate <= 5%: PASS**
- Invalid rate: `0.27%`

3. **Compute fairness reported: PASS**
- Compute-matched analysis included and reported.

4. **Homogeneity non-inferiority: PASS (for CDAT/AUT)**
- No unacceptable collapse observed on headline tasks.

5. **External-family direction check (confirm subset): PARTIAL PASS**
- Trend vs one-shot generalizes directionally.
- Compute-matched superiority still unproven.

## Decision
- **Do not claim compute-matched superiority yet.**
- **Claimable now:** robust reliability stack + consistent improvement over one-shot + no major CDAT/AUT homogeneity collapse.
- **Not claimable now:** restlessness is better than compute-matched best-of-k.

## Important Implementation Note
- Pairing logic was hardened to include replicate identity (`compute_group_id`) so paired statistics no longer collapse seed/temperature replicates.
- For analysis consistency, use:
  - `python -m creativeai.cli analyze-frontier ...`
  - (not the `creativeai` entrypoint, which showed stale behavior in this environment)
