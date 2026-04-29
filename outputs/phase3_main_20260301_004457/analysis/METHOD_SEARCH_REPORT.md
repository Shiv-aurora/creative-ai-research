# Method Search Report

- Status: **NO_GO**
- Baseline: `naive_multiturn` (unstructured multi-turn)
- Anchor: `restlessness_best` (current working method)

## Method Means
- naive_multiturn: n=190 objective=0.5959 tokens=996.4 calls=5.41
- one_shot: n=188 objective=0.5981 tokens=228.1 calls=1.70
- restlessness_best: n=190 objective=0.6263 tokens=1233.8 calls=6.27
- restlessness_triggered: n=189 objective=0.6138 tokens=456.7 calls=2.89

## Candidate Gates
- restlessness_triggered: pairs=188 obj_delta=0.0173 [0.0108,0.0254] token_reduction=0.450 [0.361,0.532] call_reduction=0.491 [0.433,0.541] pass=False

## Recommendation
- No method passed gate; retune and rerun micro-lab.

Raw JSON: `outputs/phase3_main_20260301_004457/analysis/method_search_decision.json`
