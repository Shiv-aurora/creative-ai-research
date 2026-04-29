# Method Search Report

- Status: **GO**
- Baseline: `naive_multiturn` (unstructured multi-turn)
- Anchor: `restlessness_best` (current working method)

## Method Means
- naive_multiturn: n=16 objective=0.6426 tokens=965.1 calls=4.69
- restlessness_best: n=16 objective=0.6620 tokens=1036.8 calls=5.12
- restlessness_triggered: n=16 objective=0.6498 tokens=254.6 calls=1.75

## Candidate Gates
- restlessness_triggered: pairs=16 obj_delta=0.0072 [-0.0054,0.0200] token_reduction=0.689 [0.545,0.797] call_reduction=0.637 [0.539,0.711] pass=True

## Recommendation
- Promote `restlessness_triggered` to next stage.

Raw JSON: `outputs/phase3_method_search_v4/analysis/method_search_decision.json`
