# Method Search Report

- Status: **GO**
- Baseline: `naive_multiturn` (unstructured multi-turn)
- Anchor: `restlessness_best` (current working method)

## Method Means
- naive_multiturn: n=63 objective=0.6349 tokens=1117.3 calls=4.86
- one_shot: n=64 objective=0.6377 tokens=172.8 calls=1.25
- restlessness_best: n=64 objective=0.6542 tokens=1052.9 calls=5.09
- restlessness_triggered: n=64 objective=0.6423 tokens=205.5 calls=1.47

## Candidate Gates
- restlessness_triggered: pairs=63 obj_delta=0.0075 [0.0015,0.0138] token_reduction=0.767 [0.722,0.812] call_reduction=0.697 [0.663,0.729] pass=True

## Recommendation
- Promote `restlessness_triggered` to next stage.

Raw JSON: `outputs/phase3_confirm_local_v1/analysis/method_search_decision.json`
