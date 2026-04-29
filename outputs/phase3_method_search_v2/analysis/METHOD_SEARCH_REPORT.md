# Method Search Report

- Status: **NO_GO**
- Baseline: `naive_multiturn` (unstructured multi-turn)
- Anchor: `restlessness_best` (current working method)

## Method Means
- naive_multiturn: n=16 objective=0.6426 tokens=965.1 calls=4.69
- restlessness_best: n=16 objective=0.6620 tokens=1036.8 calls=5.12
- restlessness_triggered: n=16 objective=0.6393 tokens=196.4 calls=1.38

## Candidate Gates
- restlessness_triggered: pairs=16 obj_delta=-0.0032 [-0.0179,0.0111] token_reduction=0.777 [0.728,0.825] call_reduction=0.713 [0.680,0.745] pass=False

## Recommendation
- No method passed gate; retune and rerun micro-lab.

Raw JSON: `outputs/phase3_method_search_v2/analysis/method_search_decision.json`
