# Method Search Report

- Status: **NO_GO**
- Baseline: `naive_multiturn` (unstructured multi-turn)
- Anchor: `restlessness_best` (current working method)

## Method Means
- naive_multiturn: n=16 objective=0.6426 tokens=965.1 calls=4.69
- one_shot: n=16 objective=0.6393 tokens=196.4 calls=1.38
- restlessness_adaptive: n=16 objective=0.6614 tokens=538.4 calls=3.31
- restlessness_best: n=16 objective=0.6620 tokens=1036.8 calls=5.12
- restlessness_triggered: n=16 objective=0.6560 tokens=380.6 calls=2.44

## Candidate Gates
- restlessness_triggered: pairs=16 obj_delta=0.0135 [0.0013,0.0247] token_reduction=0.489 [0.238,0.714] call_reduction=0.507 [0.351,0.656] pass=False
- restlessness_adaptive: pairs=16 obj_delta=0.0189 [0.0069,0.0302] token_reduction=0.286 [0.069,0.490] call_reduction=0.306 [0.197,0.406] pass=False
- restlessness_best: pairs=16 obj_delta=0.0195 [0.0075,0.0311] token_reduction=-0.274 [-0.522,-0.028] call_reduction=-0.090 [-0.176,-0.014] pass=False

## Recommendation
- No method passed gate; retune and rerun micro-lab.

Raw JSON: `outputs/phase3_method_search_v1/analysis/method_search_decision.json`
