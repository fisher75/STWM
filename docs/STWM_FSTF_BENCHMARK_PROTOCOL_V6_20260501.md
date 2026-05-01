# STWM FSTF Benchmark Protocol V6 20260501

## Task
- STWM-FSTF: Future Semantic Trace Field Prediction

## Input / Output
- input: observed video-derived trace + observed semantic memory
- output: future trace units + future semantic prototype field + visibility / reappearance / identity belief

## Protocol
- free_rollout_required: `True`
- candidate_scorer_used: `False`
- future_candidate_leakage: `False`
- terminology: `semantic trace-unit field`

## Main Table Contract
- mixed, VSPW, VIPSEG, stable/changed split, copy delta, same-output baselines
