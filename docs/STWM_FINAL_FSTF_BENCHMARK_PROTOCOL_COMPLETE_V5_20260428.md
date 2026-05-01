# STWM Final FSTF Benchmark Protocol Complete V5 20260428

## Definition
- STWM-FSTF: Future Semantic Trace Field Prediction

## Input / Output
- input: observed video-derived trace + observed semantic memory
- output: future trace field / trace units + future semantic prototype field + visibility / reappearance + identity belief

## Protocol
- free_rollout_requirement: `True`
- val-only selection; test-once evaluation
- changed subset and stable subset both required
- trace regression guardrail required
- terminology: `semantic trace-unit field`
