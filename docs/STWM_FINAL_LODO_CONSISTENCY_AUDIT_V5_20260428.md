# STWM Final LODO Consistency Audit V5 20260428

## Status
- lodo_completed: `True`
- expected_run_count: `20`
- completed_run_count: `20`
- failed_run_count: `0`
- trusted_lodo_conclusion: `negative`
- codex_prose_both_negative_consistent_with_json: `True`

## Direction Summary
- VSPW->VIPSeg changed delta: `-0.0348`; CI `[-0.05916584851976654, -0.02511738734130705]`
- VIPSeg->VSPW changed delta: `-0.0308`; CI `[-0.07454895165728198, -0.01974344670813944]`

## Interpretation
- LODO is completed and negative in both directions. This is a cross-dataset domain-shift limitation, not evidence that the mixed free-rollout result is invalid.
