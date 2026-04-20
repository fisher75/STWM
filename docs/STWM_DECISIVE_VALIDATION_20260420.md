# STWM Decisive Validation 20260420

## Summary
- matched_6seed_improved: `False`
- killer_baselines_passed: `False`
- strict_bootstrap_claim_level: `moderate_claim`
- utility_v3_claim_ready: `True`
- true_ood_claim_ready: `False`
- mechanism_claim_ready: `False`
- appearance_claim_allowed: `False`
- paper_target_recommendation: `borderline_needs_one_last_fix`
- oral_spotlight_readiness: `not_ready`
- next_step_choice: `run_one_last_surgical_fix`

## Blocking Reasons
- matched_6seed: missing matched checkpoints for TUSB-v3.1/cropenc/legacysem prevent a real 6-seed judge; current evidence remains partial-seed only
- killer_baselines: teacher-only and object-slot/no-trace killer baselines are not materialized; trace-only TUSB is only available for seed123
- ood: dataset-split proxy is positive, but conservative held-out OOD split is not materialized; do not upgrade to true OOD claim
- mechanism: 6-seed mechanism claim remains blocked by missing seeds 654/789/321
- appearance: signal present offline but not activated in batch/loss path; threshold/plumbing issue remains before loss becomes nonzero
