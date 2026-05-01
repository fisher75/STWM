# STWM Final Artifact Consistency Reconcile V6 20260501

## Mixed Fullscale Main Result
- expected_checkpoint_count: `10`
- existing_checkpoint_count: `10`
- existing_train_report_count: `10`
- existing_log_count: `10`
- empty_log_count: `10`

## Selected C
- canonical_selected_C: `32`
- canonical_selected_seed: `456`
- stale_conflicting_paths: `['/raid/chen034/workspace/stwm/reports/stwm_mixed_fullscale_v2_val_selection_20260428.json', '/raid/chen034/workspace/stwm/reports/stwm_mixed_fullscale_v2_decision_20260428.json']`

## Claim Status
- can_claim_5seed_main_result: `True`
- can_claim_selected_C: `True`
- can_claim_baseline_complete: `False`
- can_claim_scaling_complete: `False`

## Note
- Mixed fullscale main-result artifacts are real. The main inconsistency comes from older non-complete reports that still point to C64 before the full 10-candidate val-only selection converged to C32 seed456.
