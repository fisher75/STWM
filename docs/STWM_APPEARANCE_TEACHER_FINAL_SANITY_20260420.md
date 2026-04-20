# STWM Appearance + Teacher Final Sanity 20260420

## Appearance Plumbing
- appearance_claim_allowed: `False`
- offline_appearance_drift_high_ratio: `0.2`
- batch_appearance_drift_high_ratio_mean: `0.0`
- appearance_refine_loss_nonzero_ratio: `0.0`
- exact_breakpoint: signal present offline but not activated in batch/loss path; threshold/plumbing issue remains before loss becomes nonzero

## Teacher Sanity
- teacher_prior_upgrade_available: `False`
- best_available_teacher: `clip_vit-b_16_temporal_weighted_masked_mean_v5_driftcal`
- exact_blocking_reason: no stronger frozen teacher cache was materialized for a clean small-scale retrieval sanity inside this pass
