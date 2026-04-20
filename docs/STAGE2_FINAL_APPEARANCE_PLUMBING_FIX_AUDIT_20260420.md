# Stage2 Final Appearance Plumbing Fix Audit 20260420

- offline_appearance_drift_high_ratio: 0.2
- dataloader_appearance_drift_high_ratio: 0.2
- batch_appearance_drift_high_ratio_mean: 0.0
- appearance_refine_loss_nonzero_ratio: 0.0
- exact_breakpoint: signal present offline but not activated in batch/loss path; threshold/plumbing issue remains before loss becomes nonzero
