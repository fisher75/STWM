# STWM OSTF V34.5 V34.4 Residual Objective Audit

- residual_supervised_as_standalone_target: `True`
- final_semantic_is_pointwise_plus_residual: `True`
- residual_direct_loss_aligned_with_final_composition: `False`
- delta_residual_objective_missing: `True`
- force_gate_one_hurts_due_residual_content: `True`
- oracle_fail_is_borderline: `True`
- residual_positive_ratio_too_broad: `True`
- semantic_hard_signal_failed_despite_changed_signal_positive: `True`
- identity_auc_oracle_only: `True`
- recommended_fix: `Train unit_semantic_residual as an orthogonal delta correction over frozen pointwise_semantic_belief, and narrow residual utility positives with split-specific confidence/error quantiles.`
