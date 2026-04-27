# STWM World Model Upgrade V3 Reality Check 20260427

- smalltrain_train_steps: `4`
- future_semantic_state_loss_weights_nonzero: `True`
- future_semantic_state_losses_finite: `True`
- checkpoint_contains_future_semantic_state_head: `True`
- export_reads_v2_smalltrain_checkpoint: `True`
- export_has_required_raw_fields: `False`
- export_forward_scope: `future_semantic_state_head_checkpoint_forward_with_manifest_surrogate_features`
- export_full_stage1_stage2_forward_executed: `False`
- consuming_eval_reads_export_not_old_association_report: `True`
- trace_rollout_regression_detected: `False`
- free_rollout_semantic_state_output_from_free_rollout_predict: `True`
- v2_reality_check_conclusion: `fix_export_eval_before_training`

## Blocking Reasons
- V2 export does not include raw future_semantic_embedding/future_identity_embedding/future_uncertainty tensors; it includes norm/mean summaries, so embedding degeneracy cannot be fully audited.
- V2 export forward_scope is head checkpoint forward with manifest surrogate features, not full Stage1/Stage2 feature-based export.
- V2 export lacks raw future_semantic_embedding/future_identity_embedding tensors; only norm summaries are available, so unit/horizon embedding degeneracy cannot be ruled out.
