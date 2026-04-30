# STWM Free-Rollout Semantic Trace Field V4 V3 Audit

- audit_name: `stwm_free_rollout_semantic_trace_field_v4_v3_audit`
- v3_eval_teacher_forced: `True`
- v3_free_rollout_path_called: `False`
- v3_free_rollout_semantic_field_signal: `unclear`
- free_rollout_unclear_reason: `V3 runner evaluates seed checkpoints with _teacher_forced_predict; free-rollout script was not completed in V3.`
- heldout_materialized_item_count: `34`
- requested_heldout_item_count: `128`
- nominal_v3_test_item_count: `48`
- nominal_v3_val_plus_test_item_count: `96`
- checkpoints_support_free_rollout_export: `True`
- free_rollout_function: `stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain._free_rollout_predict`
- minimum_free_rollout_eval_entry: `code/stwm/tools/eval_free_rollout_semantic_trace_field_20260428.py`
- best_c32_checkpoint: `outputs/checkpoints/stwm_semantic_memory_world_model_v3_20260428/c32_seed456_final.pt`
- best_c64_checkpoint: `outputs/checkpoints/stwm_semantic_memory_world_model_v3_20260428/c64_seed456_final.pt`
- candidate_scorer_used: `False`
- future_candidate_leakage: `False`
- paper_claim_allowed: `False`
