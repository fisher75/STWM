# STWM Free-Rollout Semantic Trace Field V5 Val Selection

- audit_name: `stwm_free_rollout_semantic_trace_field_v5_val_selection`
- selection_split: `val`
- best_selected_on_val_only: `True`
- selection_rule: `primary changed_subset_top5 gain over copy; secondary overall top5 gain; tie lower trace coord error`
- selected_prototype_count: `64`
- selected_seed: `789`
- selected_checkpoint_path: `outputs/checkpoints/stwm_semantic_memory_world_model_v3_20260428/c64_seed789_final.pt`
- selected_changed_gain_over_copy: `0.19037199802941251`
- selected_overall_gain_over_copy: `0.10970831384438773`
- selected_future_trace_coord_error: `0.5217391947905222`
- candidate_count: `10`
- test_metrics_used_for_selection: `False`
