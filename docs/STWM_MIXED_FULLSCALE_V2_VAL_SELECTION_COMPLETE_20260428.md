# STWM Free-Rollout Semantic Trace Field V5 Val Selection

- audit_name: `stwm_free_rollout_semantic_trace_field_v5_val_selection`
- selection_split: `val`
- best_selected_on_val_only: `True`
- selection_rule: `primary changed_subset_top5 gain over copy; secondary overall top5 gain; tie lower trace coord error`
- selected_prototype_count: `32`
- selected_seed: `456`
- selected_checkpoint_path: `outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/c32_seed456_final.pt`
- selected_changed_gain_over_copy: `0.07506448753492034`
- selected_overall_gain_over_copy: `0.03327793351154307`
- selected_future_trace_coord_error: `0.2742422410264248`
- candidate_count: `10`
- test_metrics_used_for_selection: `False`
