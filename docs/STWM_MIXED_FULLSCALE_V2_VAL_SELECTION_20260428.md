# STWM Free-Rollout Semantic Trace Field V5 Val Selection

- audit_name: `stwm_free_rollout_semantic_trace_field_v5_val_selection`
- selection_split: `val`
- best_selected_on_val_only: `True`
- selection_rule: `primary changed_subset_top5 gain over copy; secondary overall top5 gain; tie lower trace coord error`
- selected_prototype_count: `64`
- selected_seed: `42`
- selected_checkpoint_path: `outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/c64_seed42_final.pt`
- selected_changed_gain_over_copy: `0.06391121359549856`
- selected_overall_gain_over_copy: `0.03184787229470065`
- selected_future_trace_coord_error: `0.31277955605126007`
- candidate_count: `1`
- test_metrics_used_for_selection: `False`
