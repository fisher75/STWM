# STWM Full-Model Future Semantic State Export Repair V2 Summary 20260427

- head_only_surrogate_passed: `True`
- full_model_teacher_forced_export_available: `True`
- full_model_free_rollout_export_available: `True`
- random_hidden_used_in_full_model_modes: `False`
- old_association_report_used: `False`
- full_model_teacher_forced_valid_ratio: `1.0`
- full_model_free_rollout_valid_ratio: `1.0`
- teacher_forced_output_degenerate: `False`
- free_rollout_output_degenerate: `False`
- safe_for_medium_training: `True`
- world_model_output_now_claimable: `True`
- recommended_next_step_choice: `proceed_to_medium_semantic_state_training`

The V1 overclaim is corrected. The head-only path is now explicitly marked non-claimable, while both full-model teacher-forced and full-model free-rollout exports use Stage2 dataset batches and trainer prediction paths.
