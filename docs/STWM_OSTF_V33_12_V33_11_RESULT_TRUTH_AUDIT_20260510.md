# STWM OSTF V33.12 V33.11 Result Truth Audit

- v33_11_checkpoint_fresh: `True`
- identity_preservation_fixed: `True`
- stable_preservation_failed: `True`
- changed_hard_failed_vs_strongest_baseline: `True`
- v33_11_oracle_not_actually_run: `True`
- current_teacher_name: `clip_vit_b32_local`
- current_prototype_K: `32`
- strongest_baseline_by_subset: `{'changed': 'sample_level_prototype_frequency', 'global': 'sample_level_prototype_frequency', 'semantic_hard': 'sample_level_prototype_frequency', 'stable': 'last_observed_copy'}`
- prototype_target_space_bottleneck_suspected: `True`
- recommended_fix: `run_true_v33_11_oracle_then_repair_semantic_teacher_prototype_target_space`
