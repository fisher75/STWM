# STWM OSTF V34 Against Teacher Prototype Baselines

- does_v34_escape_teacher_only_path: `True`
- does_v34_use_teacher_as_measurement_only: `True`
- does_v34_improve_trace_conditioned_semantic_belief: `False`
- baselines: `{'v33_14_best_teacher_prototype_target_probe': {'best_teacher_by_val': 'dinov2_base', 'target_space_learnability_passed': False, 'changed_signal_positive': False, 'semantic_hard_signal_positive': True}, 'teacher_only_nearest_observed_measurement': 'represented_by_copy_observed_measurement_baseline_in_v34_eval', 'sample_frequency_baseline': 'represented_by_v33_14_probe_sweep', 'copy_baseline': 'represented_by_v34_stable/copy cosine comparisons', 'v33_13_gate_repaired_model': {'stable_preservation_not_degraded_top5': False, 'changed_top5_beats_strongest_baseline': False, 'semantic_hard_top5_beats_strongest_baseline': False}, 'v34_semantic_trace_units': {'stable_preservation': {'test': False, 'val': False}, 'changed_semantic_signal': {'test': False, 'val': False}, 'semantic_belief_consistency': {'test': 0.9986651539802551, 'val': 0.9988549947738647}}}`
