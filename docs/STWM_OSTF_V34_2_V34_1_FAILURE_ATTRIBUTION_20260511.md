# STWM OSTF V34.2 V34.1 Failure Attribution

- z_dyn_source_is_trace_dynamics: `False`
- z_dyn_source_is_semantic_measurement: `True`
- z_sem_source_is_semantic_measurement: `True`
- z_dyn_z_sem_factorization_real: `False`
- identity_key_source: `V34.1 identity_key is derived from unit_sem through FactorizedTraceSemanticState, not from a dual-source z_dyn+z_sem state.`
- unit_confidence_used_in_loss: `False`
- point_to_unit_target_is_permutation_invariant: `False`
- real_pointwise_no_unit_baseline_exists: `False`
- trace_units_better_than_pointwise_proven: `False`
- semantic_field_failed_because_units_or_targets_or_losses: `unit_architecture`
- recommended_fix: `Implement dual-source trace/semantic unit state, replace fixed slot CE with permutation-aware pairwise binding, and train a real pointwise no-unit baseline.`
