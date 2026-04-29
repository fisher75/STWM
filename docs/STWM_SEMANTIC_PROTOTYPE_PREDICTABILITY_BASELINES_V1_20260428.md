# STWM Semantic Prototype Predictability Baselines V1

- Simple MLP probes are trained only as diagnostics and do not update STWM.
- Observed crop features are measurement diagnostics, not candidate scorer inputs.

- audit_name: `stwm_semantic_prototype_predictability_baselines_v1`
- target_predictable_from_observed_semantics: `True`
- trace_only_sufficient: `True`
- semantic_input_load_bearing: `True`
- simple_probe_beats_frequency: `True`
- no_stwm_backbone_update: `True`
- candidate_scorer_used: `False`
- future_candidate_input_used: `False`
