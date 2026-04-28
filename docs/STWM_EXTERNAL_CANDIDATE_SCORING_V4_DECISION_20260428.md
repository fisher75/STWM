# STWM External Candidate Scoring V4 Decision

- strong_candidate_measurement_available: `True`
- candidate_feature_source: `crop_encoder_feature`
- target_candidate_appearance_load_bearing: `True`
- predicted_semantic_load_bearing: `False`
- predicted_identity_load_bearing: `False`
- posterior_v4_improves_over_distance_only: `True`
- posterior_v4_improves_over_weak_posterior_v3: `True`
- heldout_signal_positive: `True`
- paper_world_model_claimable: `False`
- semantic_state_branch_status: `appendix_diagnostic`
- recommended_next_step_choice: `improve_candidate_measurement_with_vlm_features`

Strong crop-encoder candidate measurement is available and target-candidate appearance is clearly load-bearing, but predicted FutureSemanticTraceState semantic/identity components do not beat appearance-only; paper-level world-model claim remains unsupported.
