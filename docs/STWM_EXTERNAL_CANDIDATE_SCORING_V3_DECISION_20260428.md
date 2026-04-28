# STWM External Candidate Scoring V3 Decision

- scoring_failure_confirmed: `True`
- candidate_semantic_features_available: `True`
- semantic_identity_scoring_used: `True`
- posterior_v1_improves_over_distance_only: `False`
- posterior_v1_signal_positive: `True`
- heldout_signal_positive: `False`
- paper_world_model_claimable: `False`
- semantic_state_branch_status: `appendix_diagnostic`
- recommended_next_step_choice: `improve_candidate_measurement_features`

Candidate measurement features removed tie/index-0 bias, but fixed posterior_v1 does not improve over corrected distance_only on overall or heldout metrics. Semantic/identity compatibility is wired but weak; branch should remain diagnostic and candidate measurement should be improved before any claim or training.
