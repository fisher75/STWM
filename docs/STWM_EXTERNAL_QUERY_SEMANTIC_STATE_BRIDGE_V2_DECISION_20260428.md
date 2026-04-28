# STWM External Query Semantic State Bridge V2 Decision

- external_query_bridge_built: `True`
- external_query_full_model_export_available: `True`
- external_query_eval_available: `True`
- external_query_signal_positive: `False`
- target_quality: `external_candidate_expanded`
- no_future_candidate_leakage: `True`
- signal_transfer_from_internal_to_external: `False`
- paper_world_model_claimable: `False`
- semantic_state_branch_status: `stop_branch_write_official_method`
- recommended_next_step_choice: `stop_semantic_state_branch_and_write_official_method`

Full-model external query export/eval is valid and leak-free, but candidate-expanded metrics are not positive (AUROC below 0.5 and AP only modestly above base rate), so semantic-state branch should not be promoted as paper-level world-model evidence.
