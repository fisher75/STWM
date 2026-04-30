# STWM Papergrade Protocol V2 Decision

- audit_name: `stwm_papergrade_protocol_v2_decision`
- vipseg_included: `False`
- mixed_dataset_protocol_available: `False`
- cross_dataset_protocol_available: `False`
- residual_beats_copy_mixed: `False`
- residual_beats_copy_vspw: `True`
- residual_beats_copy_vipseg: `NA`
- changed_gain_CI_excludes_zero_mixed: `False`
- changed_gain_CI_excludes_zero_vspw: `True`
- trace_regression_detected: `False`
- world_model_output_contract_satisfied: `True`
- paper_world_model_claimable: `true`
- paper_world_model_claim_scope: `VSPW-only semantic-memory eligible free-rollout protocol; VIPSeg limitation must be explicit.`
- semantic_field_branch_status: `main_contribution_candidate`
- recommended_next_step_choice: `proceed_to_paper_assets_with_vspw_only_limitation`
- vipseg_blocker: `observed semantic memory cache reports direct_cache_item_hits only for VSPW; VIPSeg observed_feature_mask is all false. The likely blocker is missing/zero VIPSeg observed predecode semantic crops or missing VIPSeg teacher/predecode cache entries, not missing future targets.`
