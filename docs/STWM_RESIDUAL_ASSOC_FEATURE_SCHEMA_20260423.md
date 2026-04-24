# STWM Residual Association Feature Schema 20260423

- head_type: external_base_plus_linear_logistic_residual
- final_score_definition: ExternalTeacherScore(candidate) + ResidualTraceScore(candidate)
- feature_names: ["unit_identity_score_norm", "coord_score_norm", "tusb_semantic_target_score_norm", "external_rank_score", "external_margin_to_top", "coord_rank_score", "coord_margin_to_top", "candidate_count_scaled", "is_occlusion_reappearance", "is_long_gap_persistence", "is_crossing_ambiguity", "external_x_unit", "external_x_coord"]
