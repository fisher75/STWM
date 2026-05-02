# STWM OSTF V18 Decision Logic Audit V19

- V18_M128_vs_constant_velocity_positive: `True`
- V18_M512_vs_constant_velocity_positive: `False`
- current_v18_decision_only_tracks_M512: `True`
- best_M128_or_M512: `V18_M128`
- best_M128_or_M512_beats_constant_velocity: `True`
- semantic_oracle_leakage_exists: `True`
- metrics_allowed_for_claim: `['point_L1_px', 'endpoint_error_px', 'PCK_4px', 'PCK_8px', 'PCK_16px', 'PCK_32px', 'visibility_F1', 'object_extent_iou', 'corrected_semantic_top1', 'corrected_semantic_top5']`
- metrics_must_be_discarded_or_qualified: `['old_analytic_semantic_top1', 'old_analytic_semantic_top5']`
