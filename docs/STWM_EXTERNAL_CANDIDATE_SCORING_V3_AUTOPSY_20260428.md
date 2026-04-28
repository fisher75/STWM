# STWM External Candidate Scoring V3 Autopsy

- score_formula_current: `candidate_score = distance_score * (0.5 + 0.5 * item_reappearance_event_prob) * (0.5 + 0.5 * item_visibility_prob)`
- visibility_reappearance_same_for_candidates_within_item: `True`
- semantic_embedding_used_for_scoring: `False`
- identity_embedding_used_for_scoring: `False`
- candidate_crop_semantic_feature_used_for_scoring: `False`
- all_candidate_score_equal_ratio: `0.9922879177377892`
- predicted_candidate_index_0_ratio: `1.0`
- distance_only_top1: `0.13212435233160622`
- distance_only_AP: `0.06995069572340755`
- distance_only_AUROC: `0.4843919640571625`

This v2 negative result is primarily a scoring-bridge failure/distance-only posterior limitation, not direct evidence that the FutureSemanticTraceState world-model direction failed.
