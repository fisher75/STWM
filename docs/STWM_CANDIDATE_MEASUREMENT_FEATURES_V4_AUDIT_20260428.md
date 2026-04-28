# STWM Candidate Measurement Features V4 Audit

- stage2_semantic_encoder: `SemanticEncoder with SemanticCropEncoder crop_visual_encoder path`
- can_reuse_crop_visual_encoder: `True`
- crop_encoder_weights_available: `True`
- observed_target_crop_feature_constructible: `True`
- future_candidate_crop_feature_constructible: `True`
- observed_target_to_candidate_appearance_similarity_constructible: `True`
- predicted_future_semantic_to_candidate_similarity_constructible: `True`
- predicted_future_identity_to_candidate_similarity_constructible: `True`
- candidate_feature_enters_rollout_input: `False`
- recommended_feature_mode: `crop_encoder_feature`

Future candidate crops are used only as posterior measurement observations after rollout export; they are not model input.
