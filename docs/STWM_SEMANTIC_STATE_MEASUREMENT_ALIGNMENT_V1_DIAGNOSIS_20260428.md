# STWM Semantic-State Measurement Alignment V1 Diagnosis

- future_semantic_embedding_dim: `256`
- future_identity_embedding_dim: `256`
- crop_encoder_feature_dim: `256`
- same_space_before_alignment: `False`
- learnable_alignment_head_needed: `True`
- future_candidate_used_as_input: `False`
- candidate_feature_used_for_rollout: `False`

Proceed with a small dev-only alignment probe; do not update STWM backbone and do not treat target-candidate appearance-only as world-model evidence.
