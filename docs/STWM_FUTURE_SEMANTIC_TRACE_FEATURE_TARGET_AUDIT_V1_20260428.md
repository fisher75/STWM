# STWM Future Semantic Trace Feature Target Audit V1

- can_construct_future_crop_per_trace_unit: `True`
- can_compute_future_crop_frozen_semantic_feature: `True`
- can_construct_B_H_K_D_future_semantic_feature_target: `True`
- can_construct_B_H_K_target_mask: `True`
- audit_conclusion: `targets_available_for_small_sanity`

The minimum viable supervision path is future GT bbox crop -> frozen semantic feature target. This is training supervision, not rollout input.
