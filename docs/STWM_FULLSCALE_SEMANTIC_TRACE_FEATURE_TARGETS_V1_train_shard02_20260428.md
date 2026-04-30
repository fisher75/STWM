# STWM Future Semantic Trace Feature Targets V1

- feature_backbone: `openai_clip_vit_b_32_local`
- feature_source: `future_gt_bbox_crop_clip_vit_b32`
- feature_dim: `512`
- item_count: `701`
- target_shape: `[701, 8, 8, 512]`
- target_mask_shape: `[701, 8, 8]`
- valid_target_ratio: `0.7424884094151213`
- no_future_candidate_leakage: `True`
- cache_path: `outputs/cache/stwm_fullscale_semantic_trace_feature_targets_v1_train_shard02_20260428/future_semantic_trace_feature_targets_v1.npz`

Future GT crops are used only as supervised targets. They are not inserted into rollout inputs.
