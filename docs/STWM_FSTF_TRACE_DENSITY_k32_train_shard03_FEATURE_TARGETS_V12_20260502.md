# STWM Future Semantic Trace Feature Targets V1

- feature_backbone: `openai_clip_vit_b_32_local`
- feature_source: `future_gt_bbox_crop_clip_vit_b32`
- feature_dim: `512`
- item_count: `702`
- target_shape: `[702, 8, 32, 512]`
- target_mask_shape: `[702, 8, 32]`
- valid_target_ratio: `0.19537370904558404`
- no_future_candidate_leakage: `True`
- cache_path: `outputs/cache/stwm_fstf_trace_density_k32_train_shard03_feature_targets_v12_20260502/future_semantic_trace_feature_targets_v1.npz`

Future GT crops are used only as supervised targets. They are not inserted into rollout inputs.
