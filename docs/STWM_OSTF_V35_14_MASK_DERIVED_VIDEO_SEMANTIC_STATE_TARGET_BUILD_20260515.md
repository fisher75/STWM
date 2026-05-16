# STWM OSTF V35.14 Mask-Derived Video Semantic State Target Build

- mask_derived_video_semantic_state_targets_built: True
- sample_count: 32
- video_semantic_target_source: mask_label / panoptic_instance / object_track
- semantic_changed_is_real_video_state: true
- identity_confuser_target_built: True
- current_video_cache_insufficient_for_semantic_change_benchmark: False
- future_teacher_embedding_input_allowed: false
- recommended_next_step: eval_mask_derived_video_semantic_state_predictability

## 中文总结
V35.14 已从真实 VSPW/VIPSeg mask/panoptic label 沿 CoTracker future trace 采样构建 video semantic state targets；changed/hard 不再来自 CLIP/KMeans。
