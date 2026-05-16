# STWM OSTF V35.14 Video Semantic Target Truth Audit

- video_semantic_target_truth_audit_done: true
- cache_sample_count: 32
- mask_label_available: True
- panoptic_instance_available: True
- v35_13_target_predictability_failed: True
- semantic_changed_is_real_video_state: false
- recommended_fix: build_mask_derived_future_semantic_state_targets

## 中文总结
V35.13 的 video semantic target 仍主要来自 CLIP/KMeans 或其派生 coarse state，不是直接来自真实 mask/panoptic label；当前 trace cache 对应的 VSPW/VIPSeg mask 文件可用，因此下一步应构建 mask-derived future semantic state targets。
