# STWM OSTF V35.21 Domain-Normalized Video Semantic State Target Build

- domain_normalized_video_semantic_state_targets_built: True
- sample_count: 325
- domain_normalization_scope: per_video_percentile_rank
- target_split_balanced_after_normalization: True
- future_teacher_embedding_input_allowed: false
- recommended_next_step: eval_domain_normalized_video_semantic_predictability

## 中文总结
V35.21 已将 mask-boundary / visibility risk 做 per-video percentile calibration，避免 VSPW held-out 因 mask 密度/置信度尺度不同而被全局阈值压成稀疏正例。
