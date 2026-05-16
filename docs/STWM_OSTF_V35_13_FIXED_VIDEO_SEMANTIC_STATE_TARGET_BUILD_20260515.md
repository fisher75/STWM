# STWM OSTF V35.13 Fixed Video Semantic State Target Build

- fixed_video_semantic_state_targets_built: True
- sample_count: 32
- future_teacher_embeddings_input_allowed: false
- leakage_safe: true
- recommended_next_step: eval_fixed_video_semantic_state_target_predictability

## 中文总结
V35.13 已把 video semantic target 从 future CLIP KMeans 精确 cluster 修成可观测的 coarse state：changed/risk/abstain + stable copy family。下一步重跑 observed-only predictability。
