# STWM OSTF V35.16 Video Identity Pairwise Target Build

- video_identity_pairwise_targets_built: True
- sample_count: 192
- same_frame_hard_negative_built: true
- same_semantic_confuser_built: true
- trajectory_crossing_target_built: true
- occlusion_reappear_target_built: true
- future_teacher_embedding_input_allowed: false
- recommended_next_step: train_video_identity_pairwise_retrieval_head

## 中文总结
V35.16 已把 video identity 从单点 same-instance 改成 pairwise/retrieval target，包含 same-semantic confuser、空间近邻 hard negative、轨迹 crossing、occlusion/reappear 分层。
