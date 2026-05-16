# STWM OSTF V36 / V35.49 因果 Trace Contract 审计

- frontend_reads_future_frames: True
- cotracker_offline_sees_future_frames: True
- future_points_from_full_clip_frontend: True
- semantic_input_contains_future_trace: True
- identity_input_contains_future_trace: True
- v35_49_is_causal_past_only_world_model: False
- v35_49_is_teacher_trace_upper_bound: True
- claim_boundary_requires_rename: True
- recommended_fix: build_v36_past_only_observed_trace_then_run_frozen_v30_rollout

## 中文总结
V35.49 的 raw-video rerun 是 full-clip CoTracker teacher trace closure：frontend 读取 obs+future 帧，future_points 来自 full-clip tracks，并且 semantic/identity 输入特征会使用 future trace 字段。因此 V35.49 只能标注为 teacher-trace upper-bound，不是严格因果 past-only world model。
