# STWM OSTF V35.11 Video Identity Measurement Base And Stable Copy Adapter

- video_identity_measurement_base_eval_ran: true
- sample_count: 32
- measurement_identity_retrieval_passed: True
- learned_v35_10_identity_retrieval_passed: False
- identity_domain_shift_detected: True
- stable_copy_adapter_passed: True
- video_input_trace_measurement_closure_passed: True
- full_video_semantic_identity_field_claim_allowed: false
- recommended_next_step: build_video_derived_future_semantic_state_targets

## 中文总结
V35.11 显示 video-derived trace + observed CLIP measurement 的 identity base 是可用的；V35.10 learned identity 失败主要是域迁移问题。stable semantic 在当前稳定类别标签 cache 上应由 copy adapter 保底。但 changed/hard semantic target 仍不可评估，因此不能 claim 完整 semantic field success。

## 关键指标
- measurement_identity_exclude_same_point_top1: 0.805078125
- measurement_identity_instance_pooled_top1: 0.7093005952380953
- stable_copy_top1: 1.0
