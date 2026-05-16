# STWM OSTF V36 Decision

- v35_49_is_teacher_trace_upper_bound: True
- past_only_observed_trace_input_built: True
- future_trace_predicted_from_past_only: True
- v30_beats_strongest_prior: False
- causal_unified_slice_built: True
- causal_benchmark_passed: False
- semantic_three_seed_passed: True
- stable_preservation: True
- identity_real_instance_three_seed_passed: True
- visualization_ready: True
- m128_h32_causal_video_world_model_claim_allowed: False
- m128_h32_teacher_trace_upper_bound_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: fix_v30_vs_strongest_prior

## 中文总结
V36 已完成因果 contract 审计与 causal pipeline 重建，但当前 causal benchmark 未过；不能 claim M128/H32 causal video world model。
