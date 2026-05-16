# STWM OSTF V36.3 Full 325 Causal Benchmark Rerun Decision

- selected_clip_count: 325
- trace_source: v36_2c_conservative_copy_default_selector
- future_trace_predicted_from_past_only: true
- trace_no_harm_copy_val/test: True / True
- trace_beats_strongest_prior_val/test: True / True
- semantic_three_seed_passed: True
- stable_preservation: True
- identity_real_instance_three_seed_passed: True
- identity_pseudo_targets_excluded_from_claim: true
- future_leakage_detected: false
- trajectory_degraded: false
- causal_benchmark_passed: True
- m128_h32_causal_video_world_model_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: write_v36_causal_claim_boundary_and_packaging_audit

## 中文总结
V36.3 full 325 causal benchmark rerun 通过：selector trace 由 past-only observed trace 决定，semantic 三 seed、stable preservation、real-instance identity 三 seed 均通过。
