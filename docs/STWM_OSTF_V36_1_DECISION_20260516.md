# STWM OSTF V36.1 Decision

- strongest_prior_name: last_observed_copy
- v30_beats_strongest_prior_trace_ADE: False
- v30_minus_strongest_prior_ADE: 10.000746079679544
- v36_v30_downstream_beats_strongest_prior_semantic: True
- v36_v30_downstream_beats_strongest_prior_identity: True
- v36_v30_downstream_utility_beats_strongest_prior_slice: True
- m128_h32_causal_video_world_model_claim_allowed: false
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: run_v36_2_frozen_v30_prior_selector_calibration

## 中文总结
V36.1 说明 V30 trace ADE 虽未赢 strongest prior，但 downstream utility 仍有机会；下一步应做 frozen V30 prior selector/calibration。
