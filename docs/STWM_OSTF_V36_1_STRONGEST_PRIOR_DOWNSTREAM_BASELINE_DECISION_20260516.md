# STWM OSTF V36.1 Strongest-Prior Downstream Baseline Decision

- strongest_prior_name: last_observed_copy
- semantic_three_seed_passed: True
- stable_preservation: True
- identity_real_instance_three_seed_passed: True
- v36_v30_downstream_beats_strongest_prior_semantic: True
- v36_v30_downstream_beats_strongest_prior_identity: True
- v36_v30_downstream_utility_beats_strongest_prior_slice: True
- recommended_next_step: run_v36_2_frozen_v30_prior_selector_calibration

## 中文总结
V36 V30 predicted trace 虽然 ADE 输给 strongest prior，但 downstream semantic/identity utility 至少有一条线优于 strongest-prior slice；下一步应做 frozen V30 prior selector/calibration。
