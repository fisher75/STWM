# STWM OSTF V36.2 Frozen V30 Prior Selector Calibrated Rollout

- train_new_large_model: false
- V30 frozen: true
- train_points_used: 161408
- candidate_methods: ['v30', 'last_observed_copy', 'last_visible_copy', 'constant_velocity', 'damped_velocity', 'global_median_velocity']
- calibrated_selector_ADE_all: 92.42791821231258
- strongest_prior_all: last_observed_copy
- strongest_prior_ADE_all: 116.70810838118665
- calibrated_minus_strongest_prior_ADE_all: -24.28019016887407
- calibrated_beats_strongest_prior_val: False
- calibrated_beats_strongest_prior_test: False
- calibrated_rollout_passed: False
- recommended_next_step: fix_v30_prior_selector_calibration

## 中文总结
V36.2 selector 还没有在 val/test 同时赢 strongest prior；需要继续修 prior selector/calibration，而不是训练 semantic/identity。
