# STWM OSTF V36.2b Split-Safe Bucket Prior Selector

- train_new_large_model: false
- V30 frozen: true
- selector_type: split_safe_bucket_validated_monotonic_rule
- policy_count: 25
- selected_method_histogram: {'damped_velocity': 212, 'last_observed_copy': 13, 'global_median_velocity': 29, 'constant_velocity': 53, 'last_visible_copy': 18}
- bucket_selector_ADE_all: 135.58107112638606
- strongest_prior_all: last_observed_copy
- strongest_prior_ADE_all: 116.70810838118665
- bucket_selector_minus_strongest_prior_ADE_all: 18.872962745199416
- bucket_selector_beats_strongest_prior_val: True
- bucket_selector_beats_strongest_prior_test: False
- bucket_selector_rollout_passed: False
- recommended_next_step: fix_v30_prior_selector_calibration

## 中文总结
V36.2b split-safe bucket selector 仍未在 val/test 同时赢 strongest prior；需要继续修 trace calibration，不允许跑 V36.3 claim。
