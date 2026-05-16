# STWM OSTF V36.2c Conservative Copy-Default Selector

- train_new_large_model: false
- V30 frozen: true
- default_strategy: last_observed_copy
- switch_rule: only if train+val both beat copy
- selected_method_histogram: {'damped_velocity': 223, 'last_observed_copy': 84, 'constant_velocity': 18}
- copy_default_selector_ADE_all: 103.0988803659595
- strongest_prior_all: last_observed_copy
- strongest_prior_ADE_all: 116.70810838118665
- no_harm_copy_val: True
- no_harm_copy_test: True
- beats_strongest_prior_val: True
- beats_strongest_prior_test: True
- copy_default_selector_passed: True
- recommended_next_step: eval_v36_2c_downstream_secondary_gate

## 中文总结
V36.2c copy-default selector 满足 no-harm 与 strongest-prior gate；下一步评估 downstream secondary gate。
