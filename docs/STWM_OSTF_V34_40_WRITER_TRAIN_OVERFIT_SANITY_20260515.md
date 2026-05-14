# V34.40 writer train-overfit sanity 中文报告

- 中文结论: `V34.40 writer train-overfit sanity 完成；检查 prototype-conditioned mixture writer 是否至少能在训练集形成 assignment-bound 正 correction。`
- selected_train_config: `{'scale': 2.0, 'stable': True, 'train_gain_anchor': 0.027933398067273796, 'train_gain_pointwise': 0.46765982050819077}`
- train_overfits_prototype_mixture: `True`
- train_intervention_delta: `{'normal_hard_changed_gain_vs_anchor': 0.027933398067273796, 'normal_hard_changed_gain_vs_pointwise': 0.46765982050819077, 'shuffle_assignment_delta': 0.04232193788497379, 'zero_unit_memory_delta': 0.0279333997246054}`
- recommended_next_step: `fix_prototype_mixture_generalization`
