# V34.39 writer train-overfit sanity 中文报告

- 中文结论: `V34.39 writer train-overfit sanity 完成；检查 prototype-blended learned writer 是否至少能在训练集复现正 correction。`
- selected_train_config: `{'scale': 2.0, 'stable': True, 'train_gain_anchor': 0.047678157006091175, 'train_gain_pointwise': 0.487404587857579}`
- train_overfits_blended_target: `True`
- train_intervention_delta: `{'normal_hard_changed_gain_vs_anchor': 0.047678157006091175, 'normal_hard_changed_gain_vs_pointwise': 0.487404587857579, 'shuffle_assignment_delta': 0.09670680057818398, 'zero_unit_memory_delta': 0.04767816707399364}`
- recommended_next_step: `fix_writer_generalization_or_target_predictability`
