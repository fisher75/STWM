# V34.36 writer train-overfit sanity 中文报告

- 中文结论: `V34.36 writer train-overfit sanity 完成；检查 learned writer 是否至少能在训练集复现 predictability-filtered target 的正 correction。`
- selected_train_config: `{'scale': 1.0, 'stable': True, 'train_gain_anchor': 0.05586557884769423, 'train_gain_pointwise': 0.49559200489756705}`
- train_overfits_filtered_target: `True`
- train_intervention_delta: `{'normal_hard_changed_gain_vs_anchor': 0.05586557884769423, 'normal_hard_changed_gain_vs_pointwise': 0.49559200489756705, 'shuffle_assignment_delta': 0.11113587502582145, 'zero_unit_memory_delta': 0.05586558411398168}`
- recommended_next_step: `fix_writer_generalization`
