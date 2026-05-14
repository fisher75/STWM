# V34.39 prototype blend target sweep 中文报告

- 中文结论: `V34.39 prototype blend target sweep 完成；不训练 writer，只评估 crossfit filtered delta 与 prototype-smoothed delta 的 convex blend 是否保留 cached upper bound。`
- gate_mode: `oracle_mask`
- selected: `{'key': 'alpha_1.00_scale_2.00', 'passed': True, 'val': {'alpha': 1.0, 'scale': 2.0, 'hard_changed_gain_vs_anchor': 0.016149454319368036, 'hard_changed_gain_vs_pointwise': 0.49148358868544895, 'semantic_hard_signal': True, 'changed_semantic_signal': True, 'stable_preservation': True, 'shuffle_assignment_delta': 0.014447617011922098, 'zero_unit_memory_delta': 0.016149454093759064}, 'test': {'alpha': 1.0, 'scale': 2.0, 'hard_changed_gain_vs_anchor': 0.014296148508100178, 'hard_changed_gain_vs_pointwise': 0.49429404237936203, 'semantic_hard_signal': True, 'changed_semantic_signal': True, 'stable_preservation': True, 'shuffle_assignment_delta': 0.013829471329322218, 'zero_unit_memory_delta': 0.014296145805355986}}`
- blend_cached_target_passed: `True`
- recommended_next_step: `train_blended_prototype_writer`
