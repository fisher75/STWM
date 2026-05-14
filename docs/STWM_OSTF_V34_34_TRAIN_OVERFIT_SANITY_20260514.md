# V34.34 train-overfit sanity 中文报告

- 中文结论: `V34.34 train-overfit sanity 完成；直接评估训练 split，判断 cross-attention value writer 是否至少能在训练集超过 evidence anchor。`
- selected_train_config: `{'val_gain_anchor': 0.07290908571165383, 'val_gain_pointwise': 0.5126355023441656, 'gate_mode': 'sparse_gate', 'scale': 1.0, 'stable': True, 'train_gain_anchor': 0.07290908571165383, 'train_gain_pointwise': 0.5126355023441656}`
- train_overfits_anchor: `True`
- train_beats_copy_topk: `True`
- train_intervention_delta: `{'zero_semantic_measurements_delta': 0.6809648468170858, 'shuffle_semantic_measurements_delta': 0.13398353447574157, 'shuffle_assignment_delta': 0.23768644897888686, 'zero_unit_memory_delta': 0.07290908156058018, 'normal_hard_changed_gain_vs_pointwise': 0.5126355023441656, 'normal_hard_changed_gain_vs_anchor': 0.07290908571165383}`
- recommended_next_step: `fix_value_decoder_generalization`
