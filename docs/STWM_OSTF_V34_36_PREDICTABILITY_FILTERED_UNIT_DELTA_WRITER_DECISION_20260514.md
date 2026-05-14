# V34.36 predictability-filtered unit_delta writer 决策中文报告

- 中文结论: `V34.36 predictability-filtered unit_delta writer 完成；目标从原始 sample-specific oracle delta 改成 observed-predictable filtered component，仍不训练 learned gate。`
- probe_passed: `False`
- selected_config_by_val: `{'gate_mode': 'predictable_oracle_mask', 'scale': 0.25, 'val_gain_anchor': -0.0011452059719279516, 'val_gain_pointwise': 0.4741889200380029, 'stable': True}`
- beats_copy_topk_baseline: `False`
- unit_residual_improves_evidence_anchor: `False`
- assignment_load_bearing_on_system: `True`
- unit_memory_load_bearing_on_system: `False`
- semantic_hard_signal: `{'val': True, 'test': True}`
- changed_semantic_signal: `{'val': True, 'test': True}`
- stable_preservation: `{'val': True, 'test': True}`
- recommended_next_step: `fix_predictability_filtered_targets`
