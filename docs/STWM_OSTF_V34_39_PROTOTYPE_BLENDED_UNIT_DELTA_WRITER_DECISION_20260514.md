# V34.39 prototype-blended unit_delta writer 决策中文报告

- 中文结论: `V34.39 prototype-blended unit_delta writer 完成；不训练 gate、不跑 M512，只验证 learned writer 是否能在 copy/top-k evidence anchor 上泛化出正 correction。`
- prototype_blended_targets_built: `True`
- blend_alpha: `0.9`
- probe_passed: `False`
- selected_config_by_val: `{'gate_mode': 'predictable_oracle_mask', 'scale': 0.25, 'stable': True, 'val_gain_anchor': -0.0006226438115575239, 'val_gain_pointwise': 0.4747114826258945}`
- beats_copy_topk_baseline: `False`
- unit_residual_improves_evidence_anchor: `False`
- assignment_load_bearing_on_system: `True`
- unit_memory_load_bearing_on_system: `False`
- semantic_hard_signal: `{'test': True, 'val': True}`
- changed_semantic_signal: `{'test': True, 'val': True}`
- stable_preservation: `{'test': True, 'val': True}`
- recommended_next_step: `fix_writer_generalization_or_target_predictability`
