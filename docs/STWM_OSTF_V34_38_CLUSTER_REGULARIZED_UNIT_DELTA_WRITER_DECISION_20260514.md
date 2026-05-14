# V34.38 cluster-regularized unit_delta writer 决策中文报告

- 中文结论: `V34.38 cluster-regularized unit_delta writer 完成；本轮不训练 gate、不跑 M512，只验证 prototype-smoothed correction target 是否能提升 learned writer 泛化。`
- cluster_regularized_targets_built: `True`
- probe_passed: `False`
- selected_config_by_val: `{'gate_mode': 'predictable_oracle_mask', 'scale': 0.25, 'stable': True, 'val_gain_anchor': -0.0005183768020078169, 'val_gain_pointwise': 0.47481575288954075}`
- beats_copy_topk_baseline: `False`
- unit_residual_improves_evidence_anchor: `False`
- assignment_load_bearing_on_system: `True`
- unit_memory_load_bearing_on_system: `False`
- semantic_hard_signal: `{'test': True, 'val': True}`
- changed_semantic_signal: `{'test': True, 'val': True}`
- stable_preservation: `{'test': True, 'val': True}`
- recommended_next_step: `fix_cluster_regularized_targets_or_writer_generalization`
