# V34.33 cached oracle unit delta readout sanity 中文报告

- 中文结论: `V34.33 cached oracle unit delta readout sanity 完成；直接读取物化 oracle unit_delta 缓存，不训练新模型，用同一 evidence-anchor 协议验证 target/readout 上界。`
- cached_oracle_readout_passed: `True`
- selected_config_by_val: `{'gate_mode': 'sparse_gate', 'scale': 2.0, 'stable': True, 'val_gain_anchor': 0.18144904665986566, 'val_gain_pointwise': 0.6567831832569689}`
- beats_copy_topk_baseline: `True`
- unit_residual_improves_evidence_anchor: `True`
- assignment_load_bearing_on_system: `True`
- unit_memory_load_bearing_on_system: `True`
- semantic_hard_signal: `{'val': True, 'test': True}`
- changed_semantic_signal: `{'val': True, 'test': True}`
- stable_preservation: `{'val': True, 'test': True}`
- intervention_delta: `{'val': {'normal_hard_changed_gain_vs_anchor': 0.18144904665986566, 'normal_hard_changed_gain_vs_pointwise': 0.6567831832569689, 'shuffle_assignment_delta': 0.22706041686004202, 'zero_unit_memory_delta': 0.18144904866527906}, 'test': {'normal_hard_changed_gain_vs_anchor': 0.16094101774179806, 'normal_hard_changed_gain_vs_pointwise': 0.6409389055787699, 'shuffle_assignment_delta': 0.21403490396257596, 'zero_unit_memory_delta': 0.16094100900476382}}`
- v30_backbone_frozen: `True`
- future_leakage_detected: `False`
- trajectory_degraded: `False`
- learned_gate_training_ran: `False`
- integrated_semantic_field_claim_allowed: `False`
- recommended_next_step: `fix_value_decoder_capacity`
