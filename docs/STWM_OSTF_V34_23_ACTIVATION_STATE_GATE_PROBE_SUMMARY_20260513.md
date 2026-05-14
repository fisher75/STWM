# V34.23 activation-state gate probe 中文报告

- 中文结论: `V34.23 activation-state gate probe 已完成；只训练 activation reader/gate probe，V30、V34.20 residual 和 selector 全部 frozen，仍不声明 semantic field success。`
- activation_state_gate_probe_ran: `True`
- activation_state_gate_probe_passed: `True`
- gate_name: `benefit_soft`
- v30_backbone_frozen: `True`
- future_leakage_detected: `False`
- learned_gate_training_ran: `True`
- semantic_hard_signal: `{'val': True, 'test': True}`
- changed_semantic_signal: `{'val': True, 'test': True}`
- stable_preservation: `{'val': True, 'test': True}`
- hard_changed_gain: `{'val': 0.13908015269108118, 'test': 0.1301964509373285}`
- semantic_measurements_load_bearing_on_residual: `True`
- assignment_load_bearing_on_residual: `True`
- unit_memory_load_bearing_on_residual: `True`
- integrated_semantic_field_claim_allowed: `False`
- recommended_next_step: `run_v34_23_seed123_replication`
