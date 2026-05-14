# V34.22 activation-state reader predictability probe 中文报告

- 中文结论: `V34.22 activation-state reader predictability probe 已完成；它训练 observed-only cross-attention reader 来预测 V34.20 有效发力区域，但没有接入 learned gate。`
- probe_ran: `True`
- activation_state_reader_built: `True`
- v30_backbone_frozen: `True`
- future_leakage_detected: `False`
- uses_future_teacher_as_input: `False`
- learned_gate_training_ran: `False`
- aligned_activation_predictable: `False`
- utility_activation_predictable: `False`
- benefit_activation_predictable: `False`
- gate_predictability_passed: `False`
- reader_soft_gate_probe_passed: `True`
- best_reader_gate_by_val: `benefit_soft`
- reader_gate_intervention_load_bearing: `True`
- attention_stats: `{'train': {'attention_entropy_mean': 0.4142986238002777, 'attention_max_mean': 0.6730059385299683, 'attention_nontrivial': True}, 'val': {'attention_entropy_mean': 0.43258970975875854, 'attention_max_mean': 0.6564854979515076, 'attention_nontrivial': True}, 'test': {'attention_entropy_mean': 0.3913945257663727, 'attention_max_mean': 0.6864622831344604, 'attention_nontrivial': True}}`
- recommended_next_step: `train_activation_state_gate_probe`
