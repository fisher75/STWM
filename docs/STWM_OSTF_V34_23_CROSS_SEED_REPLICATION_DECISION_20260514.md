# V34.23 跨 seed 复现决策中文总结

- 中文结论: `V34.23 seed42/123/456 复现决策已汇总；即使多 seed 通过，也仍不能声明 semantic field success，因为 stable gate over-open 风险仍需单独处理。`
- seed42_reference_passed: `True`
- seed_replication_pass: `{'seed123': True, 'seed456': True}`
- required_seeds_done: `True`
- all_required_seeds_passed: `True`
- semantic_hard_signal: `{'seed123': {'test': True, 'val': True}, 'seed456': {'test': True, 'val': True}}`
- changed_semantic_signal: `{'seed123': {'test': True, 'val': True}, 'seed456': {'test': True, 'val': True}}`
- stable_preservation: `{'seed123': {'test': True, 'val': True}, 'seed456': {'test': True, 'val': True}}`
- intervention_delta: `{'seed123': {'val': {'zero_semantic_measurements_delta': 0.06458909182427698, 'shuffle_semantic_measurements_delta': 0.017268203207526783, 'shuffle_assignment_delta': 0.01138819611394662, 'zero_unit_memory_delta': 0.13736590049612876}, 'test': {'zero_semantic_measurements_delta': 0.06358644551017938, 'shuffle_semantic_measurements_delta': 0.017109913644339544, 'shuffle_assignment_delta': 0.010938949968199038, 'zero_unit_memory_delta': 0.1283807137732116}}, 'seed456': {'val': {'zero_semantic_measurements_delta': 0.06345170941241042, 'shuffle_semantic_measurements_delta': 0.017737810449537053, 'shuffle_assignment_delta': 0.013006553052289163, 'zero_unit_memory_delta': 0.1367530822752139}, 'test': {'zero_semantic_measurements_delta': 0.06171658246094264, 'shuffle_semantic_measurements_delta': 0.01673341476075106, 'shuffle_assignment_delta': 0.011190871098709101, 'zero_unit_memory_delta': 0.1272145604539638}}}`
- gate_over_open_and_stable_update: `{'seed123': {'val_stable_gate_mean': 0.4829382614176264, 'test_stable_gate_mean': 0.5827061920196386, 'val_stable_over_open_rate': 0.9246385364414281, 'test_stable_over_open_rate': 0.9589550038332252, 'val_stable_over_update_rate': 0.00029507229271171436, 'test_stable_over_update_rate': 0.00044229521731438344, 'val_gate_order_ok': True, 'test_gate_order_ok': True}, 'seed456': {'val_stable_gate_mean': 0.4754994103957081, 'test_stable_gate_mean': 0.5696893329432517, 'val_stable_over_open_rate': 0.9316022425494246, 'test_stable_over_open_rate': 0.964144601049714, 'val_stable_over_update_rate': 0.00029507229271171436, 'test_stable_over_update_rate': 0.00044229521731438344, 'val_gate_order_ok': True, 'test_gate_order_ok': True}}`
- 阶段性分析: `seed123 与 seed456 是复现实验，不是继续修 bug；当前复现重点是验证 activation-state gate probe 的跨 seed 稳定性和干预因果性，同时明确 gate 过开风险。`
- 论文相关问题解决方案参考: `建议继续沿用 residual main-path preservation、counterfactual intervention、slot/object-memory assignment ablation 的评估框架；若进入下一轮修复，应优先做 gate calibration / sparse gate regularization，而不是扩大模型或改 trajectory backbone。`
- integrated_semantic_field_claim_allowed: `False`
- recommended_next_step: `stop_and_analyze_claim_boundary`
