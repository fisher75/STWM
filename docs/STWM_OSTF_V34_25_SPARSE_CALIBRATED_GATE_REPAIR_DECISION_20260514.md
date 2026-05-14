# V34.25 sparse-calibrated gate repair 决策中文总结

- 中文结论: `V34.25 sparse-calibrated gate repair 已完成；本轮只修 gate calibration/sparsity，V30 与 residual content 保持 frozen，不跑 H64/H96，不声明 semantic field success。`
- sparse_calibrated_gate_repair_ran: `True`
- sparse_calibrated_gate_repair_passed: `True`
- all_seeds_passed: `True`
- semantic_hard_signal: `{'val': True, 'test': True}`
- changed_semantic_signal: `{'val': True, 'test': True}`
- stable_preservation: `{'val': True, 'test': True}`
- stable_overopen_controlled: `{'val': True, 'test': True}`
- stable_over_open_rate: `{'val': {'mean': 0.18776433559555425, 'std': 0.007730268360322995, 'min': 0.17910888167601063, 'max': 0.19787547949247566}, 'test': {'mean': 0.3154744353364392, 'std': 0.013006187037044968, 'min': 0.29981718464351004, 'max': 0.3316624402901457}}`
- stable_over_update_rate: `{'val': {'mean': 0.00023605783416937148, 'std': 0.0, 'min': 0.00023605783416937148, 'max': 0.00023605783416937148}, 'test': {'mean': 0.00021623321735369857, 'std': 6.0588685134644266e-05, 'min': 0.0001474317391047945, 'max': 0.000294863478209589}}`
- semantic_measurements_load_bearing_on_residual: `True`
- assignment_load_bearing_on_residual: `True`
- unit_memory_load_bearing_on_residual: `True`
- claim_boundary: `如果本轮通过，只能 claim sparse-calibrated residual gate repair 在 M128/H32 多 seed 上减轻 stable over-open，仍不能 claim integrated semantic field success 或 identity field success。`
- integrated_semantic_field_claim_allowed: `False`
- recommended_next_step: `run_v34_25_claim_boundary_visualization`
