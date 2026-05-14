# V34.25 sparse-calibrated gate repair 中文总结

- 中文结论: `V34.25 sparse-calibrated gate repair 已完成；本轮只修 gate calibration/sparsity，V30 与 residual content 保持 frozen，不跑 H64/H96，不声明 semantic field success。`
- seeds: `['seed42', 'seed123', 'seed456']`
- all_seeds_passed: `True`
- semantic_hard_signal: `{'val': True, 'test': True}`
- changed_semantic_signal: `{'val': True, 'test': True}`
- stable_preservation: `{'val': True, 'test': True}`
- stable_overopen_controlled: `{'val': True, 'test': True}`
- stable_over_open_rate: `{'val': {'mean': 0.18776433559555425, 'std': 0.007730268360322995, 'min': 0.17910888167601063, 'max': 0.19787547949247566}, 'test': {'mean': 0.3154744353364392, 'std': 0.013006187037044968, 'min': 0.29981718464351004, 'max': 0.3316624402901457}}`
- stable_over_update_rate: `{'val': {'mean': 0.00023605783416937148, 'std': 0.0, 'min': 0.00023605783416937148, 'max': 0.00023605783416937148}, 'test': {'mean': 0.00021623321735369857, 'std': 6.0588685134644266e-05, 'min': 0.0001474317391047945, 'max': 0.000294863478209589}}`
- hard_changed_gain: `{'val': {'mean': 0.10542878104247234, 'std': 0.001566648670797119, 'min': 0.10372560060108238, 'max': 0.10750754866461162}, 'test': {'mean': 0.09398295335273571, 'std': 0.001705889848804947, 'min': 0.09200319909315699, 'max': 0.0961667819338595}}`
- semantic_measurements_load_bearing_on_residual: `True`
- assignment_load_bearing_on_residual: `True`
- unit_memory_load_bearing_on_residual: `True`
- 阶段性分析: `V34.24 的 blocker 是 stable gate over-open。V34.25 因此不改 residual 内容、不改 V30、不扩大 horizon，只在 gate 上加入 stable-negative loss、预算稀疏约束、hard/changed recall 保底和 threshold/temperature Pareto sweep。这个设计直接对应当前失败模式：stable 输出不坏，但 gate 选择边界太松。`
- 论文相关问题解决方案参考: `本轮参考 sparse MoE / selective computation 的 gate budget 与稀疏路由思想，结合 memory-video 方法中 selective read 的原则，以及 Slot Attention/object-memory 中必须做 assignment intervention 的评价方式。`
- 最佳下一步方案: `若 V34.25 通过，下一步仍应先做 claim-boundary replication/visualization，而不是直接 H64/H96；若未通过，继续修 sparse gate calibration，不扩大模型。`
- integrated_semantic_field_claim_allowed: `False`
- recommended_next_step: `run_v34_25_claim_boundary_visualization`
