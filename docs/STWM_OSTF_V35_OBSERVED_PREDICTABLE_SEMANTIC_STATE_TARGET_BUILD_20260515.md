# STWM OSTF V35 可观测可预测语义状态 target 构建

- target root: `outputs/cache/stwm_ostf_v35_observed_predictable_semantic_state_targets/pointodyssey`
- semantic_cluster_count: 64
- target_coverage_by_split: {'train': 0.7639141082763672, 'val': 0.778642578125, 'test': 0.7518433339497042}
- stable_changed_ratio_by_split: {'train': {'stable': 0.43041264784238137, 'changed': 0.6600892360010087}, 'val': {'stable': 0.4133002228270185, 'changed': 0.6916834936600905}, 'test': {'stable': 0.4043451360761196, 'changed': 0.714772386654472}}
- semantic_hard_ratio_by_split: {'train': 0.15224800317594273, 'val': 0.21893904238730094, 'test': 0.24319427258264098}
- identity_target_coverage_by_split: {'train': 0.7639141082763672, 'val': 0.7781575520833334, 'test': 0.7517118736131657}
- leakage_safe: True
- exact_blockers: []

## 中文总结
V35 target suite 已从 continuous teacher embedding delta 改为 semantic cluster transition、stable/changed/hard、coarse evidence anchor family、same-instance consistency 与 uncertainty/confidence targets。future teacher embedding 只用于监督，不进入输入。
