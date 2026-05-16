# STWM OSTF V35.1 修复版可观测可预测语义状态 target 构建

- target root: `outputs/cache/stwm_ostf_v35_1_fixed_semantic_state_targets/pointodyssey`
- semantic_cluster_count: 64
- target_coverage_by_split: {'train': 0.7639141082763672, 'val': 0.778642578125, 'test': 0.7518433339497042}
- stable_changed_ratio_by_split: {'train': {'stable': 0.43041264784238137, 'changed': 0.6600892360010087}, 'val': {'stable': 0.4133002228270185, 'changed': 0.6916834936600905}, 'test': {'stable': 0.4043451360761196, 'changed': 0.714772386654472}}
- semantic_hard_ratio_by_split: {'train': 0.15224800317594273, 'val': 0.21893904238730094, 'test': 0.24319427258264098}
- identity_target_coverage_by_split: {'train': 0.7637042999267578, 'val': 0.7778125, 'test': 0.7514070589866864}
- identity_positive_ratio_by_split: {'train': 0.9531145027110322, 'val': 0.9381110218293827, 'test': 0.9098111669505635}
- uncertainty_high_ratio_by_split: {'train': 0.0, 'val': 0.0, 'test': 0.0}
- leakage_safe: True
- exact_blockers: []

## 中文总结
V35.1 修复了 V35 target suite：identity consistency 改用真实 fut_same_instance_as_obs 正负样本；uncertainty 改为 observed-risk 主导的 calibrated abstain/risk；evidence anchor family 改为更粗的 last/max-confidence/instance-pooled/changed/abstain 状态族。future teacher embedding 只用于监督，不进入输入。
