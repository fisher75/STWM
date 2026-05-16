# STWM OSTF V35.2 Identity Confuser Assignment Target Build

- identity_confuser_assignment_targets_built: True
- target_root: `outputs/cache/stwm_ostf_v35_2_identity_confuser_assignment_targets/pointodyssey`
- split_reports: {'train': {'samples': 128, 'pair_available': 2080768, 'same_pair_ratio': 0.2512447327140748, 'confuser_pair_ratio': 0.07337867556594488, 'assignment_positive_pair_ratio': 0.30686458077017714, 'assignment_negative_pair_ratio': 0.07337867556594488, 'identity_confuser_token_ratio': 0.5960260766870323, 'hard_positive_token_ratio': 0.8023824564119337, 'hard_negative_token_ratio': 0.044965057139504286}, 'val': {'samples': 75, 'pair_available': 1217682, 'same_pair_ratio': 0.24757038372908527, 'confuser_pair_ratio': 0.09081517177719635, 'assignment_positive_pair_ratio': 0.32051389443220807, 'assignment_negative_pair_ratio': 0.09081517177719635, 'identity_confuser_token_ratio': 0.6726574943875183, 'hard_positive_token_ratio': 0.8278671733577482, 'hard_negative_token_ratio': 0.060497744555788276}, 'test': {'samples': 169, 'pair_available': 2746248, 'same_pair_ratio': 0.19273496057165995, 'confuser_pair_ratio': 0.09483265895869565, 'assignment_positive_pair_ratio': 0.2697585942711656, 'assignment_negative_pair_ratio': 0.09483265895869565, 'identity_confuser_token_ratio': 0.70047113618372, 'hard_positive_token_ratio': 0.805479552074767, 'hard_negative_token_ratio': 0.0876732174835333}}
- exact_blockers: []

## 中文总结
V35.2 target 为 identity consistency 构造 same-instance positives 与 semantic/trajectory confuser hard negatives，并为 assignment 提供正负 pair supervision。该 target 不使用 future teacher embedding 作为输入。
