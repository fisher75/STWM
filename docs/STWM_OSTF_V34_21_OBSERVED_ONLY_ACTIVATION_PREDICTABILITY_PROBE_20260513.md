# V34.21 observed-only activation predictability probe 中文报告

- 中文结论: `V34.21 observed-only activation predictability probe 已完成；该 probe 不训练 learned gate，只检查 V34.20 的 hard/changed 发力区域是否能由观测侧特征预测。`
- probe_ran: `True`
- v30_backbone_frozen: `True`
- future_leakage_detected: `False`
- uses_future_teacher_as_input: `False`
- aligned_activation_predictable: `False`
- utility_activation_predictable: `False`
- benefit_activation_predictable: `False`
- gate_predictability_passed: `False`
- target_positive_ratios: `{'train': {'aligned': 0.14034320414066315, 'utility_margin_positive': 0.14033322036266327, 'benefit_positive': 0.14033572375774384}, 'val': {'aligned': 0.11074461042881012, 'utility_margin_positive': 0.10752135515213013, 'benefit_positive': 0.10845781117677689}, 'test': {'aligned': 0.1389121562242508, 'utility_margin_positive': 0.13521915674209595, 'benefit_positive': 0.13630668818950653}}`
- recommended_next_step: `fix_observed_only_activation_features`
