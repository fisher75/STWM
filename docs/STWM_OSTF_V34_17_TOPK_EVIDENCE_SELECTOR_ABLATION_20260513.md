# V34.17 top-k evidence selector ablation 中文报告

- 中文结论: `V34.17 top-k evidence selector ablation 已完成；不训练新模型，仅用 V34.14 horizon-conditioned 权重评估 top-k memory set 是否比全 soft sum 更合理。`
- topk_ablation_done: `True`
- base_selector: `v34_14_horizon_conditioned_soft_reader`
- best_topk_by_val: `{'name': 'top8', 'val': {'cosine': {'valid': 0.5592285754356205, 'hard': 0.5242023869372971, 'changed': 0.4007720805388309, 'strict': 0.28849511958374624, 'causal': 0.2597066055851657}, 'minus_pointwise': {'valid': 0.18976999064455746, 'hard': 0.15481108774181904, 'changed': 0.0787235138774436, 'strict': 0.1640532054735517, 'causal': 0.1897370248264433}, 'oracle_gap': {'valid': 0.10411684829825271, 'hard': 0.09244253651340521, 'changed': 0.0957104654067192, 'strict': 0.09896536769845343, 'causal': 0.07535153929119237}}, 'test': {'cosine': {'valid': 0.5428739203629497, 'hard': 0.5135697544731036, 'changed': 0.4002103812456573, 'strict': 0.29759261475419846, 'causal': 0.20202734875764064}, 'minus_pointwise': {'valid': 0.21218136543929317, 'hard': 0.17623984527461306, 'changed': 0.0998522353519952, 'strict': 0.1575304621073529, 'causal': 0.09591192932673215}, 'oracle_gap': {'valid': 0.1097170616354137, 'hard': 0.1014227873644082, 'changed': 0.09707786817998058, 'strict': 0.09582883060882086, 'causal': 0.06909927658462071}}}`
- best_topk_improves_hard_changed_vs_pointwise: `True`
- recommended_fix: `如果后续进入 residual，应让 residual 读取 top-k evidence set，而不是单个 selected vector 或 top-1 timestep label。`
- recommended_next_step: `build_topk_evidence_conditioned_residual_probe`
