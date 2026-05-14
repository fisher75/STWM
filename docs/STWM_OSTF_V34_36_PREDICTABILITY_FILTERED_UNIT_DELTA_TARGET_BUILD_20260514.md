# V34.36 predictability-filtered unit_delta target build 中文报告

- 中文结论: `V34.36 predictability-filtered unit_delta target 已构建；将原始 oracle unit_delta 按 observed-only ridge 可预测性过滤/收缩，只监督可预测 correction 分量。`
- target_built: `True`
- target_root: `outputs/cache/stwm_ostf_v34_36_predictability_filtered_unit_delta_targets/pointodyssey`
- ridge_lambda: `100.0`
- predictability_cos_threshold: `0.15`
- split_stats: `{'train': {'original_active_count': 39149, 'filtered_active_count': 38762, 'filtered_active_ratio_vs_original': 0.9901146900303966, 'point_predictable_ratio_vs_hard_changed': 0.9969399918162571, 'direction_cosine_mean_on_original_active': 0.740561306476593, 'direction_cosine_p50_on_original_active': 0.7847796678543091}, 'val': {'original_active_count': 22665, 'filtered_active_count': 15249, 'filtered_active_ratio_vs_original': 0.6727994705493051, 'point_predictable_ratio_vs_hard_changed': 0.7607776519441298, 'direction_cosine_mean_on_original_active': 0.20529082417488098, 'direction_cosine_p50_on_original_active': 0.199788898229599}, 'test': {'original_active_count': 53424, 'filtered_active_count': 33570, 'filtered_active_ratio_vs_original': 0.6283692722371967, 'point_predictable_ratio_vs_hard_changed': 0.710620227951754, 'direction_cosine_mean_on_original_active': 0.18664149940013885, 'direction_cosine_p50_on_original_active': 0.18197640776634216}}`
