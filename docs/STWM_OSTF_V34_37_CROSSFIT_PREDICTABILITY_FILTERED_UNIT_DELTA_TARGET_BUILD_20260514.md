# V34.37 crossfit predictability target build 中文报告

- 中文结论: `V34.37 cross-fitted predictability-filtered targets 已构建；train split 使用 out-of-fold ridge，避免 in-sample predictability 过宽。`
- target_built: `True`
- crossfit_folds: `4`
- ridge_lambda: `100.0`
- predictability_cos_threshold: `0.15`
- split_stats: `{'train': {'original_active_count': 39149, 'filtered_active_count': 29731, 'filtered_active_ratio_vs_original': 0.7594319139697054, 'point_predictable_ratio_vs_hard_changed': 0.8932911099645964, 'direction_cosine_mean_on_original_active': 0.24807637929916382, 'direction_cosine_p50_on_original_active': 0.2487800419330597}, 'val': {'original_active_count': 22665, 'filtered_active_count': 15249, 'filtered_active_ratio_vs_original': 0.6727994705493051, 'point_predictable_ratio_vs_hard_changed': 0.7607776519441298, 'direction_cosine_mean_on_original_active': 0.20529083907604218, 'direction_cosine_p50_on_original_active': 0.1997889131307602}, 'test': {'original_active_count': 53424, 'filtered_active_count': 33570, 'filtered_active_ratio_vs_original': 0.6283692722371967, 'point_predictable_ratio_vs_hard_changed': 0.710620227951754, 'direction_cosine_mean_on_original_active': 0.18664149940013885, 'direction_cosine_p50_on_original_active': 0.18197639286518097}}`
