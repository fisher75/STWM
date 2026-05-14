# V34.39 prototype-blended unit_delta target build 中文报告

- 中文结论: `V34.39 prototype-blended unit_delta target 已构建；不是硬替换 centroid，而是在 V34.37 crossfit target 上加入轻度 prototype smoothing，保留 cached upper bound 的同时降低 sample-specific 噪声。`
- target_built: `True`
- target_root: `outputs/cache/stwm_ostf_v34_39_prototype_blended_unit_delta_targets/pointodyssey`
- base_target_root: `outputs/cache/stwm_ostf_v34_37_crossfit_predictability_filtered_unit_delta_targets/pointodyssey`
- prototype_target_root: `outputs/cache/stwm_ostf_v34_38_cluster_regularized_unit_delta_targets/pointodyssey`
- blend_alpha: `0.9`
- split_stats: `{'train': {'active_count': 29731, 'point_positive_count': 50211, 'point_positive_ratio_all_tokens': 0.09576988220214844, 'blended_delta_norm_mean': 0.15411020815372467, 'base_delta_norm_mean_on_active': 0.1702384203672409, 'prototype_delta_norm_mean_on_active': 0.017212921753525734}, 'val': {'active_count': 15249, 'point_positive_count': 20153, 'point_positive_ratio_all_tokens': 0.06560221354166666, 'blended_delta_norm_mean': 0.11137209832668304, 'base_delta_norm_mean_on_active': 0.12327616661787033, 'prototype_delta_norm_mean_on_active': 0.012210647575557232}, 'test': {'active_count': 33570, 'point_positive_count': 51375, 'point_positive_ratio_all_tokens': 0.07421730538091716, 'blended_delta_norm_mean': 0.09016016125679016, 'base_delta_norm_mean_on_active': 0.09985543042421341, 'prototype_delta_norm_mean_on_active': 0.01000511646270752}}`
