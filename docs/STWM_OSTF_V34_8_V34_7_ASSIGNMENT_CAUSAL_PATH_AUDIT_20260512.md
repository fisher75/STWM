# V34.8 对 V34.7 assignment 因果路径的中文审计

- 中文结论: `V34.7 的 assignment 形式路径存在，但因果 load-bearing 未成立；target 正样本过宽，semantic measurement 与 assignment intervention 在 test 上不足。`
- artifact_packaging_truly_fixed: `True`
- assignment_target_json_missing: `False`
- visualization_json_missing: `False`
- assignment_path_formally_present: `True`
- assignment_path_causally_load_bearing: `False`
- semantic_measurement_causally_load_bearing: `False`
- target_positive_ratio_too_broad: `True`
- target_positive_ratio_by_split: `{'test': 0.446871196193427, 'train': 0.914088491166895, 'val': 0.5454133635334089}`
- assignment_shortcut_suspected: `True`
- semantic_shortcut_suspected: `True`
- recommended_fix: `构建更严格的 causal assignment residual targets，并重写 residual memory，禁止 pointwise/global shortcut；先跑 oracle probe，不允许先训练 learned gate。`
