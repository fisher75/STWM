# V34.12 V34.11 measurement truth audit 中文报告

- 中文结论: `V34.12 truth audit 确认：V34.11 quality/visual JSON 在当前 repo 存在；quality probe 使用 future target 做逐 token oracle best 上界，且 teacher source 名称与 final decision 的 best_measurement_bank 不一致。`
- quality_probe_json_missing: `False`
- visualization_json_missing: `False`
- final_decision_depends_on_missing_quality_json: `False`
- quality_probe_uses_oracle_best_measurement: `True`
- measurement_teacher_name_inconsistent: `True`
- local_probe_is_oracle_masked_residual: `True`
- model_forward_gate_zero_by_default: `True`
- semantic_usage_score_only_indirectly_used: `False`
- recommended_fix: `先用 non-oracle selector 重估 measurement quality；再实现 raw temporal semantic evidence encoder，避免 pooled-vector repeat 和 oracle best 过乐观。`
