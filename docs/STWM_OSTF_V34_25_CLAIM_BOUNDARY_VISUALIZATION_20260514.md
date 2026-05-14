# V34.25 claim-boundary visualization 中文总结

## 中文结论
V34.25 claim-boundary visualization 已完成；图像只用于说明 sparse gate calibration 如何降低 stable over-open，以及 selected gate 下 semantic/assignment/unit 干预仍然 load-bearing，不用于声明 semantic field success。

## 生成图像
- stable_overopen_before_after: `outputs/visualizations/stwm_ostf_v34_25_claim_boundary_20260514/v34_25_stable_overopen_before_after.png`；原因：展示 V34.25 sparse calibration 是否实质降低 V34.23 的 stable gate over-open 风险。
- seed123_calibration_pareto: `outputs/visualizations/stwm_ostf_v34_25_claim_boundary_20260514/v34_25_seed123_calibration_pareto.png`；原因：展示阈值/温度校准下 hard/changed gain 与 stable over-open 的 Pareto 权衡。
- seed42_calibration_pareto: `outputs/visualizations/stwm_ostf_v34_25_claim_boundary_20260514/v34_25_seed42_calibration_pareto.png`；原因：展示阈值/温度校准下 hard/changed gain 与 stable over-open 的 Pareto 权衡。
- seed456_calibration_pareto: `outputs/visualizations/stwm_ostf_v34_25_claim_boundary_20260514/v34_25_seed456_calibration_pareto.png`；原因：展示阈值/温度校准下 hard/changed gain 与 stable over-open 的 Pareto 权衡。
- intervention_deltas: `outputs/visualizations/stwm_ostf_v34_25_claim_boundary_20260514/v34_25_intervention_deltas.png`；原因：展示 selected sparse gate 下 semantic / assignment / unit memory 干预仍为 load-bearing。

## 阶段性分析
V34.25 解决的是 V34.24 暴露的 gate calibration/sparsity 风险。可视化显示 test stable over-open 从 V34.23 的高过开区间降到 V34.25 的受控区间，同时 hard/changed gain 与干预 delta 保留。

## 最佳下一步方案
下一步应停下来做 V34.25 claim-boundary 总结与 external baseline/视频输入闭环计划，仍不要 claim semantic field success。

## 关键字段
- visualization_ready: `True`
- integrated_semantic_field_claim_allowed: `False`
- recommended_next_step: `stop_and_prepare_v34_25_claim_boundary_summary`
