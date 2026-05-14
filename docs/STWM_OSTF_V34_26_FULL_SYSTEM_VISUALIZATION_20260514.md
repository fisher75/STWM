# V34.26 full-system visualization 中文报告

## 中文结论
V34.26 visualization 已完成；图像展示 baseline 排名、干预 delta、M128 future trace + sparse semantic gate overlay，不声明 semantic field success。

## 图像清单
- baseline_hard_changed_gain: `outputs/visualizations/stwm_ostf_v34_26_full_system_benchmark_20260514/v34_26_baseline_hard_changed_gain.png`；原因：比较 V34.25 与非 oracle baseline 的 hard/changed gain。
- v3425_intervention_deltas: `outputs/visualizations/stwm_ostf_v34_26_full_system_benchmark_20260514/v34_26_v3425_intervention_deltas.png`；原因：验证 selected sparse gate 下 semantic/assignment/unit memory 仍为因果路径。
- m128_future_trace_sparse_gate_overlay: `outputs/visualizations/stwm_ostf_v34_26_full_system_benchmark_20260514/v34_26_m128_future_trace_sparse_gate_overlay.png`；原因：从 val eval 中选 hard/changed gain 最高样本，展示 M128 future trace 与 sparse semantic gate 叠加。

## 关键字段
- visualization_ready: `True`
- recommended_next_step: `fix_full_system_baseline_gap`
