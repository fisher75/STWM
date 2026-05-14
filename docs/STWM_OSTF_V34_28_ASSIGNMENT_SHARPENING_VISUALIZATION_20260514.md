# V34.28 assignment sharpening visualization 中文报告

## 中文结论
V34.28 visualization 已完成；图像展示 assignment sharpening 是否恢复 assignment load-bearing。

## 图像清单
- assignment_variant_gain: `outputs/visualizations/stwm_ostf_v34_28_assignment_sharpening_20260514/v34_28_assignment_variant_gain.png`；原因：比较 soft/power/top-k assignment 读出对 evidence-anchor residual 增益的影响。
- assignment_loadbearing_delta: `outputs/visualizations/stwm_ostf_v34_28_assignment_sharpening_20260514/v34_28_assignment_loadbearing_delta.png`；原因：检测 sharpen/top-1 assignment 是否让 residual correction 对 assignment counterfactual 敏感。
