# V34.27 evidence-anchored visualization 中文报告

## 中文结论
V34.27 visualization 已完成；图像展示 evidence-anchored 系统与 copy/top-k baseline 排名，以及 best 配置的干预因果 delta。

## 图像清单
- evidence_anchor_test_ranking: `outputs/visualizations/stwm_ostf_v34_27_evidence_anchored_full_system_20260514/v34_27_evidence_anchor_test_ranking.png`；原因：按 test hard/changed gain 排名，检查 evidence-anchored 系统是否超过 copy/top-k baseline。
- best_evidence_anchor_interventions: `outputs/visualizations/stwm_ostf_v34_27_evidence_anchored_full_system_20260514/v34_27_best_evidence_anchor_interventions.png`；原因：验证 best evidence-anchor 配置下 semantic/assignment/unit memory 是否仍为因果路径。

## 关键字段
- visualization_ready: `True`
- recommended_next_step: `fix_full_system_baseline_gap`
