# TraceWM Stage 1 Model-Fix Round (2026-04-08)

## Scope Lock

- Stage 1 trace-only only.
- No Stage 2, no WAN/MotionCrafter VAE, no DynamicReplica, no data expansion.
- Diagnose first, then apply exactly three minimal joint fixes.

## Input Evidence

- point_summary: /home/chen034/workspace/stwm/reports/tracewm_stage1_iter1_pointodyssey_only_summary.json
- kubric_summary: /home/chen034/workspace/stwm/reports/tracewm_stage1_iter1_kubric_only_summary.json
- joint_summary: /home/chen034/workspace/stwm/reports/tracewm_stage1_iter1_joint_po_kubric_summary.json
- comparison: /home/chen034/workspace/stwm/reports/tracewm_stage1_iter1_comparison_20260408.json
- iter1_log: /home/chen034/workspace/stwm/logs/tracewm_stage1_iter1_20260408.log

## Diagnosis Conclusions

- joint early/late behavior: joint_lags_early_and_persists
- early_avg_gap_vs_best_single: 0.002085203514
- late_avg_gap_vs_best_single: 0.000002277065
- point proxy conclusion: joint_better_than_point_on_proxy
- kubric proxy conclusion: tradeoff_joint_improves_3d_proxy_but_degrades_tapvid_vs_kubric_single
- loss scale imbalance suspected: True (ratio=3.000)
- sampler bias suspected: True (ratio=3.000)

## Exactly Three Fix Experiments

- tracewm_stage1_fix_joint_balanced_sampler: sampler only.
- tracewm_stage1_fix_joint_loss_normalized: loss normalization only.
- tracewm_stage1_fix_joint_source_conditioned: source conditioning only.

- diagnosis_json: /home/chen034/workspace/stwm/reports/tracewm_stage1_iter1_diagnosis_20260408.json
