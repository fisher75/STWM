# STWM Object Bias Promotion Decision V1

Generated: 2026-04-04 02:05:09
Seed: 42
Baseline: full_v4_2_seed42_fixed_nowarm_lambda1_objdiag_v1

## Completion Validation

- all_runs_done: True
- all_runs_reached_expected_steps: True
- all_selection_sidecars_present: True
- watcher_matrix_report_present: True

## Official Ranking

Rule: query_localization_error asc, query_top1_acc desc, future_trajectory_l1 asc

| rank | run | selected_best_step | query_localization_error | query_top1_acc | future_trajectory_l1 | future_mask_iou |
|---|---|---:|---:|---:|---:|---:|
| 1 | full_v4_2_seed42_objbias_alpha050_objdiag_v1 | 1200 | 0.004615 | 0.961832 | 0.004739 | 0.160491 |
| 2 | full_v4_2_seed42_fixed_nowarm_lambda1_objdiag_v1 | 1200 | 0.004738 | 0.936387 | 0.004592 | 0.160508 |
| 3 | full_v4_2_seed42_objbias_gated_objdiag_v1 | 1200 | 0.004738 | 0.936387 | 0.004592 | 0.160508 |
| 4 | wo_object_bias_v4_2_seed42_objdiag_v1 | 1200 | 0.004832 | 0.969466 | 0.004966 | 0.160501 |
| 5 | full_v4_2_seed42_objbias_alpha025_objdiag_v1 | 1200 | 0.005220 | 0.959288 | 0.005351 | 0.160504 |
| 6 | full_v4_2_seed42_objbias_delayed200_objdiag_v1 | 1200 | 0.006126 | 0.933842 | 0.006189 | 0.160513 |

## Promotion Verdict

- best_variant: full_v4_2_seed42_objbias_alpha050_objdiag_v1
- best_vs_baseline_significant: False
- recommend_promotion: False
- reason: best_variant_not_significant_vs_baseline(all_three_better=False, rel_q=0.0259, rel_f=-0.0320, abs_top1=0.0254)

