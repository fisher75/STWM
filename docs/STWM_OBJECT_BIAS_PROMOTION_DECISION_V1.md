# STWM Object Bias Promotion Decision V1

Generated: 2026-04-04 10:55:15

## A) Matrix Completion Check

- all_done: True
- all_reached_target_step(>=1200): True
- all_selection_sidecar_present: True
- watcher_report_exists: True (/home/chen034/workspace/stwm/reports/stwm_object_bias_diag_matrix_v1.json)
- watcher_doc_exists: True (/home/chen034/workspace/stwm/docs/STWM_OBJECT_BIAS_DIAG_MATRIX_REPORT_V1.md)
- matrix_complete: True

## B) Official Ranking

| Rank | Variant | selected_best_step | query_localization_error | query_top1_acc | future_trajectory_l1 | future_mask_iou |
|---|---|---:|---:|---:|---:|---:|
| 1 | full_v4_2_seed42_objbias_alpha050_objdiag_v1 | 1200 | 0.004615304541 | 0.961832061069 | 0.004738580260 | 0.160491347754 |
| 2 | full_v4_2_seed42_fixed_nowarm_lambda1_objdiag_v1 | 1200 | 0.004738058184 | 0.936386768448 | 0.004591513989 | 0.160507558278 |
| 3 | full_v4_2_seed42_objbias_gated_objdiag_v1 | 1200 | 0.004738058184 | 0.936386768448 | 0.004591513989 | 0.160507558278 |
| 4 | wo_object_bias_v4_2_seed42_objdiag_v1 | 1200 | 0.004831898000 | 0.969465648855 | 0.004966347756 | 0.160501194115 |
| 5 | full_v4_2_seed42_objbias_alpha025_objdiag_v1 | 1200 | 0.005220116095 | 0.959287531807 | 0.005351031005 | 0.160503676239 |
| 6 | full_v4_2_seed42_objbias_delayed200_objdiag_v1 | 1200 | 0.006126386670 | 0.933842239186 | 0.006188759277 | 0.160512917642 |

## C) Promotion Verdict

- 当前最佳变体: full_v4_2_seed42_objbias_alpha050_objdiag_v1
- 相对 current full_nowarm 变化: query_loc_rel_improve=2.5908%, query_top1_delta=2.5445%, future_traj_rel_regress=3.2030%, future_mask_iou_drop=0.000016210524
- 是否显著优于 current full_nowarm: True
- 是否建议替换 current full 进入下一轮 clean matrix: True
- 结论: 建议 promotion（先 seed42 replacement，再 seed123 replication）

