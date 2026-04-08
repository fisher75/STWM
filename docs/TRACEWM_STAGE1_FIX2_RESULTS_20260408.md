# TraceWM Stage 1 Fix Round 2 Results (2026-04-08)

- generated_at_utc: 2026-04-08T04:00:03.422384+00:00
- round2_doc: /home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_MODEL_FIX_ROUND2_20260408.md
- comparison_json: /home/chen034/workspace/stwm/reports/tracewm_stage1_fix2_comparison_20260408.json

## Run Metrics

| run | val_total_loss | tf_free_gap | tapvid_free_endpoint_l2 | tapvid3d_limited_free_endpoint_l2 | score_vs_best_single |
|---|---:|---:|---:|---:|---:|
| tracewm_stage1_fix2_joint_balanced_lossnorm | 0.000001 | -0.000000 | 0.263109 | 11.687996 | -2.434776 |
| tracewm_stage1_fix2_point_warmup_then_joint_balanced_lossnorm | 0.000002 | -0.000000 | 0.263682 | 11.471092 | -7.905773 |
| tracewm_stage1_fix2_kubric_warmup_then_joint_balanced_lossnorm | 0.000004 | -0.000000 | 0.260837 | 11.863145 | -21.783505 |

## Required Answers

1. balanced+lossnorm itself surpasses best single: False
2. point warmup better than no-warmup: False
3. kubric warmup better than no-warmup: False
4. best joint recipe among three: tracewm_stage1_fix2_joint_balanced_lossnorm
5. any_joint_surpasses_best_single: False
6. if no surpass recommendation: stop_joint_and_keep_best_single
7. if has surpass recommendation: None

- final_recommendation: stop_joint_and_keep_best_single
