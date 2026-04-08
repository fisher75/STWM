# TraceWM Stage 1 Final Rescue Results (2026-04-08)

- generated_at_utc: 2026-04-08T04:28:29.417387+00:00
- freeze_doc: /home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_FINAL_JOINT_RESCUE_20260408.md
- selected_best_gradient_method_for_combo: pcgrad
- comparison_json: /home/chen034/workspace/stwm/reports/tracewm_stage1_final_rescue_comparison_20260408.json

## Run Metrics

| run | train_mode | val_total_loss | tapvid_free_endpoint_l2 | tapvid3d_limited_free_endpoint_l2 | private_parameter_count | score_vs_best_single |
|---|---|---:|---:|---:|---:|---:|
| tracewm_stage1_rescue_pcgrad | pcgrad | 0.000001 | 0.263380 | 11.632824 | 0 | -2.205434 |
| tracewm_stage1_rescue_gradnorm | gradnorm | 0.000001 | 0.263777 | 11.764711 | 0 | -2.602337 |
| tracewm_stage1_rescue_shared_private | shared_private | 0.000001 | 0.363989 | 13.660166 | 8480 | -2.652275 |
| tracewm_stage1_rescue_shared_private_plus_best_grad | shared_private_plus_pcgrad | 0.000001 | 0.362240 | 13.662463 | 8480 | -5.729593 |

## Required Answers

1. q1_best_rescue_run: tracewm_stage1_rescue_pcgrad
2. q2_any_rescue_surpasses_best_single: False
3. q3_best_gradient_method: pcgrad
4. q4_shared_private_helpful: False
5. q5_shared_private_plus_best_grad_helpful: False
6. q6_best_on_tapvid: tracewm_stage1_rescue_pcgrad
7. q7_best_on_tapvid3d_limited: tracewm_stage1_rescue_pcgrad
8. q8_final_joint_decision: stop_joint_and_keep_best_single
next_step_choice: stop_joint_and_keep_best_single
