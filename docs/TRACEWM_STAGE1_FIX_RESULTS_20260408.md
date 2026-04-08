# TraceWM Stage 1 Fix Results (2026-04-08)

- generated_at_utc: 2026-04-08T03:46:22.924275+00:00
- diagnosis_json: /home/chen034/workspace/stwm/reports/tracewm_stage1_iter1_diagnosis_20260408.json
- comparison_json: /home/chen034/workspace/stwm/reports/tracewm_stage1_fix_comparison_20260408.json

## Fix Run Metrics

| run | val_total_loss | tf_free_gap | tapvid_free_endpoint_l2 | tapvid3d_limited_free_endpoint_l2 | effectiveness_score |
|---|---:|---:|---:|---:|---:|
| tracewm_stage1_fix_joint_balanced_sampler | 0.000001 | -0.000000 | 0.263256 | 11.769157 | 0.619755 |
| tracewm_stage1_fix_joint_loss_normalized | 0.000001 | -0.000000 | 0.261579 | 11.755750 | 0.278901 |
| tracewm_stage1_fix_joint_source_conditioned | 0.000001 | -0.000000 | 0.268903 | 13.746309 | 0.529810 |

## Required Answers

1. Most effective fix: tracewm_stage1_fix_joint_balanced_sampler
2. Any fix surpasses best single: False
3. Best for TAP-Vid: tracewm_stage1_fix_joint_loss_normalized
4. Best for TAPVid-3D limited: tracewm_stage1_fix_joint_loss_normalized
5. TF/free gap improvement: {'tracewm_stage1_fix_joint_balanced_sampler': {'abs_gap': 1.5631044902875146e-08, 'improved_vs_iter1_joint': True}, 'tracewm_stage1_fix_joint_loss_normalized': {'abs_gap': 4.472471459848748e-08, 'improved_vs_iter1_joint': True}, 'tracewm_stage1_fix_joint_source_conditioned': {'abs_gap': 4.664535424581118e-09, 'improved_vs_iter1_joint': True}}
6. Next recommendation: continue_stage1_model_fix
