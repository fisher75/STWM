# STWM Replacement Clean Matrix Seed42 Report V1

Generated: 2026-04-04 12:44:16
Queue status dir: /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status
Run root: /home/chen034/workspace/stwm/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1/seed_42

## Monitoring Summary

- all_entered_running: True
- all_train_log_growth_observed: True
- all_sidecar_generated: True
- all_terminal(done/failed): True

## Runs

| run | state | max_step | train_log_rows | growth_observed | sidecar | job_id | status_file | main_log |
|---|---|---:|---:|---|---|---|---|---|
| full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2 | done | 2000 | 2448 | True | True | 20260404_114420_721 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775274260546_full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775274260546_full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2.log |
| full_v4_2_seed42_objbias_alpha050_replacement_v1 | done | 2000 | 2470 | True | True | 20260404_114420_14383 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775274260881_full_v4_2_seed42_objbias_alpha050_replacement_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775274260881_full_v4_2_seed42_objbias_alpha050_replacement_v1.log |
| wo_semantics_v4_2_seed42_control_v1 | done | 2000 | 2109 | True | True | 20260404_114421_22183 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775274261057_wo_semantics_v4_2_seed42_control_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775274261057_wo_semantics_v4_2_seed42_control_v1.log |
| wo_object_bias_v4_2_seed42_control_v1 | done | 2000 | 2103 | True | True | 20260404_114421_7606 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775274261294_wo_object_bias_v4_2_seed42_control_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775274261294_wo_object_bias_v4_2_seed42_control_v1.log |

## Official Ranking Snapshot

| rank | run | query_localization_error | query_top1_acc | future_trajectory_l1 |
|---|---|---:|---:|---:|
| 1 | wo_object_bias_v4_2_seed42_control_v1 | 0.002259 | 0.979644 | 0.002430 |
| 2 | full_v4_2_seed42_objbias_alpha050_replacement_v1 | 0.005062 | 0.956743 | 0.005183 |
| 3 | full_v4_2_seed42_fixed_nowarm_lambda1_rerun_v2 | 0.006695 | 0.926209 | 0.006538 |
| 4 | wo_semantics_v4_2_seed42_control_v1 | 0.008401 | 0.895674 | 0.008473 |

