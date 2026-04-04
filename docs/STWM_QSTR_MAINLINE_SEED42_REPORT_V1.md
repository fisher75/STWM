# STWM QSTR Mainline Seed42 Report V1

Generated: 2026-04-04 20:19:36
Submit TSV: /home/chen034/workspace/stwm/reports/stwm_qstr_mainline_seed42_submit_v1.tsv
Queue status dir: /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status

## Monitoring Summary

- all_entered_running: True
- all_train_log_growth_observed: True
- all_sidecar_generated: True
- all_terminal(done/failed): True

## Runs

| run | unique_change_point | state | max_step | train_log_rows | sidecar | job_id | status_file | main_log |
|---|---|---|---:|---:|---|---|---|---|
| trace_sem_baseline_seed42_qstr_control_v1 | non-object baseline: --neutralize-object-bias | done | 2000 | 2000 | True | 20260404_172250_22511 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775294570125_trace_sem_baseline_seed42_qstr_control_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775294570125_trace_sem_baseline_seed42_qstr_control_v1.log |
| qstr_only_seed42_challenge_v1 | qstr-only: query-conditioned semantic residual routing + neutral path | done | 2000 | 2000 | True | 20260404_172250_32008 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775294570485_qstr_only_seed42_challenge_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775294570485_qstr_only_seed42_challenge_v1.log |
| qstr_temporal_consistency_seed42_challenge_v1 | qstr + temporal semantic consistency regularization | done | 2000 | 2000 | True | 20260404_172250_24376 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775294570704_qstr_temporal_consistency_seed42_challenge_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775294570704_qstr_temporal_consistency_seed42_challenge_v1.log |

## Official Ranking (Within This 3-Run QSTR Matrix)

| rank | run | query_localization_error | query_top1_acc | future_trajectory_l1 | future_mask_iou | identity_consistency | identity_switch_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | trace_sem_baseline_seed42_qstr_control_v1 | 0.002259 | 0.979644 | 0.002430 | na | na | na |
| 2 | qstr_temporal_consistency_seed42_challenge_v1 | 0.004972 | 0.944020 | 0.005124 | na | na | na |
| 3 | qstr_only_seed42_challenge_v1 | 0.006653 | 0.936387 | 0.006760 | na | na | na |

## Provisional Conclusion

- best_run_by_official_rule: trace_sem_baseline_seed42_qstr_control_v1
- matrix_complete: True

