# STWM QTSA Mainline Seed42 Report V1

Generated: 2026-04-04 23:26:57
Submit TSV: /home/chen034/workspace/stwm/reports/stwm_qtsa_mainline_seed42_submit_v1.tsv
Queue status dir: /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status

## Monitoring Summary

- all_entered_running: True
- all_train_log_growth_observed: True
- all_sidecar_generated: True
- all_terminal(done/failed): True

## Runs

| run | unique_change_point | state | max_step | train_log_rows | sidecar | job_id | status_file | main_log |
|---|---|---|---:|---:|---|---|---|---|
| trace_sem_baseline_seed42_qtsa_control_v1 | strongest non-object baseline; semantics not in transition | done | 2000 | 2000 | True | 20260404_203011_16712 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775305811473_trace_sem_baseline_seed42_qtsa_control_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775305811473_trace_sem_baseline_seed42_qtsa_control_v1.log |
| qtsa_readout_only_seed42_challenge_v1 | query-conditioned trace-semantic alignment/readout only; no transition rewrite | done | 2000 | 2000 | True | 20260404_203011_20366 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775305811663_qtsa_readout_only_seed42_challenge_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775305811663_qtsa_readout_only_seed42_challenge_v1.log |
| qtsa_readout_temporal_consistency_seed42_challenge_v1 | qtsa readout + temporal semantic consistency regularization | done | 2000 | 2000 | True | 20260404_203011_395 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775305811826_qtsa_readout_temporal_consistency_seed42_challenge_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775305811826_qtsa_readout_temporal_consistency_seed42_challenge_v1.log |

## Official Ranking (Within This 3-Run QTSA Matrix)

| rank | run | query_localization_error | query_top1_acc | future_trajectory_l1 | future_mask_iou | identity_consistency | identity_switch_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | trace_sem_baseline_seed42_qtsa_control_v1 | 0.001640 | 0.977099 | 0.001842 | na | na | na |
| 2 | qtsa_readout_only_seed42_challenge_v1 | 0.004681 | 0.951654 | 0.004789 | na | na | na |
| 3 | qtsa_readout_temporal_consistency_seed42_challenge_v1 | 0.005884 | 0.941476 | 0.006023 | na | na | na |

## Provisional Conclusion

- best_run_by_official_rule: trace_sem_baseline_seed42_qtsa_control_v1
- matrix_complete: True

