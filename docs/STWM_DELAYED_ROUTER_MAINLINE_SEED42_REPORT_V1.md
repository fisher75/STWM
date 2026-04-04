# STWM Delayed Router Mainline Seed42 Report V1

Generated: 2026-04-04 16:10:15
Submit TSV: reports/stwm_delayed_router_mainline_submit_v1.tsv
Queue status dir: /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status

## Monitoring Summary

- all_entered_running: True
- all_train_log_growth_observed: True
- all_sidecar_generated: True
- all_terminal(done/failed): True

## Runs

| run | unique_change_point | state | max_step | train_log_rows | sidecar | job_id | status_file | main_log |
|---|---|---|---:|---:|---|---|---|---|
| delayed_only_seed42_challenge_v1 | Enable delay only: --object-bias-delay-steps 200 | done | 2000 | 2000 | True | 20260404_151517_29745 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775286917650_delayed_only_seed42_challenge_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775286917650_delayed_only_seed42_challenge_v1.log |
| two_path_residual_seed42_challenge_v1 | Residual two-path proxy: --object-bias-alpha 0.50 + gated(th=0.5) | done | 2000 | 2000 | True | 20260404_151517_6593 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775286917999_two_path_residual_seed42_challenge_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775286917999_two_path_residual_seed42_challenge_v1.log |
| delayed_residual_router_seed42_challenge_v1 | Combined: delay 200 + residual alpha 0.50 + gated(th=0.5) | done | 2000 | 2000 | True | 20260404_151518_26484 | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/status/1775286918437_delayed_residual_router_seed42_challenge_v1.status.json | /home/chen034/workspace/stwm/outputs/queue/stwm_protocol_v2_frontend_default_v1/d1_train/logs/1775286918437_delayed_residual_router_seed42_challenge_v1.log |

## Official Ranking (Within This 3-Run Matrix)

| rank | run | query_localization_error | query_top1_acc | future_trajectory_l1 | future_mask_iou | identity_consistency | identity_switch_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | two_path_residual_seed42_challenge_v1 | 0.005062 | 0.956743 | 0.005183 | na | na | na |
| 2 | delayed_residual_router_seed42_challenge_v1 | 0.005392 | 0.956743 | 0.005341 | na | na | na |
| 3 | delayed_only_seed42_challenge_v1 | 0.005995 | 0.933842 | 0.005896 | na | na | na |

## Provisional Conclusion

- best_run_by_official_rule: two_path_residual_seed42_challenge_v1
- matrix_complete: True

