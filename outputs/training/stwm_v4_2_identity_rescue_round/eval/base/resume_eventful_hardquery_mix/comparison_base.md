# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/base/resume_eventful_hardquery_mix`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.264580 +- 0.005567 | 0.238591 +- 0.004513 | 1.679482 +- 0.066362 | 1.940812 +- 0.000000 | -0.025990 +- 0.010080 | 0.924470 +- 0.075230 | 0.100000 +- 0.000000 | 0.216667 +- 0.000000 |
| wo_identity_v4_2 | 0.262335 +- 0.005692 | 0.246105 +- 0.015542 | 1.795711 +- 0.167583 | 0.000000 +- 0.000000 | -0.016230 +- 0.021234 | 0.000000 +- 0.000000 | 0.050000 +- 0.050000 | 0.216667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | -0.002245 +- 0.000125 | +0.007515 +- 0.011029 | +0.116229 +- 0.101221 | -1.940812 +- 0.000000 | +0.009760 +- 0.011154 | -0.924470 +- 0.075230 | -0.050000 +- 0.050000 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 0 | 1 | 2 | 0 | 1 | 2 | 1 | 0 |
