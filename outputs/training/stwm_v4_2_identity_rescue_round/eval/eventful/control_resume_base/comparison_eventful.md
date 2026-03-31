# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/eventful/control_resume_base`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.259914 +- 0.000519 | 0.256590 +- 0.002349 | 2.686739 +- 0.002166 | 1.940812 +- 0.000000 | -0.003324 +- 0.002867 | 0.980325 +- 0.015602 | 0.116667 +- 0.000000 | 0.566667 +- 0.000000 |
| wo_identity_v4_2 | 0.264014 +- 0.000647 | 0.264401 +- 0.007486 | 2.711468 +- 0.008686 | 0.000000 +- 0.000000 | 0.000387 +- 0.006839 | 0.000000 +- 0.000000 | 0.175000 +- 0.008333 | 0.566667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | +0.004101 +- 0.000129 | +0.007811 +- 0.009835 | +0.024729 +- 0.010852 | -1.940812 +- 0.000000 | +0.003711 +- 0.009707 | -0.980325 +- 0.015602 | +0.058333 +- 0.008333 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 2 | 1 | 2 | 0 | 1 | 2 | 0 | 0 |
