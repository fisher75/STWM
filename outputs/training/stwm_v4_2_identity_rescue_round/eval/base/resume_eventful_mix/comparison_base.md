# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/base/resume_eventful_mix`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.283579 +- 0.020237 | 0.264824 +- 0.023774 | 1.502617 +- 0.505944 | 1.940812 +- 0.000000 | -0.018755 +- 0.003537 | 0.925977 +- 0.070844 | 0.050000 +- 0.000000 | 0.216667 +- 0.000000 |
| wo_identity_v4_2 | 0.266784 +- 0.005076 | 0.261431 +- 0.005881 | 0.990702 +- 0.001755 | 0.000000 +- 0.000000 | -0.005352 +- 0.000805 | 0.000000 +- 0.000000 | 0.050000 +- 0.000000 | 0.216667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | -0.016795 +- 0.015162 | -0.003393 +- 0.017893 | -0.511916 +- 0.507699 | -1.940812 +- 0.000000 | +0.013403 +- 0.002732 | -0.925977 +- 0.070844 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 0 | 1 | 0 | 0 | 2 | 2 | 0 | 0 |
