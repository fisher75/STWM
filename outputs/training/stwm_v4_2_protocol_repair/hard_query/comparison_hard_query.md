# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_protocol_repair/hard_query`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.334775 +- 0.030581 | 0.352572 +- 0.040896 | 2.746668 +- 0.005940 | 2.359194 +- 0.000421 | 0.017797 +- 0.010315 | 0.966290 +- 0.001874 | 0.200000 +- 0.000000 | 0.566667 +- 0.000000 |
| wo_identity_v4_2 | 0.312500 +- 0.008280 | 0.315885 +- 0.005334 | 2.748963 +- 0.008952 | 0.000000 +- 0.000000 | 0.003385 +- 0.002946 | 0.000000 +- 0.000000 | 0.212500 +- 0.012500 | 0.566667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | -0.022275 +- 0.022301 | -0.036687 +- 0.035562 | +0.002294 +- 0.003012 | -2.359194 +- 0.000421 | -0.014412 +- 0.013261 | -0.966290 +- 0.001874 | +0.012500 +- 0.012500 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 1 | 0 | 1 | 0 | 0 | 2 | 0 | 0 |
