# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/base/control_resume_base`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.250779 +- 0.000502 | 0.168646 +- 0.012076 | 0.932556 +- 0.000834 | 1.940812 +- 0.000000 | -0.082132 +- 0.011574 | 0.980321 +- 0.015606 | 0.100000 +- 0.000000 | 0.216667 +- 0.000000 |
| wo_identity_v4_2 | 0.251607 +- 0.005009 | 0.174889 +- 0.006515 | 0.932039 +- 0.000041 | 0.000000 +- 0.000000 | -0.076718 +- 0.001507 | 0.000000 +- 0.000000 | 0.100000 +- 0.000000 | 0.216667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | +0.000829 +- 0.004507 | +0.006243 +- 0.005560 | -0.000516 +- 0.000875 | -1.940812 +- 0.000000 | +0.005414 +- 0.010067 | -0.980321 +- 0.015606 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 1 | 2 | 1 | 0 | 1 | 2 | 0 | 0 |
