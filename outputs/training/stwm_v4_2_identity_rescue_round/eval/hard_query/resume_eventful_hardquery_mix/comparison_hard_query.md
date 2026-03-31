# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/hard_query/resume_eventful_hardquery_mix`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.264002 +- 0.002430 | 0.290215 +- 0.012773 | 2.766986 +- 0.018052 | 1.940812 +- 0.000000 | 0.026212 +- 0.010343 | 0.646589 +- 0.353110 | 0.150000 +- 0.000000 | 0.550000 +- 0.000000 |
| wo_identity_v4_2 | 0.286927 +- 0.023367 | 0.281440 +- 0.035757 | 3.056528 +- 0.303842 | 0.000000 +- 0.000000 | -0.005486 +- 0.012390 | 0.000000 +- 0.000000 | 0.191667 +- 0.041667 | 0.550000 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | +0.022925 +- 0.020937 | -0.008774 +- 0.022984 | +0.289542 +- 0.285790 | -1.940812 +- 0.000000 | -0.031699 +- 0.002047 | -0.646589 +- 0.353110 | +0.041667 +- 0.041667 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 2 | 1 | 2 | 0 | 0 | 2 | 0 | 0 |
