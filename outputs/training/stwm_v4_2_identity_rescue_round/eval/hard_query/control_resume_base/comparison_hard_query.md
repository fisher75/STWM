# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/hard_query/control_resume_base`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.262726 +- 0.000496 | 0.263395 +- 0.013975 | 4.772167 +- 0.003160 | 1.940812 +- 0.000000 | 0.000669 +- 0.014471 | 0.980325 +- 0.015602 | 0.100000 +- 0.000000 | 0.550000 +- 0.000000 |
| wo_identity_v4_2 | 0.266336 +- 0.000936 | 0.271641 +- 0.003201 | 4.839938 +- 0.022030 | 0.000000 +- 0.000000 | 0.005306 +- 0.004137 | 0.000000 +- 0.000000 | 0.158333 +- 0.008333 | 0.550000 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | +0.003610 +- 0.000440 | +0.008246 +- 0.010774 | +0.067771 +- 0.018870 | -1.940812 +- 0.000000 | +0.004636 +- 0.010334 | -0.980325 +- 0.015602 | +0.058333 +- 0.008333 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 2 | 1 | 2 | 0 | 1 | 2 | 0 | 0 |
