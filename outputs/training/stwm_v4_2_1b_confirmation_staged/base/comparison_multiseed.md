# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `outputs/training/stwm_v4_2_1b_confirmation_staged/base`
Seeds: `42, 123, 456`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.406253 +- 0.012013 | 0.436565 +- 0.018064 | 1.135111 +- 0.069270 | 2.549300 +- 0.000694 | 0.030311 +- 0.008723 | 0.965851 +- 0.018351 | 0.102778 +- 0.019642 | 0.225000 +- 0.000000 |
| wo_semantics_v4_2 | 0.488988 +- 0.052532 | 0.488326 +- 0.018365 | 0.000000 +- 0.000000 | 2.553599 +- 0.001151 | -0.000662 +- 0.034243 | 0.641872 +- 0.143791 | 0.044444 +- 0.051069 | 0.225000 +- 0.000000 |
| wo_object_bias_v4_2 | 0.396091 +- 0.028969 | 0.429387 +- 0.036371 | 1.208837 +- 0.140048 | 2.550017 +- 0.001373 | 0.033295 +- 0.007485 | 0.410969 +- 0.408782 | 0.116667 +- 0.000000 | 0.225000 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | +0.082735 +- 0.040520 | +0.051761 +- 0.006211 | -1.135111 +- 0.069270 | +0.004299 +- 0.001292 | -0.030973 +- 0.039211 | -0.323979 +- 0.125481 | -0.058333 +- 0.070711 | +0.000000 +- 0.000000 |
| wo_object_bias_v4_2 | -0.010162 +- 0.036870 | -0.007178 +- 0.052714 | +0.073726 +- 0.076613 | +0.000718 +- 0.000715 | +0.002984 +- 0.016121 | -0.554882 +- 0.393495 | +0.013889 +- 0.019642 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | 3 | 3 | 0 | 3 | 1 | 3 | 2 | 0 |
| wo_object_bias_v4_2 | 1 | 2 | 3 | 2 | 2 | 3 | 0 | 0 |
