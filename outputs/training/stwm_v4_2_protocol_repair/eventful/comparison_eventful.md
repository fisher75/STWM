# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_protocol_repair/eventful`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.311272 +- 0.004461 | 0.323548 +- 0.011019 | 1.771284 +- 0.002332 | 2.360175 +- 0.000928 | 0.012276 +- 0.006557 | 0.917320 +- 0.026158 | 0.200000 +- 0.000000 | 0.566667 +- 0.000000 |
| wo_identity_v4_2 | 0.308645 +- 0.001392 | 0.317200 +- 0.003025 | 1.757693 +- 0.000685 | 0.000000 +- 0.000000 | 0.008555 +- 0.001634 | 0.000000 +- 0.000000 | 0.183333 +- 0.000000 | 0.566667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | -0.002626 +- 0.003070 | -0.006348 +- 0.007993 | -0.013592 +- 0.003017 | -2.360175 +- 0.000928 | -0.003721 +- 0.004924 | -0.917320 +- 0.026158 | -0.016667 +- 0.000000 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 1 | 1 | 0 | 0 | 1 | 2 | 2 | 0 |
