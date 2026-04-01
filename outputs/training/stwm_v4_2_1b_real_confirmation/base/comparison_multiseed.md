# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `outputs/training/stwm_v4_2_1b_real_confirmation/base`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.259457 +- 0.000034 | 0.257318 +- 0.001087 | 2.364562 +- 0.012029 | 2.764496 +- 0.001212 | -0.002139 +- 0.001121 | 0.574007 +- 0.422897 | 0.028889 +- 0.000000 | 0.057778 +- 0.000000 |
| wo_semantics_v4_2 | 0.279507 +- 0.024517 | 0.272129 +- 0.024597 | 0.000000 +- 0.000000 | 2.764514 +- 0.000060 | -0.007377 +- 0.000080 | 0.897437 +- 0.044152 | 0.017778 +- 0.001111 | 0.057778 +- 0.000000 |
| wo_object_bias_v4_2 | 0.301646 +- 0.006413 | 0.307587 +- 0.006055 | 2.403811 +- 0.039850 | 2.763983 +- 0.000696 | 0.005942 +- 0.000358 | 0.887938 +- 0.107690 | 0.029444 +- 0.000556 | 0.057778 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | +0.020050 +- 0.024483 | +0.014811 +- 0.025684 | -2.364562 +- 0.012029 | +0.000018 +- 0.001272 | -0.005239 +- 0.001201 | +0.323430 +- 0.378746 | -0.011111 +- 0.001111 | +0.000000 +- 0.000000 |
| wo_object_bias_v4_2 | +0.042189 +- 0.006379 | +0.050269 +- 0.007142 | +0.039250 +- 0.027821 | -0.000514 +- 0.000516 | +0.008081 +- 0.000763 | +0.313931 +- 0.315207 | +0.000556 +- 0.000556 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | 1 | 1 | 0 | 1 | 0 | 1 | 2 | 0 |
| wo_object_bias_v4_2 | 2 | 2 | 2 | 1 | 2 | 1 | 0 | 0 |
