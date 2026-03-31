# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/hard_query/resume_eventful_mix`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.285917 +- 0.007566 | 0.303250 +- 0.014265 | 4.362680 +- 0.096813 | 1.940812 +- 0.000000 | 0.017332 +- 0.021831 | 0.758034 +- 0.238787 | 0.191667 +- 0.025000 | 0.550000 +- 0.000000 |
| wo_identity_v4_2 | 0.280083 +- 0.001733 | 0.312764 +- 0.004491 | 4.469336 +- 0.003055 | 0.000000 +- 0.000000 | 0.032682 +- 0.006224 | 0.000000 +- 0.000000 | 0.216667 +- 0.000000 | 0.550000 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | -0.005835 +- 0.005834 | +0.009515 +- 0.009773 | +0.106656 +- 0.099868 | -1.940812 +- 0.000000 | +0.015349 +- 0.015607 | -0.758034 +- 0.238787 | +0.025000 +- 0.025000 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 0 | 1 | 2 | 0 | 1 | 2 | 0 | 0 |
