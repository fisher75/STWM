# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed`
Seeds: `42, 123, 456`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.050809 +- 0.003275 | 0.051175 +- 0.002910 | 1.038251 +- 0.002731 | 2.358817 +- 0.000377 | 0.000366 +- 0.001522 | 0.807672 +- 0.168442 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 |
| wo_semantics_v4_2 | 0.171720 +- 0.025330 | 0.172328 +- 0.025847 | 0.000000 +- 0.000000 | 2.367549 +- 0.002794 | 0.000608 +- 0.000902 | 0.481930 +- 0.259599 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 |
| wo_identity_v4_2 | 0.052933 +- 0.016863 | 0.053115 +- 0.016916 | 1.055313 +- 0.028668 | 0.000000 +- 0.000000 | 0.000182 +- 0.000412 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 | 0.000000 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | +0.120911 +- 0.026661 | +0.121153 +- 0.025737 | -1.038251 +- 0.002731 | +0.008732 +- 0.002792 | +0.000242 +- 0.001156 | -0.325742 +- 0.395056 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |
| wo_identity_v4_2 | +0.002124 +- 0.013697 | +0.001940 +- 0.014791 | +0.017062 +- 0.030958 | -2.358817 +- 0.000377 | -0.000184 +- 0.001279 | -0.807672 +- 0.168442 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | 3 | 3 | 0 | 3 | 2 | 2 | 0 | 0 |
| wo_identity_v4_2 | 1 | 1 | 1 | 0 | 1 | 3 | 0 | 0 |
