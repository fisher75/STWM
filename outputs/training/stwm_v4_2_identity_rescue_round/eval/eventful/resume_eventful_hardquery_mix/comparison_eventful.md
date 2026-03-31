# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/eventful/resume_eventful_hardquery_mix`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.262043 +- 0.006478 | 0.279433 +- 0.019116 | 1.965076 +- 0.166476 | 1.940812 +- 0.000000 | 0.017391 +- 0.012638 | 0.695630 +- 0.304070 | 0.141667 +- 0.025000 | 0.566667 +- 0.000000 |
| wo_identity_v4_2 | 0.280018 +- 0.021497 | 0.284829 +- 0.026529 | 2.212258 +- 0.115266 | 0.000000 +- 0.000000 | 0.004811 +- 0.005032 | 0.000000 +- 0.000000 | 0.208333 +- 0.041667 | 0.566667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | +0.017975 +- 0.015019 | +0.005396 +- 0.007413 | +0.247182 +- 0.281741 | -1.940812 +- 0.000000 | -0.012579 +- 0.007606 | -0.695630 +- 0.304070 | +0.066667 +- 0.066667 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 2 | 1 | 1 | 0 | 0 | 2 | 0 | 0 |
