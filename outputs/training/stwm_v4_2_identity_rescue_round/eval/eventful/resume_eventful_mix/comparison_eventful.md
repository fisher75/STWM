# STWM V4.2 Mini-Val Multi-Seed Comparison

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_identity_rescue_round/eval/eventful/resume_eventful_mix`
Seeds: `42, 123`

## Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_v4_2 | 0.284487 +- 0.014256 | 0.294738 +- 0.008406 | 3.150810 +- 1.355682 | 1.940812 +- 0.000000 | 0.010250 +- 0.022662 | 0.726045 +- 0.270776 | 0.208333 +- 0.025000 | 0.566667 +- 0.000000 |
| wo_identity_v4_2 | 0.271899 +- 0.001675 | 0.309691 +- 0.007033 | 1.815396 +- 0.003922 | 0.000000 +- 0.000000 | 0.037792 +- 0.008707 | 0.000000 +- 0.000000 | 0.233333 +- 0.000000 | 0.566667 +- 0.000000 |

## Delta vs full_v4_2 (mean +- std)

| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | -0.012588 +- 0.012581 | +0.014953 +- 0.001373 | -1.335414 +- 1.351759 | -1.940812 +- 0.000000 | +0.027542 +- 0.013955 | -0.726045 +- 0.270776 | +0.025000 +- 0.025000 | +0.000000 +- 0.000000 |

## Full Better Count Across Seeds

Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.

| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wo_identity_v4_2 | 0 | 2 | 1 | 0 | 2 | 2 | 0 | 0 |
