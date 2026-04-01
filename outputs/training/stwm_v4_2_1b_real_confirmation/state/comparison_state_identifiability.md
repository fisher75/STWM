# STWM V4.2 State-Identifiability Summary

Runs root: `outputs/training/stwm_v4_2_1b_real_confirmation/state`
Seeds: `42, 123`

## Overall Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| full_v4_2 | 0.255702 +- 0.001181 | 0.266480 +- 0.015289 | 0.010778 +- 0.014108 | 0.162500 +- 0.000000 | 0.554167 +- 0.000000 |
| wo_semantics_v4_2 | 0.253749 +- 0.000225 | 0.262949 +- 0.003830 | 0.009200 +- 0.003605 | 0.108333 +- 0.000000 | 0.554167 +- 0.000000 |
| wo_object_bias_v4_2 | 0.300650 +- 0.045565 | 0.312395 +- 0.064109 | 0.011745 +- 0.018544 | 0.164583 +- 0.002083 | 0.554167 +- 0.000000 |

## Overall Delta vs full_v4_2 (run - full)

| run | d_trajectory_l1 | d_query_localization_error | d_query_traj_gap | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | -0.001954 +- 0.000956 | -0.003531 +- 0.011459 | -0.001577 +- 0.010503 | -0.054167 +- 0.000000 | +0.000000 +- 0.000000 |
| wo_object_bias_v4_2 | +0.044948 +- 0.044384 | +0.045915 +- 0.048820 | +0.000967 +- 0.004436 | +0.002083 +- 0.002083 | +0.000000 +- 0.000000 |

## Per-Type Delta vs full_v4_2 (run - full)

| query_type | clip_count | run | d_traj | d_query | d_query_gap | d_reconnect_success | full_better_query_count |
|---|---:|---|---:|---:|---:|---:|---:|
| same_category_distractor | 18 | wo_semantics_v4_2 | -0.001954 +- 0.000956 | -0.003531 +- 0.011459 | -0.001577 +- 0.010503 | -0.054167 +- 0.000000 | 1 |
| same_category_distractor | 18 | wo_object_bias_v4_2 | +0.044948 +- 0.044384 | +0.045915 +- 0.048820 | +0.000967 +- 0.004436 | +0.002083 +- 0.002083 | 1 |
| spatial_disambiguation | 18 | wo_semantics_v4_2 | -0.001954 +- 0.000956 | -0.003531 +- 0.011459 | -0.001577 +- 0.010503 | -0.054167 +- 0.000000 | 1 |
| spatial_disambiguation | 18 | wo_object_bias_v4_2 | +0.044948 +- 0.044384 | +0.045915 +- 0.048820 | +0.000967 +- 0.004436 | +0.002083 +- 0.002083 | 1 |
| relation_conditioned_query | 17 | wo_semantics_v4_2 | -0.002710 +- 0.001263 | -0.004164 +- 0.012130 | -0.001454 +- 0.010867 | -0.057522 +- 0.000000 | 1 |
| relation_conditioned_query | 17 | wo_object_bias_v4_2 | +0.039808 +- 0.039399 | +0.033083 +- 0.050516 | -0.006725 +- 0.011117 | +0.002212 +- 0.002212 | 1 |
| future_conditioned_reappearance_aware | 11 | wo_semantics_v4_2 | -0.003743 +- 0.001841 | -0.007031 +- 0.018245 | -0.003288 +- 0.016404 | -0.087838 +- 0.000000 | 1 |
| future_conditioned_reappearance_aware | 11 | wo_object_bias_v4_2 | +0.021456 +- 0.020816 | +0.028433 +- 0.024114 | +0.006976 +- 0.003298 | +0.047297 +- 0.047297 | 2 |
