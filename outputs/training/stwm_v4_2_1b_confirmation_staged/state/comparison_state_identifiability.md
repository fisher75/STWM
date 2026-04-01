# STWM V4.2 State-Identifiability Summary

Runs root: `outputs/training/stwm_v4_2_1b_confirmation_staged/state`
Seeds: `42, 123, 456`

## Overall Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| full_v4_2 | 0.494497 +- 0.030433 | 0.487315 +- 0.006262 | -0.007182 +- 0.027207 | 0.166667 +- 0.023570 | 0.550000 +- 0.000000 |
| wo_semantics_v4_2 | 0.527424 +- 0.008088 | 0.524835 +- 0.012672 | -0.002589 +- 0.004757 | 0.094444 +- 0.062854 | 0.550000 +- 0.000000 |
| wo_object_bias_v4_2 | 0.484196 +- 0.044852 | 0.512931 +- 0.033658 | 0.028735 +- 0.018458 | 0.200000 +- 0.023570 | 0.550000 +- 0.000000 |

## Overall Delta vs full_v4_2 (run - full)

| run | d_trajectory_l1 | d_query_localization_error | d_query_traj_gap | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | +0.032926 +- 0.022346 | +0.037520 +- 0.009522 | +0.004593 +- 0.031408 | -0.072222 +- 0.086424 | +0.000000 +- 0.000000 |
| wo_object_bias_v4_2 | -0.010301 +- 0.065574 | +0.025616 +- 0.039294 | +0.035918 +- 0.044340 | +0.033333 +- 0.023570 | +0.000000 +- 0.000000 |

## Per-Type Delta vs full_v4_2 (run - full)

| query_type | clip_count | run | d_traj | d_query | d_query_gap | d_reconnect_success | full_better_query_count |
|---|---:|---|---:|---:|---:|---:|---:|
| same_category_distractor | 18 | wo_semantics_v4_2 | +0.032926 +- 0.022346 | +0.037520 +- 0.009522 | +0.004593 +- 0.031408 | -0.072222 +- 0.086424 | 3 |
| same_category_distractor | 18 | wo_object_bias_v4_2 | -0.010301 +- 0.065574 | +0.025616 +- 0.039294 | +0.035918 +- 0.044340 | +0.033333 +- 0.023570 | 2 |
| spatial_disambiguation | 18 | wo_semantics_v4_2 | +0.032926 +- 0.022346 | +0.037520 +- 0.009522 | +0.004593 +- 0.031408 | -0.072222 +- 0.086424 | 3 |
| spatial_disambiguation | 18 | wo_object_bias_v4_2 | -0.010301 +- 0.065574 | +0.025616 +- 0.039294 | +0.035918 +- 0.044340 | +0.033333 +- 0.023570 | 2 |
| relation_conditioned_query | 17 | wo_semantics_v4_2 | +0.050800 +- 0.006891 | +0.052350 +- 0.019696 | +0.001549 +- 0.026581 | -0.077381 +- 0.092597 | 3 |
| relation_conditioned_query | 17 | wo_object_bias_v4_2 | -0.011305 +- 0.063649 | +0.025059 +- 0.035586 | +0.036363 +- 0.045967 | +0.035714 +- 0.025254 | 2 |
| future_conditioned_reappearance_aware | 11 | wo_semantics_v4_2 | +0.088464 +- 0.026924 | +0.087336 +- 0.072660 | -0.001127 +- 0.047027 | -0.166667 +- 0.173675 | 2 |
| future_conditioned_reappearance_aware | 11 | wo_object_bias_v4_2 | -0.014926 +- 0.060215 | +0.039662 +- 0.045328 | +0.054588 +- 0.071169 | +0.052632 +- 0.037216 | 3 |
