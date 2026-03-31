# STWM V4.2 State-Identifiability Summary

Runs root: `/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_state_identifiability`
Seeds: `42, 123, 456`

## Overall Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| full_v4_2 | 0.254923 +- 0.002212 | 0.271854 +- 0.020581 | 0.016931 +- 0.019562 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |
| wo_semantics_v4_2 | 0.254079 +- 0.000459 | 0.268620 +- 0.007898 | 0.014540 +- 0.008355 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |
| wo_object_bias_v4_2 | 0.254879 +- 0.002167 | 0.285993 +- 0.016796 | 0.031114 +- 0.016817 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |

## Overall Delta vs full_v4_2 (run - full)

| run | d_trajectory_l1 | d_query_localization_error | d_query_traj_gap | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | -0.000844 +- 0.002367 | -0.003235 +- 0.012969 | -0.002391 +- 0.011308 | +0.000000 +- 0.040825 | +0.000000 +- 0.000000 |
| wo_object_bias_v4_2 | -0.000044 +- 0.000044 | +0.014138 +- 0.035827 | +0.014182 +- 0.035813 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |

## Per-Type Delta vs full_v4_2 (run - full)

| query_type | clip_count | run | d_traj | d_query | d_query_gap | d_reconnect_success | full_better_query_count |
|---|---:|---|---:|---:|---:|---:|---:|
| same_category_distractor | 18 | wo_semantics_v4_2 | -0.000844 +- 0.002367 | -0.003235 +- 0.012969 | -0.002391 +- 0.011308 | +0.000000 +- 0.040825 | 1 |
| same_category_distractor | 18 | wo_object_bias_v4_2 | -0.000044 +- 0.000044 | +0.014138 +- 0.035827 | +0.014182 +- 0.035813 | +0.000000 +- 0.000000 | 2 |
| spatial_disambiguation | 18 | wo_semantics_v4_2 | -0.000844 +- 0.002367 | -0.003235 +- 0.012969 | -0.002391 +- 0.011308 | +0.000000 +- 0.040825 | 1 |
| spatial_disambiguation | 18 | wo_object_bias_v4_2 | -0.000044 +- 0.000044 | +0.014138 +- 0.035827 | +0.014182 +- 0.035813 | +0.000000 +- 0.000000 | 2 |
| relation_conditioned_query | 17 | wo_semantics_v4_2 | -0.000908 +- 0.003099 | -0.002669 +- 0.013860 | -0.001760 +- 0.012551 | +0.000000 +- 0.043741 | 1 |
| relation_conditioned_query | 17 | wo_object_bias_v4_2 | -0.000021 +- 0.000032 | -0.001816 +- 0.027953 | -0.001795 +- 0.027931 | +0.000000 +- 0.000000 | 1 |
| future_conditioned_reappearance_aware | 11 | wo_semantics_v4_2 | -0.000554 +- 0.004965 | -0.002674 +- 0.019437 | -0.002120 +- 0.018202 | +0.000000 +- 0.064460 | 1 |
| future_conditioned_reappearance_aware | 11 | wo_object_bias_v4_2 | -0.000062 +- 0.000070 | +0.020507 +- 0.053635 | +0.020569 +- 0.053602 | +0.000000 +- 0.000000 | 2 |
