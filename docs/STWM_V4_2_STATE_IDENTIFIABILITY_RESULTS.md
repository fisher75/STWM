# STWM V4.2 State-Identifiability Results

## Scope

This document summarizes second-contribution evaluation under the formal state-identifiability protocol.

Core artifacts:

- `outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.json`
- `outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.md`

Seeds:

- `42, 123, 456`

Runs:

- `full_v4_2`
- `wo_semantics_v4_2`
- `wo_object_bias_v4_2` (matched-budget representation control)

## Overall Aggregate (mean +- std)

| run | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate | reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| full_v4_2 | 0.254923 +- 0.002212 | 0.271854 +- 0.020581 | 0.016931 +- 0.019562 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |
| wo_semantics_v4_2 | 0.254079 +- 0.000459 | 0.268620 +- 0.007898 | 0.014540 +- 0.008355 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |
| wo_object_bias_v4_2 | 0.254879 +- 0.002167 | 0.285993 +- 0.016796 | 0.031114 +- 0.016817 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |

## Delta vs full_v4_2 (run - full)

| run | d_trajectory_l1 | d_query_localization_error | d_query_traj_gap | d_reconnect_success_rate | d_reappearance_event_ratio |
|---|---:|---:|---:|---:|---:|
| wo_semantics_v4_2 | -0.000844 +- 0.002367 | -0.003235 +- 0.012969 | -0.002391 +- 0.011308 | +0.000000 +- 0.040825 | +0.000000 +- 0.000000 |
| wo_object_bias_v4_2 | -0.000044 +- 0.000044 | +0.014138 +- 0.035827 | +0.014182 +- 0.035813 | +0.000000 +- 0.000000 | +0.000000 +- 0.000000 |

Interpretation:

1. Against matched-budget representation control, `full_v4_2` keeps a clear mean advantage on query localization and query-traj gap.
2. Against `wo_semantics_v4_2`, the overall mean gap is small and mixed, suggesting semantics gain in this eval-only sweep is not dominant.

## Per-Type Delta Highlights (run - full)

| query_type | run | d_query_localization_error | d_query_traj_gap | full_better_query_count (3 seeds) |
|---|---|---:|---:|---:|
| same_category_distractor | wo_semantics_v4_2 | -0.003235 +- 0.012969 | -0.002391 +- 0.011308 | 1 |
| same_category_distractor | wo_object_bias_v4_2 | +0.014138 +- 0.035827 | +0.014182 +- 0.035813 | 2 |
| relation_conditioned_query | wo_semantics_v4_2 | -0.002669 +- 0.013860 | -0.001760 +- 0.012551 | 1 |
| relation_conditioned_query | wo_object_bias_v4_2 | -0.001816 +- 0.027953 | -0.001795 +- 0.027931 | 1 |
| future_conditioned_reappearance_aware | wo_semantics_v4_2 | -0.002674 +- 0.019437 | -0.002120 +- 0.018202 | 1 |
| future_conditioned_reappearance_aware | wo_object_bias_v4_2 | +0.020507 +- 0.053635 | +0.020569 +- 0.053602 | 2 |

Interpretation:

1. `future_conditioned_reappearance_aware` is where `full_v4_2` shows the clearest positive mean edge over `wo_object_bias_v4_2`.
2. `relation_conditioned_query` remains the weakest and most mixed slice.

## Matched-Budget Representation Control Definition

`wo_object_bias_v4_2` keeps identical model preset and parameter budget, but neutralizes tokenizer object-bias inputs:

- `prior_features -> 0`
- `teacher_objectness -> 0.5`

No architecture or loss change is introduced.
