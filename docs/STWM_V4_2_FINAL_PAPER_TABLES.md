# STWM V4.2 Final Paper Tables

## Scope

This file is the final main-text table draft set for paperization mode.

No new experiment direction is introduced.

Machine-readable companion:

- `reports/stwm_v4_2_final_paper_tables.json`

## Table 1. Mainline Results (Base Multi-Seed)

Source:

- `outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.json`
- `reports/week2_minival_v2_3_multiseed_summary.json` (legacy old/current baseline row)

| Run | Trajectory Metric | Trajectory (mean +- std) | Query Error (mean +- std) | Query-Traj Gap (mean +- std) | Seeds | Note |
|---|---|---:|---:|---:|---|---|
| full_v4_2 | trajectory_l1 | 0.050809 +- 0.003275 | 0.051175 +- 0.002910 | 0.000366 +- 0.001522 | 42/123/456 | mainline |
| wo_semantics_v4_2 | trajectory_l1 | 0.171720 +- 0.025330 | 0.172328 +- 0.025847 | 0.000608 +- 0.000902 | 42/123/456 | ablation |
| legacy_week2_v2_3_full_old_current_baseline | future_trajectory_l1 | 0.044811 +- 0.006757 | 0.044947 +- 0.006835 | 0.000135 (derived mean) | 42/123/456 | context baseline; not strict apples-to-apples |

Readout:

1. `full_v4_2` vs `wo_semantics_v4_2` is the strict base-protocol mainline comparison.
2. legacy baseline row is included for continuity/context only, not as strict same-pipeline claim support.

## Table 2. State-Identifiability Results

Source:

- `outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.json`

| Run | Trajectory L1 | Query Error | Query-Traj Gap | Reconnect Success | Reappearance Ratio |
|---|---:|---:|---:|---:|---:|
| full_v4_2 | 0.254923 +- 0.002212 | 0.271854 +- 0.020581 | 0.016931 +- 0.019562 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |
| wo_semantics_v4_2 | 0.254079 +- 0.000459 | 0.268620 +- 0.007898 | 0.014540 +- 0.008355 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |
| wo_object_bias_v4_2 | 0.254879 +- 0.002167 | 0.285993 +- 0.016796 | 0.031114 +- 0.016817 | 0.133333 +- 0.023570 | 0.550000 +- 0.000000 |

Delta vs full (query error):

- `wo_semantics_v4_2`: `-0.003235 +- 0.012969` (mixed)
- `wo_object_bias_v4_2`: `+0.014138 +- 0.035827` (full better on mean)

## Table 3. Harder-Protocol Decoupling

Source:

- `reports/stwm_v4_2_state_identifiability_decoupling_v1.json`

| Run | abs_corr | close_ratio | decoupling_score | proxy_like_count |
|---|---:|---:|---:|---:|
| full_v4_2 | 0.731209 +- 0.088596 | 0.000000 +- 0.000000 | 0.634395 +- 0.044298 | 0 |
| wo_semantics_v4_2 | 0.722384 +- 0.083634 | 0.016667 +- 0.023570 | 0.630475 +- 0.037095 | 0 |
| wo_object_bias_v4_2 | 0.848629 +- 0.010777 | 0.016667 +- 0.023570 | 0.567352 +- 0.015353 | 0 |

Delta vs full (run - full):

- `wo_semantics_v4_2`: d_corr `-0.008826`, d_close `+0.016667`, d_score `-0.003920`
- `wo_object_bias_v4_2`: d_corr `+0.117420`, d_close `+0.016667`, d_score `-0.067043`

Old-v1 reference signature (for boundary comparison):

- `abs_corr > 0.95`
- `close_ratio > 0.70`
- near exact-equality behavior
