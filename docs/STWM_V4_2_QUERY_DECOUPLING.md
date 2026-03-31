# STWM V4.2 Query-Trajectory Decoupling (Seed42)

Source:

- `reports/stwm_v4_2_query_decoupling_seed42.json`

## Purpose

Check whether query-localization is still effectively a trajectory proxy.

This is posthoc only and does not modify main evaluator.

## Metrics

- `pearson_corr(query_localization_error, trajectory_l1)`
- `close_ratio`: fraction where `|query_error - trajectory_l1| <= 0.002`
- `mean_abs_gap`
- `decoupling_score` (higher is less proxy-like)

## Results

| Run | corr | close_ratio | mean_abs_gap | decoupling_score | proxy_like |
|---|---:|---:|---:|---:|---|
| full_v4_2 | 0.9554 | 0.0667 | 0.01127 | 0.4890 | False |
| wo_semantics_v4_2 | 0.9949 | 0.1583 | 0.00997 | 0.4234 | False |
| wo_identity_v4_2 | 0.9801 | 0.0917 | 0.01171 | 0.4641 | False |

Comparison vs full:

- `wo_semantics_v4_2` has higher correlation and close-ratio, lower decoupling score.
- `wo_identity_v4_2` also has slightly higher correlation and close-ratio, lower decoupling score.

## Interpretation

1. Query is not an exact copy of trajectory (close ratio is low in all runs).
2. Coupling is still high (corr remains high), so decoupling is partial, not complete.
3. `full_v4_2` is relatively less proxy-like than both ablations in this seed42 run.

## Claim Boundary

- acceptable claim: V4.2 shows movement toward query-trajectory decoupling.
- not acceptable claim yet: query is fully independent from trajectory.
