# STWM V4.2 Multi-Seed Query-Trajectory Decoupling

Source:

- `reports/stwm_v4_2_query_decoupling_multiseed.json`

## Purpose

Check whether query-localization remains less proxy-like than old-v1 coupling state under multi-seed validation.

This is posthoc only and does not modify the main evaluator.

## Metrics

- `pearson_corr(query_localization_error, trajectory_l1)`
- `close_ratio`: fraction where `|query_error - trajectory_l1| <= 0.002`
- `decoupling_score` (higher is less proxy-like)
- `proxy_like`: `abs(corr) > 0.95` and `close_ratio > 0.70`
- `exact_equal_ratio`: strict equality ratio (`|gap| <= 1e-12`)

## Full_v4_2 Per-Seed

| Seed | corr | close_ratio | decoupling_score | proxy_like |
|---|---:|---:|---:|---|
| 42 | 0.9554 | 0.0667 | 0.4890 | False |
| 123 | 0.9209 | 0.0750 | 0.5021 | False |
| 456 | 0.9548 | 0.1167 | 0.4643 | False |

Aggregate (`full_v4_2`):

- corr mean/std: `0.9437 +- 0.0161`
- close_ratio mean/std: `0.0861 +- 0.0219`
- decoupling_score mean/std: `0.4851 +- 0.0157`
- exact_equal_ratio mean/std: `0.0000 +- 0.0000`
- proxy_like_count: `0/3`

## Cross-Run Aggregate Snapshot

| Run | corr mean+-std | close_ratio mean+-std | decoupling_score mean+-std |
|---|---:|---:|---:|
| full_v4_2 | 0.9437 +- 0.0161 | 0.0861 +- 0.0219 | 0.4851 +- 0.0157 |
| wo_semantics_v4_2 | 0.9958 +- 0.0009 | 0.1306 +- 0.0208 | 0.4368 +- 0.0100 |
| wo_identity_v4_2 | 0.9125 +- 0.0654 | 0.0889 +- 0.0443 | 0.4993 +- 0.0509 |

Comparison vs full (`ablation - full`):

- `wo_semantics_v4_2`
  - `delta_corr_abs`: `+0.0522 +- 0.0169`
  - `delta_close_ratio`: `+0.0444 +- 0.0349`
  - `delta_decoupling_score`: `-0.0483 +- 0.0173`
- `wo_identity_v4_2`
  - `delta_corr_abs`: `-0.0312 +- 0.0501`
  - `delta_close_ratio`: `+0.0028 +- 0.0314`
  - `delta_decoupling_score`: `+0.0142 +- 0.0400`

## Old-v1 Reference And Judgement

Reference used in report:

- old-v1 proxy-like signature:
  - `abs_corr > 0.95`
  - `close_ratio > 0.70`
  - near-equality behavior

Judgement from analysis output:

- `full_stably_better_than_old_v1_proxy_state = True`
- `full_proxy_like_count = 0`

## Interpretation

1. query decoupling is **not** a seed42-only false positive.
2. full_v4_2 is stably outside old-v1 proxy-collapse signature across 3 seeds.
3. coupling is still high in correlation space (around `0.94`), so claim boundary remains:
   - valid: better than old-v1 coupling state
   - not yet valid: fully independent query branch
