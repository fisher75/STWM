# STWM V4.2 State-Identifiability Decoupling

## Scope

This report checks whether query behavior collapses to trajectory shortcut under the state-identifiability hard protocol.

Artifact:

- `reports/stwm_v4_2_state_identifiability_decoupling_v1.json`

Runs:

- `full_v4_2`
- `wo_semantics_v4_2`
- `wo_object_bias_v4_2`

Seeds:

- `42, 123, 456`

## Aggregate

| run | abs_corr | close_ratio | decoupling_score | proxy_like_count |
|---|---:|---:|---:|---:|
| full_v4_2 | 0.731209 +- 0.088596 | 0.000000 +- 0.000000 | 0.634395 +- 0.044298 | 0 |
| wo_semantics_v4_2 | 0.722384 +- 0.083634 | 0.016667 +- 0.023570 | 0.630475 +- 0.037095 | 0 |
| wo_object_bias_v4_2 | 0.848629 +- 0.010777 | 0.016667 +- 0.023570 | 0.567352 +- 0.015353 | 0 |

Notes:

1. All runs stay outside old-v1 proxy-collapse signature (`abs_corr > 0.95` and `close_ratio > 0.7`).
2. `full_v4_2` vs `wo_object_bias_v4_2` shows lower correlation and higher decoupling score for full.

## Delta vs full_v4_2 (run - full)

| run | d_corr_abs | d_close_ratio | d_decoupling_score |
|---|---:|---:|---:|
| wo_semantics_v4_2 | -0.008826 +- 0.032813 | +0.016667 +- 0.023570 | -0.003920 +- 0.027023 |
| wo_object_bias_v4_2 | +0.117420 +- 0.080543 | +0.016667 +- 0.023570 | -0.067043 +- 0.044227 |

Interpretation:

1. Against matched-budget representation control, full has a meaningful decoupling advantage.
2. Relative to `wo_semantics_v4_2`, decoupling difference is small and mixed.

## Judgement

- `full_stably_better_than_old_v1_proxy_state = true`
- `full_proxy_like_count = 0`
