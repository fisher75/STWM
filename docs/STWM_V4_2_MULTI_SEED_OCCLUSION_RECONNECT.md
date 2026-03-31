# STWM V4.2 Multi-Seed Occlusion/Reconnect Analysis

Source:

- `reports/stwm_v4_2_occlusion_reconnect_multiseed.json`

## Purpose

Evaluate reconnect trend only on eventful rows (`has_reappearance_event > 0`) across seeds.

This is posthoc only and does not modify the main evaluator.

## Event Coverage

| Run | total_event_rows (3 seeds total) |
|---|---:|
| full_v4_2 | 0 |
| wo_semantics_v4_2 | 0 |
| wo_identity_v4_2 | 0 |

Paired eventful seeds for delta comparison vs full:

- full vs wo_semantics: `0`
- full vs wo_identity: `0`

## Statistical Power Check

Configured thresholds:

- `min_total_event_rows = 10`
- `min_paired_seeds = 2`

Observed output:

- `full_has_min_event_rows = False`
- `full_vs_wo_semantics_paired_seed_count = 0`
- `full_vs_wo_identity_paired_seed_count = 0`
- `sufficient_for_reconnect_claim = False`

## Interpretation

1. This round has zero reconnect-event support rows under current slice and logging rule.
2. Any reconnect trend interpretation would be over-claiming.
3. Required statement for this round: **统计力不足**.

## Claim Boundary

- acceptable claim: reconnect trend is unresolved due to event coverage.
- not acceptable claim: reconnect gain is established.
