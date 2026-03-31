# STWM V4.2 Occlusion/Reconnect Bucket Analysis (Seed42)

Source:

- `reports/stwm_v4_2_occlusion_reconnect_seed42.json`

## Purpose

Evaluate reappearance/reconnect behavior only on difficult buckets, without changing main evaluator.

## Bucket Definition

Rows with `has_reappearance_event > 0` are considered event rows.

For event rows we summarize:

- reconnect success rate
- reconnect min error
- trajectory/query errors on events

## Result

| Run | total_rows | event_rows | event_row_ratio | reconnect_success_rate |
|---|---:|---:|---:|---:|
| full_v4_2 | 120 | 0 | 0.0000 | 0.0000 |
| wo_semantics_v4_2 | 120 | 0 | 0.0000 | 0.0000 |
| wo_identity_v4_2 | 120 | 0 | 0.0000 | 0.0000 |

No run has event rows in this seed42 mini-val slice.

## Interpretation

1. Current mini-val slice does not expose reappearance events under this logging rule.
2. Reconnect-specific conclusions cannot be made from this run.
3. The issue is event coverage, not necessarily architecture failure.

## Next-Step Requirement

Before claiming reconnect benefit, we need eventful clips in the evaluation bucket.

Recommended minimal action in next 220M round:

1. increase eventful-clip proportion in sampled slice (without expanding dataset scope)
2. keep posthoc bucket analysis unchanged for comparability
