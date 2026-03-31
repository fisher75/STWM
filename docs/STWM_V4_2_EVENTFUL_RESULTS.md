# STWM V4.2 Eventful Protocol Results

## Run Setup

Protocol repair round (minimal re-validation):

- protocols:
  - eventful: `manifests/minisplits/stwm_v4_2_eventful_minival_v1.json`
  - hard_query: `manifests/minisplits/stwm_v4_2_hard_query_minival_v1.json`
- runs:
  - `full_v4_2`
  - `wo_identity_v4_2`
- seeds:
  - `42`, `123`
- steps: `120`
- sample_limit: `18`

Training outputs root:

- `outputs/training/stwm_v4_2_protocol_repair/`

## Eventful Summary (mean +- std)

Source:

- `outputs/training/stwm_v4_2_protocol_repair/eventful/comparison_eventful.json`

| Run | trajectory_l1 | query_localization_error | reappearance_event_ratio | reconnect_success_rate |
|---|---:|---:|---:|---:|
| full_v4_2 | 0.311272 +- 0.004461 | 0.323548 +- 0.011019 | 0.566667 +- 0.000000 | 0.200000 +- 0.000000 |
| wo_identity_v4_2 | 0.308645 +- 0.001392 | 0.317200 +- 0.003025 | 0.566667 +- 0.000000 | 0.183333 +- 0.000000 |

Delta (`wo_identity - full`):

- trajectory_l1: `-0.002626 +- 0.003070`
- query_localization_error: `-0.006348 +- 0.007993`
- reconnect_success_rate: `-0.016667 +- 0.000000`

Per-seed trajectory/query deltas:

- seed42:
  - d_traj: `+0.000443`
  - d_query: `+0.001646`
- seed123:
  - d_traj: `-0.005696`
  - d_query: `-0.014341`

Interpretation:

1. Event rows are non-zero and substantial in this protocol (`reappearance_event_ratio ~ 0.5667`).
2. `full_v4_2` shows consistent reconnect-success advantage over `wo_identity_v4_2` (same sign in both seeds).
3. Trajectory/query superiority of `full_v4_2` vs `wo_identity_v4_2` is not stable in this round.

## Eventful Bucket Analysis

Source:

- `reports/stwm_v4_2_eventful_occlusion_reconnect_v1.json`

Key stats:

- total event rows (full): `136`
- total event rows (wo_identity): `136`
- paired seed count for full vs wo_identity: `2`
- sufficient_for_reconnect_claim: `True` (for the executed comparison set)

Eventful delta (`wo_identity - full`):

- reconnect_success_rate: `-0.029412 +- 0.000000`
- reconnect_min_error_mean: `+0.001177 +- 0.000675`

This indicates a small but consistent reconnect advantage for `full_v4_2` on eventful buckets.

## Hard Query Summary (mean +- std)

Source:

- `outputs/training/stwm_v4_2_protocol_repair/hard_query/comparison_hard_query.json`

| Run | trajectory_l1 | query_localization_error | reappearance_event_ratio | reconnect_success_rate |
|---|---:|---:|---:|---:|
| full_v4_2 | 0.334775 +- 0.030581 | 0.352572 +- 0.040896 | 0.566667 +- 0.000000 | 0.200000 +- 0.000000 |
| wo_identity_v4_2 | 0.312500 +- 0.008280 | 0.315885 +- 0.005334 | 0.566667 +- 0.000000 | 0.212500 +- 0.012500 |

In hard-query protocol, `full_v4_2` is not stably better than `wo_identity_v4_2` on trajectory/query.

## Hard Query Decoupling Check

Source:

- `reports/stwm_v4_2_hard_query_decoupling_v1.json`
- baseline reference (same seeds 42/123): `reports/stwm_v4_2_baseline_decoupling_seed42_123.json`

`full_v4_2` decoupling aggregate:

- baseline corr: `0.9381`
- hard-query corr: `0.7004`
- baseline decoupling_score: `0.4955`
- hard-query decoupling_score: `0.6352`

Interpretation:

- harder query protocol reduces query-trajectory coupling for `full_v4_2` versus baseline on the same seeds.
- full remains non-proxy-like (`proxy_like_count=0`).
