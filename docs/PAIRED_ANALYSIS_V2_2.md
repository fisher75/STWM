# Paired Analysis V2.2

Source: `reports/week2_minival_v2_2_paired_analysis.json`

## Setup

- Baseline: `full`
- Compare runs: `wo_semantics`, `wo_identity_memory`
- Seeds: `42, 123, 456`
- Paired clip samples: `54`
- Delta sign: `compare - full`

Positive delta means compare run is larger than full.

## Per-Seed Delta Snapshot

### `wo_semantics - full`

- seed 42: trajectory `+0.019328`, query error `+0.019134`, query top1 `+0.333333`, query hit `-0.888889`
- seed 123: trajectory `-0.003478`, query error `-0.003478`, query top1 `+0.000000`, query hit `+0.000000`
- seed 456: trajectory `-0.004339`, query error `-0.004549`, query top1 `+0.000000`, query hit `+0.000000`

### `wo_identity_memory - full`

- seed 42: trajectory `+0.025038`, query error `+0.024872`, query top1 `+0.333333`, query hit `-0.944444`
- seed 123: trajectory `+0.019932`, query error `+0.019614`, query top1 `-0.055556`, query hit `-0.944444`
- seed 456: trajectory `-0.018772`, query error `-0.018961`, query top1 `+0.111111`, query hit `+0.944444`

Seed-level reversals are present, especially on seed 456.

## Clip-Level Paired Stats with Bootstrap CI

### Compare: `wo_semantics`

| Metric | Mean Delta | 95% Bootstrap CI | Interpretation |
|---|---:|---:|---|
| future_trajectory_l1 | +0.003837 | [0.000912, 0.006938] | full better |
| query_localization_error | +0.003702 | [0.000819, 0.006769] | full better |
| query_top1_acc | +0.111111 | [0.000000, 0.222222] | compare tends better |
| query_hit_rate | -0.296296 | [-0.425926, -0.185185] | full better |
| identity_consistency | +0.002315 | [-0.053241, 0.057870] | inconclusive |
| identity_switch_rate | -0.002315 | [-0.057870, 0.053241] | inconclusive |
| occlusion_recovery_acc | +0.000000 | [0.000000, 0.000000] | non-informative |

### Compare: `wo_identity_memory`

| Metric | Mean Delta | 95% Bootstrap CI | Interpretation |
|---|---:|---:|---|
| future_trajectory_l1 | +0.008733 | [0.003442, 0.013896] | full better |
| query_localization_error | +0.008508 | [0.003157, 0.013660] | full better |
| query_top1_acc | +0.129630 | [-0.018519, 0.277778] | inconclusive (compare tends better) |
| query_hit_rate | -0.314815 | [-0.555556, -0.074074] | full better |
| identity_consistency | -0.025463 | [-0.092593, 0.041667] | inconclusive (full tends better) |
| identity_switch_rate | +0.025463 | [-0.041667, 0.092593] | inconclusive (full tends better) |
| occlusion_recovery_acc | +0.000000 | [0.000000, 0.000000] | non-informative |

## Key Takeaways

1. On paired clip analysis, full has consistent advantage on trajectory and query localization error against both ablations.
2. Query metrics split by definition: top1 tends to favor ablations, while radius-hit favors full.
3. Identity deltas have the expected sign versus `wo_identity_memory`, but confidence intervals cross zero.
4. Occlusion recovery remains all-zero and currently unusable for evidence.

## Decision Use

V2.2 paired analysis supports:

- keeping full as default baseline at 220M
- delaying strong identity-memory claims
- delaying 1B scale-up until identity/occlusion probes become statistically informative