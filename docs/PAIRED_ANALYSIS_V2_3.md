# Paired Analysis V2.3

Source:

- `reports/week2_minival_v2_3_paired_analysis.json`

## Setup

- Baseline: `full`
- Compare runs: `wo_semantics`, `wo_identity_memory`
- Seeds: `42, 123, 456`
- Paired clip samples: `54`
- Delta sign: `compare - full`

## Per-Seed Delta Snapshot

### `wo_semantics - full`

- seed 42:
  - trajectory `+0.019328`
  - query error `+0.019134`
  - query top1 `+0.111111`
  - query hit `-0.888889`
- seed 123:
  - trajectory `-0.003478`
  - query error `-0.003478`
  - query top1 `+0.055556`
  - query hit `+0.000000`
- seed 456:
  - trajectory `-0.004339`
  - query error `-0.004549`
  - query top1 `+0.000000`
  - query hit `+0.000000`

### `wo_identity_memory - full`

- seed 42:
  - trajectory `+0.025038`
  - query error `+0.024872`
  - query top1 `-0.055556`
  - query hit `-0.944444`
- seed 123:
  - trajectory `+0.019932`
  - query error `+0.019614`
  - query top1 `+0.055556`
  - query hit `-0.944444`
- seed 456:
  - trajectory `-0.018772`
  - query error `-0.018961`
  - query top1 `+0.055556`
  - query hit `+0.944444`

Seed-level reversals persist.

## Clip-Level Paired Stats + Bootstrap CI

### Compare: `wo_semantics`

| Metric | Mean Delta | 95% CI | Interpretation |
|---|---:|---:|---|
| future_trajectory_l1 | +0.003837 | [0.000912, 0.006938] | full better |
| query_localization_error | +0.003702 | [0.000819, 0.006769] | full better |
| query_top1_acc | +0.055556 | [-0.037037, 0.148148] | inconclusive |
| query_hit_rate | -0.296296 | [-0.425926, -0.185185] | full better |
| identity_consistency | +0.004630 | [-0.050926, 0.060185] | inconclusive |
| identity_switch_rate | -0.004630 | [-0.060185, 0.050926] | inconclusive |
| occlusion_recovery_acc | +0.000000 | [0.000000, 0.000000] | non-informative |

### Compare: `wo_identity_memory`

| Metric | Mean Delta | 95% CI | Interpretation |
|---|---:|---:|---|
| future_trajectory_l1 | +0.008733 | [0.003442, 0.013896] | full better |
| query_localization_error | +0.008508 | [0.003157, 0.013660] | full better |
| query_top1_acc | +0.018519 | [-0.055556, 0.092593] | inconclusive |
| query_hit_rate | -0.314815 | [-0.555556, -0.074074] | full better |
| identity_consistency | -0.023148 | [-0.090278, 0.043981] | inconclusive (full tends better) |
| identity_switch_rate | +0.023148 | [-0.043981, 0.090278] | inconclusive (full tends better) |
| occlusion_recovery_acc | +0.000000 | [0.000000, 0.000000] | non-informative |

## Final Decision from Paired Evidence

1. Semantics signal remains directionally positive for full on trajectory and localization error.
2. Identity signal remains directional but statistically weak (CI crosses zero).
3. Occlusion metric remains unusable (all-zero).

Therefore:

- do not escalate to 1B
- stop evaluator churn after V2.3
- proceed with narrowed paper story (semantic trajectory primary; identity exploratory)