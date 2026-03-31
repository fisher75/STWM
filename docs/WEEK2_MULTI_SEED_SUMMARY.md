# Week2 V2.1 Multi-Seed Summary

Source summary: `reports/week2_minival_v2_1_multiseed_summary.json`

## Aggregate (mean +- std, n=3)

| Run | future_trajectory_l1 | query_localization_error | query_top1_acc | query_hit_rate | identity_consistency | identity_switch_rate | occlusion_recovery_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.044811 +- 0.006757 | 0.044947 +- 0.006835 | 0.777778 +- 0.045361 | 0.685185 +- 0.445215 | 0.083333 +- 0.015002 | 1.000000 +- 0.000000 | 0.000000 +- 0.000000 |
| wo_semantics | 0.048648 +- 0.004278 | 0.048649 +- 0.004147 | 0.777778 +- 0.090722 | 0.388889 +- 0.432716 | 0.076389 +- 0.000000 | 1.000000 +- 0.000000 | 0.000000 +- 0.000000 |
| wo_identity_memory | 0.053544 +- 0.015717 | 0.053455 +- 0.015663 | 0.703704 +- 0.052378 | 0.370370 +- 0.445215 | 0.055556 +- 0.024715 | 1.000000 +- 0.000000 | 0.000000 +- 0.000000 |

## Delta vs full (mean)

- `wo_semantics`:
  - trajectory `+0.003837`
  - query error `+0.003702`
  - query top1 `0.000000`
  - query hit `-0.296296`
- `wo_identity_memory`:
  - trajectory `+0.008733`
  - query error `+0.008508`
  - query top1 `-0.074074`
  - query hit `-0.314815`

## Per-Seed Stability Check

Trajectory / query error by seed:

- Seed 42:
  - full `0.035330 / 0.035353`
  - wo_semantics `0.054658 / 0.054487`
  - wo_identity_memory `0.060368 / 0.060225`
- Seed 123:
  - full `0.048519 / 0.048722`
  - wo_semantics `0.045041 / 0.045243` (better than full)
  - wo_identity_memory `0.068451 / 0.068336`
- Seed 456:
  - full `0.050585 / 0.050766`
  - wo_semantics `0.046246 / 0.046217` (better than full)
  - wo_identity_memory `0.031813 / 0.031805` (better than full)

Stability conclusion:

- `full > wo_semantics`: not stable across all seeds.
- `full > wo_identity_memory`: not stable across all seeds.
- Aggregate means favor full, but variance is too large for strong paper claims yet.

## Query Independence Status

In all seeds and all three runs, `query_localization_error` is no longer exactly equal to `future_trajectory_l1`.

This confirms V2.1 preserved the key decoupling fix from V2.

## Readiness Verdict

Current status is promising but not statistically stable enough for strong conclusions about identity/query superiority.
