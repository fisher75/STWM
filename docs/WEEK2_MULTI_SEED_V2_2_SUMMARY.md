# Week2 V2.2 Multi-Seed Summary

Source summary: `reports/week2_minival_v2_2_multiseed_summary.json`

## Aggregate (mean +- std, n=3)

| Run | future_trajectory_l1 | query_localization_error | query_top1_acc | query_hit_rate | identity_consistency | identity_switch_rate | occlusion_recovery_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.044811 +- 0.006757 | 0.044947 +- 0.006835 | 0.462963 +- 0.130946 | 0.685185 +- 0.445215 | 0.076389 +- 0.005670 | 0.923611 +- 0.005670 | 0.000000 +- 0.000000 |
| wo_semantics | 0.048648 +- 0.004278 | 0.048649 +- 0.004147 | 0.574074 +- 0.026189 | 0.388889 +- 0.432716 | 0.078704 +- 0.003274 | 0.921296 +- 0.003274 | 0.000000 +- 0.000000 |
| wo_identity_memory | 0.053544 +- 0.015717 | 0.053455 +- 0.015663 | 0.592593 +- 0.069290 | 0.370370 +- 0.445215 | 0.050926 +- 0.021467 | 0.949074 +- 0.021467 | 0.000000 +- 0.000000 |

## Delta vs full (mean)

Sign convention: `delta = compare - full`.

- `wo_semantics`
  - trajectory `+0.003837`
  - query error `+0.003702`
  - query top1 `+0.111111`
  - query hit `-0.296296`
  - identity consistency `+0.002315`
  - identity switch `-0.002315`

- `wo_identity_memory`
  - trajectory `+0.008733`
  - query error `+0.008508`
  - query top1 `+0.129630`
  - query hit `-0.314815`
  - identity consistency `-0.025463`
  - identity switch `+0.025463`

## Per-Seed Stability Check

Trajectory / query error by seed:

- Seed 42
  - full `0.035330 / 0.035353`
  - wo_semantics `0.054658 / 0.054487` (full better)
  - wo_identity_memory `0.060368 / 0.060225` (full better)

- Seed 123
  - full `0.048519 / 0.048722`
  - wo_semantics `0.045041 / 0.045243` (wo_semantics better)
  - wo_identity_memory `0.068451 / 0.068336` (full better)

- Seed 456
  - full `0.050585 / 0.050766`
  - wo_semantics `0.046246 / 0.046217` (wo_semantics better)
  - wo_identity_memory `0.031813 / 0.031805` (wo_identity_memory better)

Stability conclusion:

- `full > wo_semantics` is not stable across all seeds.
- `full > wo_identity_memory` is not stable across all seeds.
- Aggregate means still favor full on trajectory and query localization error.

## Query Independence Check

`query_localization_error` is no longer identical to `future_trajectory_l1` in any run, so the V2 decoupling objective remains satisfied in V2.2.

## Failure-First Figure Pack (V2.2)

Manifest:

- `outputs/visualizations/week2_figures_v2_2/figure_manifest.json`

Panels:

- `full_fail_wo_identity_worse` (4 cases)
- `full_success_wo_semantics_fail` (4 cases)
- `query_hard_success_failure` (4 cases)

## Five-Question Decision Snapshot

1. Is `full > wo_semantics` stable across seeds?
   - No (seed-level reversals exist).
2. Is `full > wo_identity_memory` stable across seeds?
   - No (seed 456 reverses ordering).
3. Is identity memory supported by robust evidence?
   - Weakly supported in aggregate, not robust across seeds.
4. Should identity memory be downgraded as a main claim?
   - Yes, keep as tentative/secondary claim for now.
5. Should we scale to 1B now?
   - No, protocol evidence is still unstable.