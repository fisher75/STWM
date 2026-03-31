# Week2 V2.3 Multi-Seed Summary

Source summary:

- `reports/week2_minival_v2_3_multiseed_summary.json`

## Aggregate (mean +- std, n=3)

| Run | future_trajectory_l1 | query_localization_error | query_top1_acc | query_hit_rate | identity_consistency | identity_switch_rate | occlusion_recovery_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.044811 +- 0.006757 | 0.044947 +- 0.006835 | 0.055556 +- 0.045361 | 0.685185 +- 0.445215 | 0.076389 +- 0.009821 | 0.923611 +- 0.009821 | 0.000000 +- 0.000000 |
| wo_semantics | 0.048648 +- 0.004278 | 0.048649 +- 0.004147 | 0.111111 +- 0.078567 | 0.388889 +- 0.432716 | 0.081019 +- 0.003274 | 0.918981 +- 0.003274 | 0.000000 +- 0.000000 |
| wo_identity_memory | 0.053544 +- 0.015717 | 0.053455 +- 0.015663 | 0.074074 +- 0.026189 | 0.370370 +- 0.445215 | 0.053241 +- 0.019913 | 0.946759 +- 0.019913 | 0.000000 +- 0.000000 |

## Delta vs full (mean)

Sign convention: `delta = compare - full`.

- `wo_semantics`
  - trajectory `+0.003837`
  - query error `+0.003702`
  - query top1 `+0.055556`
  - query hit `-0.296296`
  - identity consistency `+0.004630`
  - identity switch `-0.004630`

- `wo_identity_memory`
  - trajectory `+0.008733`
  - query error `+0.008508`
  - query top1 `+0.018519`
  - query hit `-0.314815`
  - identity consistency `-0.023148`
  - identity switch `+0.023148`

## Per-Seed Stability

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

Stability verdict:

- `full > wo_semantics`: still not stable across seeds
- `full > wo_identity_memory`: still not stable across seeds

## Query and Occlusion Status

- Query remains decoupled from trajectory (no identical-value regression to v1).
- `occlusion_recovery_acc` remains all-zero for all runs/seeds in V2.3.

## Failure Figure Pack

- Manifest:
  - `outputs/visualizations/week2_figures_v2_3/figure_manifest.json`
- Panels:
  - `full_fail_wo_identity_worse` (4)
  - `full_success_wo_semantics_fail` (4)
  - `query_hard_success_failure` (4)

## Final Round Decision

V2.3 did not satisfy the stability criteria for advancing to model scaling.

Decision:

1. Stop evaluator churn here (no V2.4/V2.5).
2. Keep semantic trajectory state as the primary story.
3. Downgrade identity memory to secondary/exploratory contribution.