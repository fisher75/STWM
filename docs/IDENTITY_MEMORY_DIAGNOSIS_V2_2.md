# Identity Memory Diagnosis V2.2

## What Changed in V2.2

Relative to V2.1, V2.2 applies minimal thresholding and event gating:

- identity hit requires minimum target overlap (`identity_target_overlap_min=0.02`)
- switch requires sufficient non-target overlap (`identity_other_overlap_min=0.15`) and failed target hit
- occlusion recovery requires minimum disappearance gap (`occlusion_min_disappear_frames=2`)

No model architecture or data split change was introduced.

## Aggregate Observations (3 seeds)

- full:
  - `identity_consistency = 0.076389 +- 0.005670`
  - `identity_switch_rate = 0.923611 +- 0.005670`
- wo_identity_memory:
  - `identity_consistency = 0.050926 +- 0.021467`
  - `identity_switch_rate = 0.949074 +- 0.021467`

Directionally, full is better than `wo_identity_memory` on identity metrics in aggregate.

## Paired Evidence (Clip-Level, 54 Paired Samples)

From `reports/week2_minival_v2_2_paired_analysis.json`:

- `identity_consistency` delta (`wo_identity_memory - full`):
  - mean `-0.025463`
  - bootstrap 95% CI `[-0.092593, 0.041667]`
- `identity_switch_rate` delta (`wo_identity_memory - full`):
  - mean `+0.025463`
  - bootstrap 95% CI `[-0.041667, 0.092593]`

Interpretation:

- point estimates favor full
- confidence intervals cross zero for both identity metrics
- identity improvement is not yet statistically robust

## Occlusion Status

- `occlusion_recovery_acc` remains `0.0` for all runs and seeds.
- Current protocol still lacks enough informative recovery events under this clip set.

## Diagnosis

1. V2.2 reduced trivial counting artifacts, but identity metrics remain near-degenerate.
2. Identity signal exists in mean deltas, but uncertainty is high and seed ranking is unstable.
3. Occlusion metric remains non-informative and cannot support module-level claims.

## Claim Boundary

Identity memory should remain a tentative, secondary contribution at this stage.

- acceptable claim: "identity branch may help under specific clips"
- not yet acceptable claim: "identity memory is a robust primary driver"

## Recommended Next Step (Before Any Scale-Up)

- keep 220M scale and fixed split
- improve event coverage for occlusion/identity probes (clip-level curation or protocol filtering)
- retain paired analysis as default report criterion