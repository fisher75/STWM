# Identity Memory Diagnosis V2.3

## Objective of V2.3

V2.3 was the final minimal attempt to make identity/occlusion evaluation informative without changing model/data/split.

Key changes:

- identity switch scored with short-sequence consistency window
- reconnect-style occlusion recovery scoring
- slight query hardening with same-class plausibility floor

## Aggregate Observation

From `reports/week2_minival_v2_3_multiseed_summary.json`:

- full:
  - `identity_consistency = 0.076389 +- 0.009821`
  - `identity_switch_rate = 0.923611 +- 0.009821`
- wo_identity_memory:
  - `identity_consistency = 0.053241 +- 0.019913`
  - `identity_switch_rate = 0.946759 +- 0.019913`

Direction remains favorable to full in aggregate.

## Paired Evidence (54 paired clips)

From `reports/week2_minival_v2_3_paired_analysis.json`:

- identity consistency delta (`wo_identity_memory - full`):
  - mean `-0.023148`
  - 95% bootstrap CI `[-0.090278, 0.043981]`
- identity switch delta (`wo_identity_memory - full`):
  - mean `+0.023148`
  - 95% bootstrap CI `[-0.043981, 0.090278]`

Interpretation:

- expected direction exists in point estimate
- confidence intervals still cross zero
- evidence remains weak, not robust enough for primary claim

## Occlusion Probe Status

- `occlusion_recovery_acc` is still `0.0` for all runs and seeds.
- reconnect-style scoring did not resolve metric informativeness under this split.

## Conclusion

Identity memory should be formally downgraded to secondary/exploratory contribution in the current paper framing.

Recommended claim boundary:

- acceptable: identity branch shows limited directional signal in some settings
- not acceptable: identity memory is a stable core driver across seeds

## Action After V2.3

Given the explicit stop condition, do not continue evaluator churn.

- no V2.4/V2.5
- proceed with narrowed story centered on semantic trajectory state