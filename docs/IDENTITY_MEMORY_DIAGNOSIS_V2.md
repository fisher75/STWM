# Identity Memory Diagnosis V2

## What V2 Changed

Relative to v1, V2 introduced target-label-aware evaluation and switched identity metrics to label-hit behavior.

- Identity metrics are no longer trivially saturated at `1.0/0.0`.
- Query metric is no longer exactly the same value as trajectory L1.

## Current V2 Observation

At step 80:

- `full` trajectory/query: `0.035330 / 0.035353`
- `wo_identity_memory` trajectory/query: `0.060368 / 0.060225`

This indicates meaningful degradation without identity memory under the harder protocol.

## Why Identity Metrics Still Look Extreme

The new identity metrics are now sensitive, but may be too brittle:

- `identity_consistency` is very low across runs.
- `identity_switch_rate` is very high across runs.
- `occlusion_recovery_acc` remains `0.0` in all runs.

Likely causes:

1. Point-on-label criterion is strict and penalizes small localization drift heavily.
2. Selected target labels may be thin/small or unstable in segmentation masks.
3. Recovery event definition still has low support in this mini split.

## Is Identity Memory Now Supported?

Partially.

- On trajectory/query error, yes: removing identity memory hurts most at final step.
- On dedicated identity/occlusion metrics, not yet convincing due to metric harshness and recovery zero-floor.

## Recommended V2.1 Fixes (Before Any Scale-Up)

1. Relax identity hit definition:
   - use small neighborhood overlap around predicted point (disk hit), not exact pixel class.
2. Add soft identity score:
   - combine point-hit and distance-to-target-mask-centroid.
3. Rework recovery metric:
   - score over short recovery window (first 2-3 frames after re-appearance), not only first frame.
4. Keep target-label selection but add minimum target area and temporal persistence filters.

## Decision Gate

- Keep identity memory as a main module candidate for now.
- Do not claim strong identity story until V2.1 metrics become non-degenerate and reproduce across seeds.
