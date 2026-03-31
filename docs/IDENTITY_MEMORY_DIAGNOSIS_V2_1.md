# Identity Memory Diagnosis V2.1

## What Improved vs V2

- Identity hit moved from exact point-hit to radius-hit (`identity_hit_radius=0.035`).
- Recovery moved from single-frame to short-window scoring (`window=3`).
- Query candidates now include harder negatives and same-class hard negatives when possible.

## Current Observations

From multi-seed aggregate:

- `identity_consistency` remains low for all runs.
- `identity_switch_rate` remains pinned at `1.0` for all runs and seeds.
- `occlusion_recovery_acc` remains `0.0` for all runs and seeds.

Interpretation:

- Metrics are no longer the v1-style trivial saturation, but identity/occlusion probes are still degenerate in practice.
- The new criterion is likely over-penalizing and still not aligned with robust identity behavior.

## Is Identity Memory Supported in V2.1?

Weakly, not decisively.

- Aggregate trajectory/query means prefer full over `wo_identity_memory`.
- But seed-level ranking is not stable (`seed 456` reverses the expected ordering).

So identity memory has signal, but not yet credible as a stable main claim.

## Likely Failure Modes

1. Target mask and neighborhood overlap are still too brittle on thin/small regions.
2. `identity_switch_rate` counts non-target foreground overlap too aggressively.
3. Recovery events may be sparse or mislabeled under current target-label extraction.

## Recommended V2.2 Adjustments

1. Use soft identity score:
   - combine target-overlap hit and distance-to-target-centroid.
2. Redefine switch criterion:
   - count switch only when non-target overlap exceeds a minimum area ratio.
3. Improve recovery metric support:
   - require explicit disappearance duration before counting re-appearance events.
4. Keep same split and rerun 3 seeds for comparability before any scale-up.

## Decision

Identity memory should stay as a candidate module, but should not be promoted as a confirmed core contribution yet.
