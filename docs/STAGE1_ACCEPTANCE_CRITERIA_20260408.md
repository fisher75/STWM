# Stage 1 Acceptance Criteria (2026-04-08)

## Objective

Formalize and freeze the Stage 1 Trace-only data protocol and execution interface.

## Data Acceptance Baseline

1. Main training ready:
- PointOdyssey = complete under sequence-based hard-complete criteria.
- Kubric (movi_e-only standard, USE_PANNING_REQUIRED=0) = complete.

2. Main evaluation ready:
- TAP-Vid = complete for Stage 1 eval interface.

3. Limited evaluation status:
- TAPVid-3D = partial but limited_eval_ready=true.

4. Optional/non-blocking:
- DynamicReplica path remains non-blocking and not in first-wave Stage 1.

## Unified Sample Contract (Mandatory)

Every loader must emit the same sample dict keys:
- dataset
- split
- clip_id
- obs_frames
- fut_frames
- obs_valid
- fut_valid
- obs_tracks_2d
- fut_tracks_2d
- obs_tracks_3d
- fut_tracks_3d
- visibility
- intrinsics
- extrinsics
- point_ids
- meta

No per-dataset key drift is allowed.

## Smoke Gate Before Tiny-Train

All four smoke checks must pass:
1. contract smoke
2. loader single-sample smoke
3. visual smoke
4. batch smoke

If any smoke fails, tiny-train must not start.

## Tiny-Train Acceptance

- Dataset scope: PointOdyssey + Kubric minisplits only.
- Task scope: trace/state forecast only.
- Must run both:
- teacher-forced mode
- free-rollout mode
- Very short sanity run is sufficient.

## Eval Smoke Acceptance

- TAP-Vid eval_mini: 2D eval smoke runnable.
- TAPVid-3D eval_mini: limited 3D eval smoke runnable.
- Must explicitly separate:
- full eval not ready
- limited eval ready

## Round Exit Condition

Stage 1 enters coding-iteration-ready only after:
- data contract fixed
- minisplits fixed and reproducible
- 4 smokes passed
- tiny-train and eval-smoke artifacts generated
