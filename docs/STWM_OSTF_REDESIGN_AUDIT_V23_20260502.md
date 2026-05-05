# STWM OSTF Redesign Audit V23

## Purpose
- This is an audit only.
- It does not authorize more same-setup OSTF training.
- The question is whether there is a credible `OSTF-v2` route beyond the current CoTracker-teacher H8/H16 regime.

## TraceAnything
- Local official repo exists:
  - `/raid/chen034/workspace/stwm/third_party/TraceAnything`
- Local checkpoint exists:
  - `/raid/chen034/workspace/stwm/models/checkpoints/traceanything/traceanything_pretrained.pt`
- The local README states the examples were tested on a single GPU with `>= 48 GB VRAM`.
- That makes a `B200` pilot feasible in principle.
- Main blocker is not availability. Main blocker is integration:
  - trajectory-field outputs must be converted into persistent per-object sampled tracks
  - object binding and query strategy must be re-audited
  - storage policy must be decided before scaling

## PointOdyssey
- Local dataset wrapper code exists.
- Old training outputs exist.
- A live PointOdyssey dataset payload was not found under the current `data/` root.
- So PointOdyssey is not immediately runnable as a GT benchmark from the present live repo.

## TAP-Vid / TAPIR
- TAP-Vid wrappers and small stage1 cache artifacts exist locally.
- No official TAPIR repo/weights were found in the live repo.
- This means TAP-style evaluation is plausible as a redesign direction, but not turnkey today.

## H32 / H64
- Current H16 already loses clips because of insufficient contiguous frames.
- So H32/H64 should not be treated as a straightforward next scaling step.
- They need re-windowing from longer source sequences and a new eligibility policy.

## B200 Cost Story
- Current CoTracker sampled-point cache gives us a concrete storage anchor:
  - about `623 MB` sampled caches
  - about `107 MB` videos
  - for `2132` clips across `M128/M512` and `H8/H16`
- For TraceAnything, the live repo does not yet contain measured inference logs.
- The best honest statement is an engineering estimate:
  - sampled-point cache only: still modest
  - raw dense trajectory-field intermediates: potentially very large

## Recommendation
- If OSTF continues, do not continue with more same-setup small models.
- Run a `TraceAnything redesign only` path first.
- A serious `OSTF-v2` likely needs:
  - a denser or stronger teacher
  - a harder point-trajectory benchmark
  - redesigned long-horizon clip windowing
