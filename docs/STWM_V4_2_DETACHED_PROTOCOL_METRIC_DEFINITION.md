# STWM V4.2 Detached Protocol Metric Definition (Defensible Claim Set)

Date: 2026-04-03  
Protocol baseline: `eval_mini_val.py --protocol-version v2_3`

## Goal

Define the strongest currently executable, cross-run comparable, detached protocol-level metric family for STWM V4.2 real checkpoints.

## Metric Family (v2_3)

All metrics are read from detached artifacts produced by `code/stwm/evaluators/eval_mini_val.py`.

### Tier A: Main-claim metrics (preferred)

1. Query grounding
   - `query_localization_error` (lower is better)
   - `query_top1_acc` (higher is better)
   - `query_hit_rate` (higher is better)
2. Identity continuity
   - `identity_consistency` (higher is better)
   - `identity_switch_rate` (lower is better)
3. Occlusion reconnect
   - `occlusion_recovery_acc` (higher is better), only claimable when recovery events are actually present in the evaluated slice.

Why Tier A: these are detached protocol objectives tied to retrieval, identity stability, and reconnect behavior rather than trainer-internal loss proxies.

### Tier B: Supporting protocol proxies

- `future_trajectory_l1` (lower is better)
- `future_mask_iou` (higher is better)
- `visibility_accuracy` (higher is better)
- `visibility_f1` (higher is better)

Why Tier B: useful for context and sanity checks, but less direct for identity/query claim narrative.

## Comparison Rules

1. Use paired comparisons only:
   - same scale
   - same seed
   - same checkpoint kind (`best` vs `latest`)
2. For semantic contribution, report:
   - `delta = metric(wo_semantics) - metric(full)`
3. Direction interpretation:
   - Lower-is-better metric: `delta > 0` means semantics helped.
   - Higher-is-better metric: `delta < 0` means semantics helped.
4. Report both:
   - pairwise deltas (to expose sign flips)
   - mean delta across pairs (to summarize direction)

## Claimability Gates

A metric is claimable only if all gates pass:

1. Executable gate: metric is produced by detached evaluator (not placeholder).
2. Coverage gate: denominator/event coverage is non-trivial for that metric slice.
3. Stability gate: direction is not purely single-pair noise (show all pairs; avoid single-run cherry-pick).

If a gate fails, downgrade that metric to diagnostic status.

## Explicit Non-Claims

- This family does not substitute for standard MOT benchmark metrics (HOTA/MOTA/IDF1).
- If occlusion-event coverage is absent or near-zero, do not claim reconnect superiority.
- If identity metrics collapse to near-constant values across conditions, avoid strong identity superiority statements.

## Required Reporting Block Per Artifact

At minimum, each report row must include:

- run identity: scale, seed, run, checkpoint kind
- protocol metadata: version, obs/pred steps
- `num_clips`
- Tier A metrics (always)
- Tier B metrics (context)
- ablation flags (`disable_semantics`, etc.)

This keeps comparisons defendable and reviewer-auditable.
