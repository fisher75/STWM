# STWM Planning-Lite Task Build 20260425

This asset is a planning-lite candidate-path risk scoring probe, not closed-loop autonomous driving.

## Candidate Path Construction

For each protocol item, `K=6` deterministic synthetic path corridors are created. Each future dynamic object entity receives two high-risk corridors by stable hash. Candidate paths are synthetic because source per-item reports do not include real ego future trajectories or object coordinates.

## Risk Label Construction

Labels use the target entity id from `protocol_item_id` and synthetic target-path proximity >= `0.65`. Method risk scores use only each method's `top1_candidate_id`, not protocol top1, MRR, or target rank.

## Split

| split | item_count | hash |
|---|---:|---|
| train | 85 | `461995aab8b1ce3ba54885d517d7f7d2d6df4d689515cdd8a23cc7fde683561d` |
| val | 90 | `1ac00787bad9f9028ee85ec21145a09600c5a5303c9b6d709d5c832d449c87c0` |
| test | 405 | `c90b1b804cc113717f6a25481a958867f5ec5b36d54a7705f86660e05f4fcab9` |

- no_leakage_check_passed: `True`
- item_count: `580`
- limitations: `This is a planning-lite risk probe, not closed-loop autonomous driving planning.; Candidate paths are synthetic deterministic corridors because source reports do not contain real ego future trajectories or object coordinates.; Risk score uses method top1_candidate_id only; raw candidate score maps are unavailable.; No matching per-item rows in stwm_trace_belief_eval_20260424.json for this method/scoring_mode.`
