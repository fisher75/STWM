# STWM V4.2 Evaluator Freeze (Phase A)

Date: 2026-04-03
Status: FROZEN

## Official Detached Evaluator

- Script: code/stwm/evaluators/eval_mini_val.py
- Evaluator version tag: v2_4_detached_frozen
- Protocol alias rule: requested v2_4_detached_frozen is canonically evaluated as v2_3 metric logic

This file/script/version triplet is now the only official detached evaluator contract for STWM V4.2 protocol claims.

## Frozen Evaluation Contract

1. Official detached artifacts must be produced by code/stwm/evaluators/eval_mini_val.py.
2. Output summary must contain protocol metadata fields:
   - evaluator_version
   - requested_protocol_version
   - protocol_version
   - dataset
3. Mixed-manifest evaluation is allowed and standardized via --dataset all.
4. Legacy checkpoints (stwm_1b) and V4.2 checkpoints (stwm_v4_2) are both supported in one evaluator path.

## Stable Comparable Metrics (Locked)

The following metric key set is frozen for cross-run comparability:

- query_localization_error (lower better)
- query_top1_acc (higher better)
- query_hit_rate (higher better)
- identity_consistency (higher better)
- identity_switch_rate (lower better)
- occlusion_recovery_acc (higher better)
- future_trajectory_l1 (lower better)
- future_mask_iou (higher better)
- visibility_accuracy (higher better)
- visibility_f1 (higher better)

These keys are emitted in summary.metrics and mirrored in summary.protocol.stable_comparable_metrics.

## Claim Boundary

- Pre-freeze artifacts remain valid for engineering diagnosis.
- Official post-freeze claim evidence must cite evaluator_version = v2_4_detached_frozen and use detached artifacts generated under this contract.
