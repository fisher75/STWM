# STWM V4.2 Protocol Best Rule V2 (Phase C)

Date: 2026-04-03
Rule version: protocol_best_rule_v2
Status: FROZEN

## Official Best Checkpoint Name

- Official best checkpoint: best_protocol_main.pt

This is the only checkpoint name that can be cited as protocol-frozen official best.

## Ranking Rule (Lexicographic)

Candidate checkpoints are compared using detached eval results on protocol_val_main_v1 with this strict order:

1. primary metric: query_localization_error (lower better)
2. tie-break 1: query_top1_acc (higher better)
3. tie-break 2: future_trajectory_l1 (lower better)

If all three are equal within epsilon, incumbent is kept.

## Implementation

- Updater tool: code/stwm/tools/update_protocol_best_main.py
- Runner script: scripts/run_stwm_v4_2_protocol_candidate_eval.sh
- Sidecar evidence: best_protocol_main_selection.json

The sidecar records:

- rule version and metric definitions
- candidate checkpoint path
- eval summary path
- candidate metric triplet
- action (updated or kept_existing)
- incumbent metrics when available

## Candidate Evaluation Loop (Frozen)

For long training, evaluate candidate every 500 optimizer steps:

1. detached eval on protocol_val_main_v1
2. apply lexicographic rule
3. update best_protocol_main.pt only if improved
4. write selection sidecar snapshot

No training-loss metric is allowed to override this protocol best rule.

## Metric Tiering Constraint

- identity_consistency, identity_switch_rate, occlusion_recovery_acc, reconnect-related metrics are retained as gated diagnostics.
- They are not part of the official best ranking tuple at this phase.
- If future coverage gates are satisfied, a future rule version may promote additional axes, but not in rule v2.
