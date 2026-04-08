# TraceWM Stage 1 Final Joint Rescue (2026-04-08)

## Frozen Current Facts (Directly From Existing Report)

- source_report: /home/chen034/workspace/stwm/reports/tracewm_stage1_fix2_comparison_20260408.json
- answers.q4_best_joint_recipe.winner = tracewm_stage1_fix2_joint_balanced_lossnorm
- answers.q5_any_joint_surpasses_best_single.value = false
- final_recommendation = stop_joint_and_keep_best_single

## Final-Round Scope (Last Joint Rescue)

- This round is the last joint rescue round.
- Only the following methods are allowed:
  - PCGrad
  - GradNorm
  - shared-private adapters

## Fixed Baseline Requirement

- All runs must be built on the current best joint baseline:
  - balanced sampler + loss normalization

## Hard Stop Rule

- If this final rescue round still has no joint run surpassing best single,
  joint mainline is formally stopped and only best single is kept for next-stage preparation.

## Explicitly Forbidden In This Round

- warmup
- source-conditioning variants
- Stage 2 semantics
- WAN
- MotionCrafter VAE
- DynamicReplica
- new data
- video reconstruction / render loss
