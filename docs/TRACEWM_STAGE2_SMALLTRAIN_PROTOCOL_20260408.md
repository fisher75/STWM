# TRACEWM Stage2 Small-Train Protocol (2026-04-08)

## 1. Frozen Facts

1. Stage1 220m backbone is already fully frozen and ready.
   - whether_stage1_backbone_is_now_fully_ready = true
   - next_step_choice = freeze_stage1_and_prepare_stage2
2. Stage2 bootstrap has completed and passed.
   - bootstrap_ready = true
   - next_step_choice = start_stage2_small_train
3. Stage2 binding for this small-train round is fixed.
   - core: VSPW + VIPSeg
   - optional extension: BURST
   - TAO = access_ready (not in current main training)
   - VISOR = manual_gate (not in current main training)

## 2. Round Scope and Locked Boundaries

This round only executes Stage2 small-train on top of frozen Stage1.

The following are explicitly not changed in this round:
1. Stage1 backbone family
2. Stage1 frozen policy
3. Stage2 data contract framework
4. WAN / MotionCrafter
5. video reconstruction

## 3. Allowed Runs (Exactly Two)

Only the following runs are allowed:

1. stage2_smalltrain_core
   - datasets: VSPW + VIPSeg
   - role: primary run

2. stage2_smalltrain_core_plus_burst
   - datasets: VSPW + VIPSeg + BURST
   - role: optional extension run

Forbidden in this round:
1. TAO
2. VISOR
3. new architecture search
4. new loss family search
5. Stage2 full longtrain
6. new dataset onboarding
7. new backbone onboarding

## 4. Freeze/Trainable Policy

Continue Stage2 bootstrap freeze policy without modification.

Frozen:
1. Stage1 220m backbone
2. Stage1 tokenizer / core rollout backbone

Trainable:
1. semantic encoder
2. semantic fusion / adapter
3. optional lightweight readout head

Mandatory checks:
1. record frozen and trainable parameter counts during training
2. verify Stage1 backbone remains frozen after training
3. any backbone unfreeze is a protocol violation

## 5. Budget and Checkpoint Rules

This is small-train, not full-scale longtrain.

1. Both runs must use the same budget level.
2. Budget must be meaningfully above bootstrap smoke while still controlled.
3. Checkpoint and resume must be supported.
4. Save policy is fixed:
   - save_every_n_steps = 1000
   - best.pt
   - latest.pt
   - step_0001000.pt if reached
   - step_0002000.pt if reached
   - continue by 1000-step milestones
5. If total steps < 1000, each run must still keep best.pt and latest.pt.

## 6. Evaluation and Ranking Policy

Each run must report:
1. teacher_forced_coord_loss
2. free_rollout_coord_mean_l2
3. free_rollout_endpoint_l2
4. TAP-Vid style eval (if current Stage2 trainer is compatible)
5. TAPVid-3D limited eval (if current Stage2 trainer is compatible)
6. semantic branch metrics (if available)
7. parameter_count_frozen
8. parameter_count_trainable

Ranking order is fixed:
1. primary: free_rollout_endpoint_l2
2. secondary: free_rollout_coord_mean_l2
3. tertiary: available eval metric

total_loss is reference only.

## 7. Required Outputs

This round must generate:
1. reports/stage2_smalltrain_runs_20260408.json
2. reports/stage2_smalltrain_comparison_20260408.json
3. docs/STAGE2_SMALLTRAIN_RESULTS_20260408.md

Checkpoint directories:
1. outputs/checkpoints/stage2_smalltrain_core_20260408/
2. outputs/checkpoints/stage2_smalltrain_core_plus_burst_20260408/

## 8. Mandatory Comparison Questions

Comparison must explicitly answer:
1. whether core-only run trains stably
2. whether core+burst is better than core-only
3. whether Stage1 frozen boundary remains correct
4. whether Stage2 is now smalltrain_successful or needs_bootstrap_fix
5. next_step_choice in:
   - continue_stage2_training
   - keep_core_only_and_continue
   - refine_stage2_smalltrain

## 9. Fixed Runtime Envelope

1. single-GPU recommended runtime + selector/lease policy
2. fixed tmux session: tracewm_stage2_smalltrain_20260408
3. fixed log: /home/chen034/workspace/stwm/logs/tracewm_stage2_smalltrain_20260408.log