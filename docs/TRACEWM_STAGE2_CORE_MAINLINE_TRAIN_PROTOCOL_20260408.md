# TRACEWM Stage2 Core Mainline Training Protocol (2026-04-08)

## 1. Frozen Facts From Latest Eval-Fix

1. Latest eval-fix has completed.
2. final_recommended_mainline = stage2_core_cropenc.
3. can_continue_stage2_training = true.
4. next_step_choice = continue_stage2_training_core_only.
5. core-only is better than core+burst.

## 2. Mainline Scope For This Round

1. This round executes one formal Stage2 core-only mainline training run.
2. This round target is first formal recoverable/comparable/reproducible mainline evidence.
3. This round is not full-scale paper-final longtrain.

## 3. Locked Non-Goals (Do Not Change)

1. Stage1 backbone.
2. Stage2 semantic source mainline (must remain crop_visual_encoder).
3. Stage2 frozen/trainable boundary.
4. Stage2 data contract and data family.
5. BURST must not enter current mainline training run.
6. WAN / MotionCrafter.
7. TAO / VISOR mainline.

## 4. Allowed Single Run

Only one run is allowed:

1. stage2_core_mainline_train_20260408

Hard constraints:

1. datasets = VSPW + VIPSeg only.
2. semantic source mainline = crop_visual_encoder.
3. hand-crafted stats cannot be promoted to mainline.
4. no BURST in this run.
5. no backbone/loss/data-contract family replacement.

## 5. Frozen/Trainable Boundary Contract

Frozen:

1. Stage1 220m backbone.
2. Stage1 tokenizer and rollout backbone.

Trainable:

1. Stage2 semantic encoder/crop encoder.
2. Stage2 semantic fusion/adapter.
3. optional lightweight readout head.

Mandatory checks:

1. pre/post frozen/trainable parameter counts are logged.
2. stage1_trainable_parameter_count = 0.
3. boundary_ok = true.

## 6. Budget And Checkpoint Contract

1. Use medium budget significantly above prior small-train rounds.
2. Keep checkpoint/resume infra unchanged:
   - best.pt
   - latest.pt
   - resume_from
   - auto_resume_latest
   - save_every_n_steps = 1000
3. If steps < 1000, still require best.pt and latest.pt.
4. Report optimizer_steps, effective_batch, epochs, eval_interval.

## 7. Fixed Evaluation And Ranking Policy

Required metrics in run summary:

1. teacher_forced_coord_loss
2. free_rollout_coord_mean_l2
3. free_rollout_endpoint_l2
4. frozen/trainable parameter counts
5. boundary_ok
6. best_checkpoint_metric
7. latest_checkpoint_metric
8. current_mainline_semantic_source
9. datasets_bound_for_train
10. datasets_bound_for_eval

Sorting rule is fixed:

1. primary = free_rollout_endpoint_l2
2. secondary = free_rollout_coord_mean_l2
3. tertiary = teacher_forced_coord_loss
4. total_loss is reference-only

## 8. End-Of-Round Mandatory Answers

Final json/markdown must explicitly answer:

1. current_mainline_semantic_source is crop_visual_encoder or not.
2. frozen boundary still correct or not.
3. current_stage2_mainline stable or not.
4. whether_curve_is_still_improving = true/false.
5. next_step_choice must be one of:
   - continue_stage2_training_core_only
   - freeze_stage2_core_mainline
   - do_one_targeted_stage2_fix

## 9. Runtime Contract

1. fixed tmux session: tracewm_stage2_core_mainline_train_20260408
2. fixed log: /home/chen034/workspace/stwm/logs/tracewm_stage2_core_mainline_train_20260408.log
