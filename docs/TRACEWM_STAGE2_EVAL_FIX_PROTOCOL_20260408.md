# TRACEWM Stage2 Eval-Fix Protocol (2026-04-08)

## 1. Frozen Facts

1. Stage2 semantic-source hardening has completed.
2. current_mainline_semantic_source = crop_visual_encoder.
3. frozen_boundary_kept_correct = true.
4. Hardening conclusion: core-only is better than core+burst.
5. Hardening next step was fixed as next_step_choice = do_one_more_stage2_eval_fix.

## 2. Problem Statement For This Round

1. The bottleneck is no longer semantic source selection.
2. The bottleneck is Stage2 evaluation protocol strength for paper-level decisions.
3. This round hardens protocol comparability and decision fields only.

## 3. Locked Non-Goals (Do Not Change)

1. Stage1 backbone and Stage1 training state.
2. Stage2 semantic source mainline.
3. Stage2 data binding:
   - core = VSPW + VIPSeg
   - BURST remains optional
4. WAN / MotionCrafter.
5. TAO / VISOR mainline.
6. Stage2 long-train continuation.

## 4. Required Unified Stage2 Eval Outputs

Each Stage2 run summary must explicitly expose:

1. teacher_forced_coord_loss
2. free_rollout_coord_mean_l2
3. free_rollout_endpoint_l2
4. best_checkpoint_metric
5. latest_checkpoint_metric
6. train/val split counts actually used
7. frozen/trainable parameter counts
8. boundary_ok

## 5. Fixed Comparison Sorting

Comparison must always use:

1. primary = free_rollout_endpoint_l2
2. secondary = free_rollout_coord_mean_l2
3. tertiary = teacher_forced_coord_loss
4. total_loss is reference-only and not ranking criterion

## 6. Mandatory Comparison Fields

Comparison json must include:

1. primary_winner
2. secondary_winner
3. tertiary_winner
4. final_recommended_mainline
5. why_burst_not_better (when burst loses)
6. can_continue_stage2_training

## 7. Mandatory Comparability Checks

Comparison must explicitly state:

1. datasets_bound_for_core
2. datasets_bound_for_core_plus_burst
3. whether_same_budget
4. whether_same_frozen_policy
5. whether_same_eval_protocol

If any of items 3-5 is false:

1. mark final_recommended_mainline = invalid_comparison
2. stop Stage2 training continuation
3. set next_step_choice = fix_comparison_first

Only when items 3-5 are all true can a mainline recommendation be emitted.

## 8. Allowed Operation Envelope

1. Reuse existing hardening run json when sufficient.
2. Allow only very short re-summary/re-eval if key fields are missing.
3. No large sweep.
4. No semantic source mainline switch.
5. No TAO/VISOR onboarding.

## 9. Runtime And Logging Contract

1. Fixed tmux session: tracewm_stage2_eval_fix_20260408.
2. Fixed log path: /home/chen034/workspace/stwm/logs/tracewm_stage2_eval_fix_20260408.log.
