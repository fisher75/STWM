# TRACEWM Stage2 External-Eval Bridge Protocol (2026-04-08)

## 1. Frozen Facts

1. Stage2 core mainline train has completed.
2. current_mainline_semantic_source = crop_visual_encoder.
3. frozen_boundary_kept_correct = true.
4. next_step_choice = freeze_stage2_core_mainline.

## 2. This Round Does Not Continue Training

1. No Stage2 training continuation.
2. No Stage2 long-train.
3. No Stage1 training.
4. No architecture search.

## 3. Main Gap For This Round

1. The primary gap is external evaluation bridge readiness.
2. Existing trainer-level tapvid_style_eval / tapvid3d_limited_eval are still not sufficient for paper-facing external evaluation.

## 4. Locked Non-Goals

This round does not modify:

1. Stage1 backbone.
2. Stage2 core mainline parameters.
3. Stage2 semantic source mainline.
4. Stage2 data binding family.
5. WAN / MotionCrafter.
6. BURST / TAO / VISOR in Stage2 mainline.

## 5. Allowed Operations

1. Load frozen Stage2 core mainline checkpoint.
2. Run evaluation-side bridge only.
3. Export compatible external-eval payload(s).
4. Produce bridge json and bridge markdown reports.

## 6. Evaluation Target Scope

1. Primary checkpoint under test:
   - outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt
2. Optional secondary reference:
   - outputs/checkpoints/stage2_core_mainline_train_20260408/latest.pt
3. Eval data binding remains core-only:
   - VSPW + VIPSeg

## 7. External-Eval Output Contract

Bridge output must include:

1. checkpoint_under_test
2. datasets_bound_for_eval
3. external_eval_protocol_version
4. tap_style_eval_status
5. tap3d_style_eval_status
6. implemented metrics when runnable
7. exact blocking reason and exact missing component when partially/not implemented

Status values must be chosen from:

1. implemented_and_run
2. partially_bridged
3. not_yet_implemented

## 8. End-Of-Round Mandatory Decisions

Bridge result must explicitly answer:

1. current_stage2_mainline_checkpoint
2. whether external_eval is truly connected
3. readiness is one of:
   - paper_eval_ready
   - training_ready_but_eval_gap_remains
4. next_step_choice is one of:
   - finalize_stage2_mainline_and_prepare_paper_results
   - do_one_targeted_external_eval_fix
   - revisit_stage2_eval_inputs

## 9. Runtime Contract

1. fixed tmux session: tracewm_stage2_external_eval_bridge_20260408
2. fixed log: /home/chen034/workspace/stwm/logs/tracewm_stage2_external_eval_bridge_20260408.log
