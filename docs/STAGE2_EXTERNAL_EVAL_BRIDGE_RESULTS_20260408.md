# Stage2 External Eval Bridge Results

- generated_at_utc: 2026-04-09T06:16:11.768902+00:00
- current_stage2_mainline_checkpoint: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt
- current_mainline_semantic_source: crop_visual_encoder
- frozen_boundary_kept_correct: True
- external_eval_connected: True
- tap_style_eval_status: implemented_and_run
- tap3d_style_eval_status: not_yet_implemented
- readiness: training_ready_but_eval_gap_remains
- next_step_choice: do_one_targeted_external_eval_fix

## Eval Binding
- datasets_bound_for_eval: ['vspw', 'vipseg']

## Primary Checkpoint Eval
- checkpoint_under_test: /home/chen034/workspace/stwm/outputs/checkpoints/stage2_core_mainline_train_20260408/best.pt
- tap_style_eval_status: implemented_and_run
- tap3d_style_eval_status: not_yet_implemented
- tap_style_payload_npz: /home/chen034/workspace/stwm/reports/stage2_external_eval_bridge_tap_style_payload_20260408.npz
- tap_style_free_rollout_endpoint_l2: 0.004344
- tap_style_free_rollout_coord_mean_l2: 0.004344
- tap3d_blocking_reason: stage2 core state currently exposes 2D track targets only; no verified 3D target alignment is available in bridge path
