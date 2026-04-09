# Stage2 Core Mainline Train Results

- generated_at_utc: 2026-04-09T05:39:07.531179+00:00
- run_name: stage2_core_mainline_train_20260408
- current_mainline_semantic_source: crop_visual_encoder
- frozen_boundary_kept_correct: True
- current_stage2_mainline_stable: True
- whether_curve_is_still_improving: False
- next_step_choice: freeze_stage2_core_mainline

## Dataset Binding
- datasets_bound_for_train: ['vspw', 'vipseg']
- datasets_bound_for_eval: ['vspw', 'vipseg']

## Metrics
- teacher_forced_coord_loss: 0.000022
- free_rollout_coord_mean_l2: 0.004344
- free_rollout_endpoint_l2: 0.004344

## Checkpoint Metrics
- best_checkpoint_metric.global_step: 900
- latest_checkpoint_metric.global_step: 1200

## Training Progress
- optimizer_steps: 1200
- effective_batch: 2
- epochs_completed: 25.000000
- eval_interval: 100
