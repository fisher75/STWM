# Stage2 Eval-Fix Results

- generated_at_utc: 2026-04-09T04:39:20.273613+00:00
- current_mainline_semantic_source: crop_visual_encoder
- core_only_better_than_core_plus_burst: True
- frozen_boundary_kept_correct: True
- final_recommended_mainline: stage2_core_cropenc
- can_continue_stage2_training: True
- next_step_choice: continue_stage2_training_core_only
- invalid_comparison: False

## Sorting
- primary: free_rollout_endpoint_l2
- secondary: free_rollout_coord_mean_l2
- tertiary: teacher_forced_coord_loss
- total_loss_usage: reference_only

## Comparability
- datasets_bound_for_core: ['vspw', 'vipseg']
- datasets_bound_for_core_plus_burst: ['vspw', 'vipseg', 'burst']
- whether_same_budget: True
- whether_same_frozen_policy: True
- whether_same_eval_protocol: True

## Winners
- primary_winner: stage2_core_cropenc
- secondary_winner: stage2_core_cropenc
- tertiary_winner: stage2_core_cropenc
- why_burst_not_better: burst is not better because endpoint/mean/teacher losses are not lower than core-only: endpoint_delta(burst-core)=0.004457, coord_mean_delta(burst-core)=0.004457, teacher_delta(burst-core)=0.000459

## Core Metrics
- teacher_forced_coord_loss: 0.000470
- free_rollout_coord_mean_l2: 0.020339
- free_rollout_endpoint_l2: 0.020339

## Core+Burst Metrics
- teacher_forced_coord_loss: 0.000928
- free_rollout_coord_mean_l2: 0.024796
- free_rollout_endpoint_l2: 0.024796
