# Stage2 Small-Train Results

- generated_at_utc: 2026-04-08T17:19:12.088143+00:00
- smalltrain_status: smalltrain_successful
- next_step_choice: continue_stage2_training

## Run Metrics
| run | teacher_forced_coord_loss | free_rollout_coord_mean_l2 | free_rollout_endpoint_l2 | parameter_count_frozen | parameter_count_trainable |
|---|---:|---:|---:|---:|---:|
| stage2_smalltrain_core | 0.000839 | 0.024232 | 0.024232 | 207615754 | 3025154 |
| stage2_smalltrain_core_plus_burst | 0.000237 | 0.014631 | 0.014631 | 207615754 | 3025154 |

## Mandatory Answers
- core_only_stable: True
- core_plus_burst_better_than_core_only: True
- stage1_frozen_boundary_kept_correct: True
- smalltrain_status: smalltrain_successful
- next_step_choice: continue_stage2_training
