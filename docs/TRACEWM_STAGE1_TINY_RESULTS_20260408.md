# TraceWM Stage 1 Tiny Results (2026-04-08)

- generated_at_utc: 2026-04-07T17:10:14.740609+00:00
- task: trace_only_future_trace_state_generation
- device: cuda
- train_datasets: ['pointodyssey', 'kubric']
- teacher_forced_supported: True
- free_rollout_supported: True
- steps: 40
- batch_size: 2
- train_samples: 36
- val_samples: 14

## Metrics

- train_teacher_mse_mean: 0.022895
- train_free_mse_mean: 0.030431
- train_total_loss_mean: 0.038111
- val_teacher_mse: 0.004693
- val_free_mse: 0.007944

- checkpoint_path: /home/chen034/workspace/stwm/outputs/training/tracewm_stage1_tiny_20260408/tiny_trace_model.pt
- summary_json: /home/chen034/workspace/stwm/reports/tracewm_stage1_tiny_summary_20260408.json
