# TRACEWM Stage1-v2 Perf Results

- generated_at_utc: 2026-04-08T09:55:41.075758+00:00
- primary_bottleneck: ambiguous_compute_bound
- attribution_primary_source: prototype_220m
- selected_gpu_id: 6

## Evidence
- preflight_pass: True (source=/home/chen034/workspace/stwm/reports/stage1_v2_220m_preflight_20260408.json)
- attribution_primary_source: prototype_220m (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_confirmation_20260408.json)
- selected_gpu_id: 6 (source=/home/chen034/workspace/stwm/reports/stage1_v2_gpu_selection_audit_20260408.json)
- primary_step_time_mean: 0.46955590540892445 (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_confirmation_20260408.json)
- primary_compute_ratio: 0.8785082973954467 (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_confirmation_20260408.json)
- primary_wait_ratio: 0.008996129060661797 (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_confirmation_20260408.json)
- primary_h2d_ratio: 0.0028638853079308796 (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_confirmation_20260408.json)
- selected_gpu_avg_util: 43.125 (source=/home/chen034/workspace/stwm/reports/stage1_v2_gpu_telemetry_20260408.json)
- selected_gpu_avg_mem_util: 3.5 (source=/home/chen034/workspace/stwm/reports/stage1_v2_gpu_telemetry_20260408.json)
- aux_reference_compute_ratio: 0.8068322315072047 (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_debug_small_20260408.json)
- dataloader_best_batches_per_sec: 771.8251894463397 (source=/home/chen034/workspace/stwm/reports/stage1_v2_dataloader_profile_20260408.json)

## Top 5 Actions
- 1. Use prototype_220m timing as primary attribution basis and treat debug_small only as auxiliary (expected_gain=high)
- 2. Gate gpu_bound on both high compute ratio and high selected GPU telemetry utilization (expected_gain=high)
- 3. Use dataloader best_config from profile as recommended runtime default (expected_gain=medium-high)
- 4. Keep pin_memory=True with non_blocking H2D copies in single-GPU runs (expected_gain=medium)
- 5. Treat worker-side dataloader timing for num_workers>0 as unavailable and avoid false CPU/IO claims (expected_gain=medium)

## Recommended GPU Policy
- mode: single_gpu_only
- selected_gpu_id: 6
- selection_rule:
  - avg_gpu_util lowest
  - avg_mem_util lowest
  - active_compute_process_count lowest
  - free_mem highest
