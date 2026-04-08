# TRACEWM Stage1-v2 Perf Results

- generated_at_utc: 2026-04-08T09:33:27.533344+00:00
- primary_bottleneck: gpu_bound
- selected_gpu_id: 6

## Evidence
- preflight_pass: True (source=/home/chen034/workspace/stwm/reports/stage1_v2_220m_preflight_20260408.json)
- selected_gpu_id: 6 (source=/home/chen034/workspace/stwm/reports/stage1_v2_gpu_selection_audit_20260408.json)
- debug_step_time_mean: 0.03270272992473717 (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_debug_small_20260408.json)
- debug_batch_wait_mean: 0.0006356241375518342 (source=/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_debug_small_20260408.json)
- dataloader_best_batches_per_sec: 1400.8782674484087 (source=/home/chen034/workspace/stwm/reports/stage1_v2_dataloader_profile_20260408.json)
- selected_gpu_avg_util: 30.833333333333332 (source=/home/chen034/workspace/stwm/reports/stage1_v2_gpu_telemetry_20260408.json)
- selected_gpu_avg_mem_util: 1.5 (source=/home/chen034/workspace/stwm/reports/stage1_v2_gpu_telemetry_20260408.json)

## Top 5 Actions
- 1. Use dataloader best_config from profile as default perf config (expected_gain=high)
- 2. Keep pin_memory=True and non_blocking H2D copies in single-GPU runs (expected_gain=high)
- 3. Pin single GPU by averaged window sampling and lease guard (expected_gain=medium-high)
- 4. Tune num_workers and prefetch_factor jointly for stable batches/sec (expected_gain=medium)
- 5. Use profiler traces to remove top-k expensive kernels or sync points (expected_gain=medium)

## Recommended GPU Policy
- mode: single_gpu_only
- selected_gpu_id: 6
- selection_rule:
  - avg_gpu_util lowest
  - avg_mem_util lowest
  - active_compute_process_count lowest
  - free_mem highest
