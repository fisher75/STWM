# Stage1-v2 Recommended Runtime

- generated_at_utc: 2026-04-08T09:55:16.085148+00:00
- selected_gpu_policy: {"mode": "single_gpu_only", "selection_rule": ["avg_gpu_util lowest", "avg_mem_util lowest", "active_compute_process_count lowest", "free_mem highest"], "window": {"sample_count": 12, "sample_interval_sec": 2.0}, "memory_filter": {"required_mem_gb": 40.0, "safety_margin_gb": 8.0}, "selected_gpu_id": 6}
- required_mem_gb: 40.0
- safety_margin_gb: 8.0
- recommended_num_workers: 8
- recommended_pin_memory: True
- recommended_persistent_workers: True
- recommended_prefetch_factor: 4
- recommended_batch_size_debug_small: 2
- recommended_batch_size_prototype_220m: 2
- single_gpu_only: True

Recommended defaults are runtime settings only and do not change scientific logic.
