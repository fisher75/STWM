# STWM Frontend Cache A/B Report V1

Date: 2026-04-03
Status: Phase 3 completed

## 1) Experiment Setup

Fixed slice and fixed configuration were used for A/B.

Shared setup:

1. seed: 42
2. manifest: `data/cache/frontend_cache_pilot_v1/pilot_manifest.json`
3. model preset: `prototype_220m_v4_2`
4. micro-batch / grad-accum: 2 / 8
5. workers/prefetch/pin/persistent: 12 / 2 / on / on
6. precision/checkpointing: bf16 + activation checkpointing

Groups:

1. Raw: `--data-mode raw`
2. Frontend cache: `--data-mode frontend_cache`

## 2) Primary Runtime Results (12-step A/B)

Source summaries:

1. `outputs/training/stwm_frontend_cache_pilot_v1/raw_seed42/mini_val_summary.json`
2. `outputs/training/stwm_frontend_cache_pilot_v1/frontend_cache_seed42/mini_val_summary.json`

Metrics:

1. `step_time_p50_s`
   - raw: 23.3957
   - frontend_cache: 2.5592
   - ratio: 0.1094
   - reduction: 89.06%

2. `step_time_p95_s`
   - raw: 36.2873
   - frontend_cache: 11.3252

3. `data_time_p50_s`
   - raw: 20.7649
   - frontend_cache: 0.0294
   - ratio: 0.00142
   - reduction: 99.86%

4. `data_wait_ratio_p50`
   - raw: 0.8894
   - frontend_cache: 0.0182
   - absolute drop: 0.8712

5. `gpu_peak_memory_gb_max`
   - raw: 3.8831
   - frontend_cache: 3.8831
   - no regression

## 3) Process CPU/GPU Monitoring (6-step A/B)

Source monitor report:

- `outputs/benchmarks/frontend_cache_ab_v1_gpu3/ab_report.json`

Monitoring summary:

1. stability
   - raw exit: success
   - frontend_cache exit: success

2. CPU mean (%process)
   - raw: 90.91
   - frontend_cache: 87.32
   - delta: -3.59

3. process GPU SM util mean (% via pmon)
   - raw: 2.44
   - frontend_cache: 2.31
   - delta: -0.13

4. process used GPU memory mean (MiB)
   - raw: 3903.63
   - frontend_cache: 3056.23
   - delta: -847.40

Interpretation:

1. The gain is overwhelmingly from data-path reduction.
2. GPU SM mean stayed low in this shared environment, consistent with data bottleneck and co-tenant effects.

## 4) Consolidated JSON

Consolidated decision input:

- `reports/stwm_frontend_cache_ab_report_v1.json`

This file aggregates 12-step runtime deltas + monitored CPU/GPU deltas and computes Phase 4 go/no-go criteria flags.
