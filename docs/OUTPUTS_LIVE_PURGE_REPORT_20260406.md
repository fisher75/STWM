# Outputs Live Purge Report 20260406

- Generated: 2026-04-06 20:26:29
- outputs before: 1.02TB (1126947188881 bytes)
- outputs after: 297B (297 bytes)
- freed: 1.02TB (1126947188584 bytes)

## Per Top-Level Delta

| Dir | Before | After |
|---|---:|---:|
| outputs/audits | 109.82GB | 0B |
| outputs/background_jobs | 116.02KB | 0B |
| outputs/baselines | 0B | 0B |
| outputs/benchmarks | 18.56GB | 0B |
| outputs/eval | 71.60KB | 0B |
| outputs/monitoring | 6.65KB | 0B |
| outputs/queue | 1.48MB | 0B |
| outputs/smoke_tests | 19.92MB | 0B |
| outputs/training | 920.81GB | 0B |
| outputs/visualizations | 354.33MB | 0B |

## Archives Used

- /home/chen034/workspace/stwm/archives/outputs_training_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_audits_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_benchmarks_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_visualizations_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_smoke_tests_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_queue_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_background_jobs_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_eval_archive_20260406.tar.zst
- /home/chen034/workspace/stwm/archives/outputs_keep_reclassified_archive_20260406.tar.zst

## Reclassified KEEP Supplement Archive Coverage

- outputs/training/stwm_v4_2_real_1b
- outputs/training/stwm_v4_2_real_220m
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_semteacher_mainline_seed42_v1
- outputs/eval/stwm_v4_2_completed_protocol_eval_20260403
- outputs/eval/stwm_v4_2_completed_protocol_eval_real_evalonly_20260403
- outputs/queue/stwm_protocol_v2
- outputs/queue/stwm_protocol_v2_frontend_default_v1
- outputs/monitoring/stwm_hourly_push
- outputs/baselines

## Skeleton Status

| Dir | Exists | Empty | Mode |
|---|---|---|---|
| outputs/audits | True | True | 0o775 |
| outputs/background_jobs | True | True | 0o775 |
| outputs/baselines | True | True | 0o775 |
| outputs/benchmarks | True | True | 0o775 |
| outputs/eval | True | True | 0o775 |
| outputs/monitoring | True | True | 0o775 |
| outputs/queue | True | True | 0o775 |
| outputs/smoke_tests | True | True | 0o775 |
| outputs/training | True | True | 0o775 |
| outputs/visualizations | True | True | 0o775 |

## Notes

- live workspace 已清空旧 STWM/V4.2 产物
- 历史内容仅保留在 docs/reports/archives 中
- 新主线可以从干净 outputs 重新开始
