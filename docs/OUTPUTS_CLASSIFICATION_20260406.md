# Outputs Classification 20260406

- Generated: 2026-04-06 19:11:25
- Policy: KEEP minimal direct-value artifacts, ARCHIVE broad historical evidence, DELETE_CANDIDATE only obvious ephemeral files

## Summary

- KEEP: 9 paths
- ARCHIVE: 60 paths
- DELETE_CANDIDATE: 8 paths

## KEEP

| Path | Size | Reason |
|---|---:|---|
| outputs/baselines | 0B | Default evaluator output location in code; keep directory skeleton. |
| outputs/eval/stwm_v4_2_completed_protocol_eval_20260403 | 1.81KB | Completed protocol eval artifact; small and still directly referable. |
| outputs/eval/stwm_v4_2_completed_protocol_eval_real_evalonly_20260403 | 7.44KB | Completed real eval-only artifact; small and still directly referable. |
| outputs/monitoring/stwm_hourly_push | 6.65KB | Tiny active monitoring state and report trail; keep in-place. |
| outputs/queue/stwm_protocol_v2 | 408.50KB | Current queue namespace referenced by scripts; keep in-place. |
| outputs/queue/stwm_protocol_v2_frontend_default_v1 | 913.63KB | Current queue namespace referenced by scripts/docs; keep in-place. |
| outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_semteacher_mainline_seed42_v1 | 20.89GB | Current semteacher mainline evidence; keep in-place for active comparison and quick checkpoint access. |
| outputs/training/stwm_v4_2_real_1b | 148.32GB | Direct protocol/evaluator regression reference used repeatedly in docs; keep in-place. |
| outputs/training/stwm_v4_2_real_220m | 37.14GB | Direct protocol/evaluator regression reference used repeatedly in docs; keep in-place. |

## ARCHIVE

| Path | Size | Reason |
|---|---:|---|
| outputs/audits/stwm_v4_2_phase01_20260401_155536 | 2.64KB | Freeze-period audit trace; preserve as archive rather than in-place bulk storage. |
| outputs/audits/stwm_v4_2_phase01_20260401_155639 | 54.91GB | Freeze-period audit trace; preserve as archive rather than in-place bulk storage. |
| outputs/audits/stwm_v4_2_phase01_20260401_161909 | 54.91GB | Freeze-period audit trace; preserve as archive rather than in-place bulk storage. |
| outputs/background_jobs | 116.08KB | Historical watcher logs/status/pid traces; archive first, then delete only explicit ephemeral files. |
| outputs/benchmarks/frontend_cache_ab_v1 | 8.87KB | Older benchmark/raw cache evidence; preserve in archive. |
| outputs/benchmarks/frontend_cache_ab_v1_gpu3 | 9.28GB | Older benchmark/raw cache evidence; preserve in archive. |
| outputs/benchmarks/stwm_frontend_cache_confirm_v1 | 9.28GB | Older benchmark/raw cache evidence; preserve in archive. |
| outputs/eval/detached_protocol_v4_2_smoke | 2.94KB | Detached/smoke eval artifact from older flow; archive for provenance. |
| outputs/eval/stwm_v4_2_detached_protocol_eval_20260403 | 59.42KB | Detached/smoke eval artifact from older flow; archive for provenance. |
| outputs/queue/stwm_1b | 23.02KB | Old queue traces/backups/parked states; archive for auditability. |
| outputs/queue/stwm_1b_real | 11.32KB | Old queue traces/backups/parked states; archive for auditability. |
| outputs/queue/stwm_gpu | 0B | Old queue traces/backups/parked states; archive for auditability. |
| outputs/queue/stwm_v4_2_real_matrix | 161.12KB | Old queue traces/backups/parked states; archive for auditability. |
| outputs/smoke_tests/cutie_vspw | 408.63KB | Baseline smoke evidence referenced in docs; archive, not direct delete. |
| outputs/smoke_tests/deva_vspw | 7.56MB | Baseline smoke evidence referenced in docs; archive, not direct delete. |
| outputs/smoke_tests/sam2_vspw | 661.27KB | Baseline smoke evidence referenced in docs; archive, not direct delete. |
| outputs/smoke_tests/xmem_vspw | 400.15KB | Baseline smoke evidence referenced in docs; archive, not direct delete. |
| outputs/smoke_tests/yolo_world_ultralytics | 10.92MB | Baseline smoke evidence referenced in docs; archive, not direct delete. |
| outputs/training/stwm_frontend_cache_pilot_v1 | 9.28GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_query_gradient_audit_fix_smoke_v1 | 7.27KB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_1b_confirmation_staged | 34.22GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_1b_real_confirmation | 45.63GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_1b_smoke | 14.03KB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_diag_v1 | 27.85GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1 | 27.85GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_delayed_router_mainline_seed42_v1 | 20.89GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_qstr_mainline_seed42_v1 | 20.89GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_qtsa_mainline_seed42_v1 | 20.89GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1 | 27.85GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1 | 41.78GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_220m_protocol_object_bias_diag_v1 | 41.78GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_identity_rescue_round | 9.28GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_minival_multiseed | 6.96GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_minival_seed42 | 2.32GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_protocol_repair | 6.19GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_smoke | 36.18KB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/stwm_v4_2_state_identifiability | 848.48KB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/week2_ablations | 3.21KB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/week2_minival | 40.63GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/week2_minival_sanity | 5.08GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/week2_minival_v2 | 40.64GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/week2_minival_v2_1 | 101.59GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/week2_minival_v2_2 | 91.43GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/training/week2_minival_v2_3 | 91.43GB | Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup. |
| outputs/visualizations/stwm_v4_2_1b_confirmation_demo | 13.63MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_1b_confirmation_multiseed_casebook | 21.47MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_1b_confirmation_state_identifiability_figures | 26.98MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_final_paper_figures | 24.48MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_minival_seed42 | 35.69MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_multiseed_casebook | 32.55MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_real_1b_demo | 13.19MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_real_1b_multiseed_casebook | 28.19MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_real_1b_state_identifiability_figures | 25.84MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/stwm_v4_2_state_identifiability_figures | 30.51MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/week2_figures | 14.17MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/week2_figures_sanity | 0B | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/week2_figures_v2 | 22.25MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/week2_figures_v2_1 | 36.63MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/week2_figures_v2_2 | 14.38MB | Legacy visualization assets; retain via archive package. |
| outputs/visualizations/week2_figures_v2_3 | 14.38MB | Legacy visualization assets; retain via archive package. |

## DELETE_CANDIDATE

| Path | Size | Reason |
|---|---:|---|
| outputs/background_jobs/stwm_delayed_router_mainline_seed42_watch_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |
| outputs/background_jobs/stwm_qstr_mainline_seed42_submit_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |
| outputs/background_jobs/stwm_qstr_mainline_seed42_watch_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |
| outputs/background_jobs/stwm_qtsa_mainline_seed42_submit_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |
| outputs/background_jobs/stwm_qtsa_mainline_seed42_watch_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |
| outputs/background_jobs/stwm_two_path_residual_promotion_decision_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |
| outputs/background_jobs/stwm_two_path_residual_seed123_submit_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |
| outputs/background_jobs/stwm_two_path_residual_seed123_watch_v1.pid | 8B | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |

## Archive Groups

### training
- Path count: 26
- List file: reports/archive_lists_20260406/training_paths.txt

### eval
- Path count: 2
- List file: reports/archive_lists_20260406/eval_paths.txt

### visualizations
- Path count: 16
- List file: reports/archive_lists_20260406/visualizations_paths.txt

### queue
- Path count: 4
- List file: reports/archive_lists_20260406/queue_paths.txt

### audits
- Path count: 3
- List file: reports/archive_lists_20260406/audits_paths.txt

### benchmarks
- Path count: 3
- List file: reports/archive_lists_20260406/benchmarks_paths.txt

### smoke_tests
- Path count: 5
- List file: reports/archive_lists_20260406/smoke_tests_paths.txt

### background_jobs
- Path count: 1
- List file: reports/archive_lists_20260406/background_jobs_paths.txt
