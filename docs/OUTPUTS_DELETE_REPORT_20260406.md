# Outputs Delete Report 20260406

- Generated: 2026-04-06 19:47:54

## Preconditions

- audit_generated: PASS
- classification_generated: PASS
- archives_generated_and_validated: PASS
- delete_candidate_list_present: PASS

## Deleted Paths (DELETE_CANDIDATE Only)

| Path | Size |
|---|---:|
| outputs/background_jobs/stwm_delayed_router_mainline_seed42_watch_v1.pid | 8B |
| outputs/background_jobs/stwm_qstr_mainline_seed42_submit_v1.pid | 8B |
| outputs/background_jobs/stwm_qstr_mainline_seed42_watch_v1.pid | 8B |
| outputs/background_jobs/stwm_qtsa_mainline_seed42_submit_v1.pid | 8B |
| outputs/background_jobs/stwm_qtsa_mainline_seed42_watch_v1.pid | 8B |
| outputs/background_jobs/stwm_two_path_residual_promotion_decision_v1.pid | 8B |
| outputs/background_jobs/stwm_two_path_residual_seed123_submit_v1.pid | 8B |
| outputs/background_jobs/stwm_two_path_residual_seed123_watch_v1.pid | 8B |

## Disk Usage Change

- outputs before: 1.02TB (1126947188945 bytes)
- outputs after: 1.02TB (1126947188881 bytes)
- freed: 64B (64 bytes)

## Kept Paths

- outputs/baselines
- outputs/eval/stwm_v4_2_completed_protocol_eval_20260403
- outputs/eval/stwm_v4_2_completed_protocol_eval_real_evalonly_20260403
- outputs/monitoring/stwm_hourly_push
- outputs/queue/stwm_protocol_v2
- outputs/queue/stwm_protocol_v2_frontend_default_v1
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_semteacher_mainline_seed42_v1
- outputs/training/stwm_v4_2_real_1b
- outputs/training/stwm_v4_2_real_220m

## Archived Paths

- outputs/audits/stwm_v4_2_phase01_20260401_155536
- outputs/audits/stwm_v4_2_phase01_20260401_155639
- outputs/audits/stwm_v4_2_phase01_20260401_161909
- outputs/background_jobs
- outputs/benchmarks/frontend_cache_ab_v1
- outputs/benchmarks/frontend_cache_ab_v1_gpu3
- outputs/benchmarks/stwm_frontend_cache_confirm_v1
- outputs/eval/detached_protocol_v4_2_smoke
- outputs/eval/stwm_v4_2_detached_protocol_eval_20260403
- outputs/queue/stwm_1b
- outputs/queue/stwm_1b_real
- outputs/queue/stwm_gpu
- outputs/queue/stwm_v4_2_real_matrix
- outputs/smoke_tests/cutie_vspw
- outputs/smoke_tests/deva_vspw
- outputs/smoke_tests/sam2_vspw
- outputs/smoke_tests/xmem_vspw
- outputs/smoke_tests/yolo_world_ultralytics
- outputs/training/stwm_frontend_cache_pilot_v1
- outputs/training/stwm_query_gradient_audit_fix_smoke_v1
- outputs/training/stwm_v4_2_1b_confirmation_staged
- outputs/training/stwm_v4_2_1b_real_confirmation
- outputs/training/stwm_v4_2_1b_smoke
- outputs/training/stwm_v4_2_220m_protocol_diag_v1
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_delayed_router_mainline_seed42_v1
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_qstr_mainline_seed42_v1
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_qtsa_mainline_seed42_v1
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replacement_seed42_v1
- outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1
- outputs/training/stwm_v4_2_220m_protocol_object_bias_diag_v1
- outputs/training/stwm_v4_2_identity_rescue_round
- outputs/training/stwm_v4_2_minival_multiseed
- outputs/training/stwm_v4_2_minival_seed42
- outputs/training/stwm_v4_2_protocol_repair
- outputs/training/stwm_v4_2_smoke
- outputs/training/stwm_v4_2_state_identifiability
- outputs/training/week2_ablations
- outputs/training/week2_minival
- outputs/training/week2_minival_sanity
- outputs/training/week2_minival_v2
- outputs/training/week2_minival_v2_1
- outputs/training/week2_minival_v2_2
- outputs/training/week2_minival_v2_3
- outputs/visualizations/stwm_v4_2_1b_confirmation_demo
- outputs/visualizations/stwm_v4_2_1b_confirmation_multiseed_casebook
- outputs/visualizations/stwm_v4_2_1b_confirmation_state_identifiability_figures
- outputs/visualizations/stwm_v4_2_final_paper_figures
- outputs/visualizations/stwm_v4_2_minival_seed42
- outputs/visualizations/stwm_v4_2_multiseed_casebook
- outputs/visualizations/stwm_v4_2_real_1b_demo
- outputs/visualizations/stwm_v4_2_real_1b_multiseed_casebook
- outputs/visualizations/stwm_v4_2_real_1b_state_identifiability_figures
- outputs/visualizations/stwm_v4_2_state_identifiability_figures
- outputs/visualizations/week2_figures
- outputs/visualizations/week2_figures_sanity
- outputs/visualizations/week2_figures_v2
- outputs/visualizations/week2_figures_v2_1
- outputs/visualizations/week2_figures_v2_2
- outputs/visualizations/week2_figures_v2_3

## Archived Moved Out Of Place

- none (archive packages created; original ARCHIVE paths retained in-place in this safe pass)

## Notes

- This run archived ARCHIVE class into /archives but did not remove ARCHIVE items from original outputs tree; only DELETE_CANDIDATE files were removed.
