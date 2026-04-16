# Stage1 / Stage2 Qualitative Pack V9

- generated_at_utc: 2026-04-16T17:30:26.206903+00:00
- stage1_case_count: 9
- stage2_case_count: 11
- ready_for_paper_figure_selection: True
- ready_for_oral_backup_figure_selection: True

| case_id | dataset | tags | interpretation |
|---|---|---|---|
| stage2_calibration_clear_win_v9 | BURST | calibration-only,state-identifiability-success,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Primary Stage2 success figure candidate. |
| stage2_calibration_fail_v9 | BURST | calibration-only-fail,state-identifiability-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Use to bound claims and highlight remaining hard regimes. |
| stage2_legacysem_win_v9 | BURST | legacysem-win,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Residual failure regime. |
| stage2_cropenc_win_v9 | BURST | cropenc-win,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Residual failure regime. |
| stage2_noalign_failure_v9 | BURST | noalign-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Supports alignment being load-bearing when mechanism evidence closes. |
| stage2_densegate_failure_v9 | BURST | densegate-failure,crossing_ambiguity,small_object,appearance_change | Supports sparse gating selectivity. |
| stage2_nodelay_failure_v9 | BURST | nodelay-failure,crossing_ambiguity,small_object,appearance_change | Supports delayed auxiliary schedule. |
| stage2_anomaly_scope_v9 | BURST | anomaly-check,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Anomaly status: {"stage2_calonly_noalign_seed123_ablate_confirm_v3_20260416": {"confirmed": false, "scope": "seed123 noalign"}, "stage2_calonly_nodelay_seed42_ablate_confirm_v3_20260416": {"confirmed": true, "scope": "seed42 nodelay"}, "stage2_calonly_densegate_seed123_ablate_confirm_v3_20260416": {"confirmed": false, "scope": "seed123 densegate"}} |
| stage2_state_identifiability_success_v9 | BURST | state-identifiability-success,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Direct protocol contribution figure candidate. |
| stage2_state_identifiability_failure_v9 | BURST | state-identifiability-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Hard negative for oral backup and paper balance. |
| stage2_local_temporal_branch_v9 | BURST | local-temporal-branch,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Local temporal pilot summary: {"run_name": "stage2_localtemp_w5_seed123_20260416", "seed": 123, "local_temporal_window": 5, "endpoint_l2": 0.0005282461077264041, "semantic_hard_sidecar_score": 0.00015759576882164064} |
