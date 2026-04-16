# Stage1 / Stage2 Qualitative Pack V8

- generated_at_utc: 2026-04-16T08:21:40.113992+00:00
- stage1_case_count: 9
- stage2_case_count: 10
- ready_for_paper_figure_selection: True
- ready_for_oral_backup_figure_selection: True

| case_id | dataset | tags | interpretation |
|---|---|---|---|
| stage2_calibration_clear_win_v8 | BURST | calibration-only,state-identifiability-success,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Primary Stage2 success figure candidate. |
| stage2_calibration_fail_v8 | BURST | calibration-only-fail,state-identifiability-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Use to bound claims and highlight remaining hard regimes. |
| stage2_legacysem_win_v8 | BURST | legacysem-win,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Residual failure regime. |
| stage2_cropenc_win_v8 | BURST | cropenc-win,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Residual failure regime. |
| stage2_noalign_failure_v8 | BURST | noalign-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Supports alignment being load-bearing when mechanism evidence closes. |
| stage2_densegate_failure_v8 | BURST | densegate-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Supports sparse gating selectivity. |
| stage2_nodelay_failure_v8 | BURST | nodelay-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Supports delayed auxiliary schedule. |
| stage2_anomaly_scope_v8 | BURST | anomaly-check,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Anomaly status: {"stage2_calonly_noalign_seed123_ablate_confirm_20260416": {"confirmed": false, "scope": "seed123 noalign"}, "stage2_calonly_nodelay_seed42_ablate_confirm_20260416": {"confirmed": true, "scope": "seed42 nodelay"}} |
| stage2_state_identifiability_success_v8 | BURST | state-identifiability-success,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object,appearance_change | Direct protocol contribution figure candidate. |
| stage2_state_identifiability_failure_v8 | BURST | state-identifiability-failure,occlusion_reappearance,long_gap_persistence,crossing_ambiguity,small_object | Hard negative for oral backup and paper balance. |
