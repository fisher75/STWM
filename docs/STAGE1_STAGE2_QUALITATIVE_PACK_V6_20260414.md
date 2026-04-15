# Stage1 / Stage2 Qualitative Pack V6

- generated_at_utc: 2026-04-15T16:15:36.523583+00:00
- ready_for_human_figure_selection: True
- stage1_cases: 9
- stage2_cases: 15
- utility_eval_improved: True

| case_id | tags | interpretation |
|---|---|---|
| stage2_calibration_clear_win_v6 | calibration-only,semantic-hard | Use this case to illustrate selective readout-side semantic calibration. |
| stage2_legacysem_win_v6 | legacysem-win,baseline-comparison | Shows where static/legacy semantics can remain competitive. |
| stage2_cropenc_win_v6 | cropenc-win,baseline-comparison | Shows where plain cropenc remains close despite weaker final evidence. |
| stage2_noalign_failure_v6 | noalign-failure,mechanism-ablation | No-align degradation supports the mechanism ablation story. |
| stage2_densegate_failure_v6 | densegate-failure,mechanism-ablation | Dense gating removes selectivity and is kept as a failure/control case. |
| stage2_nodelay_failure_v6 | nodelay-failure,mechanism-ablation | Immediate auxiliary intervention is a controlled failure condition. |
| stage2_longrun_confirmation_v6 | longrun,confirmation | Use this case to decide whether longrun is a confirmation or no-improvement story. |
| stage2_query_utility_success_v6 | query-utility,success | Use this case to inspect whether lower future-state error translates into a more useful queryable state. |
| stage2_query_utility_failure_v6 | query-utility,failure | This prevents over-claiming utility and should be reviewed before figure selection. |
| stage2v5_000 | occlusion_reappearance,crossing_or_interaction_ambiguity,small_object_or_low_area | alignment/calibration branch appears genuinely helpful here |
| stage2v5_001 | appearance_change_or_semantic_shift | alignment/calibration branch appears genuinely helpful here |
| stage2v5_002 | occlusion_reappearance,crossing_or_interaction_ambiguity,small_object_or_low_area,appearance_change_or_semantic_shift | alignment/calibration branch appears genuinely helpful here |
| stage2v5_003 | occlusion_reappearance,crossing_or_interaction_ambiguity,small_object_or_low_area,appearance_change_or_semantic_shift | alignment/calibration branch appears genuinely helpful here |
| stage2v5_004 | occlusion_reappearance,crossing_or_interaction_ambiguity,small_object_or_low_area | hand-crafted semantic stats remain stronger on this failure mode |
| stage2v5_005 | occlusion_reappearance,crossing_or_interaction_ambiguity,small_object_or_low_area,appearance_change_or_semantic_shift | hand-crafted semantic stats remain stronger on this failure mode |
