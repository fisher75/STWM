# Stage1 / Stage2 Qualitative Pack V7

- generated_at_utc: 2026-04-15T19:30:48.418356+00:00
- stage1_case_count: 9
- stage2_case_count: 16
- ready_for_paper_figure_selection: True

| case_id | dataset | tags | interpretation |
|---|---|---|---|
| stage2_mechanism_summary_v7 | VSPW+VIPSeg | mechanism-summary | Compact method table for paper figure selection and caption drafting. |
| stage2_calibration_clear_win_v7 | VIPSeg | calibration-only,state-identifiability-success,occlusion_reappearance,crossing_ambiguity,small_object,appearance_change | Figure candidate for the main Stage2 success story. |
| stage2_calibration_fail_v7 | BURST | calibration-only-fail,state-identifiability-failure,occlusion_reappearance,long_gap_persistence,small_object | Failure case kept to prevent one-sided qualitative selection. |
| stage2_legacysem_win_v7 | VIPSeg | legacysem-win,occlusion_reappearance,crossing_ambiguity,small_object,appearance_change | Use this when discussing residual failure regimes. |
| stage2_cropenc_win_v7 | VIPSeg | cropenc-win,occlusion_reappearance,crossing_ambiguity,small_object,appearance_change | Shows calibration-only is not universally dominant. |
| stage2_noalign_failure_v7 | VIPSeg | noalign-failure,occlusion_reappearance,crossing_ambiguity,small_object,appearance_change | Mechanism figure candidate for semantic alignment being load-bearing. |
| stage2_densegate_failure_v7 | VIPSeg | densegate-failure,occlusion_reappearance,crossing_ambiguity,small_object,appearance_change | Mechanism figure candidate for sparse gating. |
| stage2_nodelay_failure_v7 | VIPSeg | nodelay-failure,occlusion_reappearance,crossing_ambiguity,small_object,appearance_change | Mechanism figure candidate for delayed schedule. |
| stage2_state_identifiability_success_v7 | VIPSeg | state-identifiability-success,occlusion_reappearance,crossing_ambiguity,small_object,appearance_change | Use directly in protocol contribution figures. |
| stage2_state_identifiability_failure_v7 | BURST | state-identifiability-failure,occlusion_reappearance,long_gap_persistence,small_object | Use this to bound claims and show protocol difficulty. |
| stage2_calibration_clear_win_v6 | VSPW+VIPSeg | calibration-only,semantic-hard | Use this case to illustrate selective readout-side semantic calibration. |
| stage2_legacysem_win_v6 | VSPW+VIPSeg | legacysem-win,baseline-comparison | Shows where static/legacy semantics can remain competitive. |
| stage2_cropenc_win_v6 | VSPW+VIPSeg | cropenc-win,baseline-comparison | Shows where plain cropenc remains close despite weaker final evidence. |
| stage2_noalign_failure_v6 | VSPW+VIPSeg | noalign-failure,mechanism-ablation | No-align degradation supports the mechanism ablation story. |
| stage2_densegate_failure_v6 | VSPW+VIPSeg | densegate-failure,mechanism-ablation | Dense gating removes selectivity and is kept as a failure/control case. |
| stage2_nodelay_failure_v6 | VSPW+VIPSeg | nodelay-failure,mechanism-ablation | Immediate auxiliary intervention is a controlled failure condition. |
