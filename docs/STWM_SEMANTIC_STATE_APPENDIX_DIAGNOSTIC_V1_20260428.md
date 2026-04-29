# STWM Semantic-State Appendix Diagnostic V1

- branch_purpose: `Explore explicit FutureSemanticTraceState as a world-state output beyond trace_belief_assoc.`
- internal_reappearance_signal: `Internal event-level reappearance probes showed some positive signal after target/mask repair and head-only/joint readout refinement.`
- external_query_bridge: `{'future_candidate_used_as_input': False, 'stage2_val_fallback_used': False, 'old_association_report_used': False, 'full_model_forward_executed': True, 'full_free_rollout_executed': True}`
- v7_frozen_clip_measurement_result: `{'posterior_v7_heldout_top1_AP_AUROC': [0.20441988950276244, 0.10548108231554292, 0.613760084394463], 'posterior_v7_no_predicted_state_heldout_top1_AP_AUROC': [0.2596685082872928, 0.11551220090476644, 0.6613086100836055], 'appearance_frozen_heldout_top1_AP_AUROC': [0.44751381215469616, 0.2047032542889563, 0.736440870903972], 'predicted_state_load_bearing_robust': False}`
- why_not_main_claim: `External V7 shows predicted FutureSemanticTraceState is not robustly load-bearing; appearance-only frozen measurement dominates.`
- semantic_state_branch_status: `appendix_diagnostic`

## Lessons
- world-state output/export is feasible
- external candidate measurement is dominated by appearance
- future work needs stronger state-measurement alignment or direct feature prediction
- negative V7 result must not be hidden

## Forbidden
- do not present semantic-state branch as main success
- do not hide negative V7 result
- do not package VLM appearance-only as STWM world-model evidence
