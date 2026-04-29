# STWM Paper Asset Consolidation V1 Source Audit

- official_method_name: `STWM / TUSB-v3.1 + trace_belief_assoc`
- official_stage1_status: `frozen trace-first future-state backbone`
- official_stage2_status: `TUSB-v3.1 identity-bound semantic trace units`
- official_readout: `trace_belief_assoc`
- primary_results_available: `{'official_method_freeze': True, 'main_comparison': True, 'readout_ablation': True, 'false_confuser': True, 'reacquisition_utility': True, 'planning_lite_risk': True, 'counterfactual': True, 'external_boundary': True}`
- missing_assets: `[]`
- semantic_state_branch_status: `appendix_diagnostic`

## Allowed Claims
- STWM belief readout improves over calibration-only, cropenc, and legacysem under the official frozen setting.
- Trace-conditioned belief reduces false-confuser errors.
- STWM improves occlusion-aware reacquisition utility.
- STWM improves planning-lite risk scoring / false-safe reduction under synthetic path corridors.
- Counterfactual interventions on trace belief affect association, reacquisition, and risk outputs.
- STWM remains trace-first and does not rely on teacher-only retrieval.

## Forbidden Claims
- full RGB video generation world model
- closed-loop autonomous driving planner
- appearance-change solved
- universal OOD dominance
- oral/spotlight guaranteed
- Stage1 retrained in final method
- feedback adapter works
- semantic-state branch is main contribution
- STWM beats SAM2 overall
- external overall SOTA
- SAM2/CoTracker plugin framing
