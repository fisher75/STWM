# STWM Paper Readiness Decision V1

- ready_for_paper_writing: `True`
- ready_for_method_more_exploration: `False`
- target_venue_recommendation: `CVPR/AAAI main hopeful; oral/spotlight not guaranteed`
- strongest_claim: STWM / TUSB-v3.1 + trace_belief_assoc improves future identity association and continuity-heavy utility over calibration-only, cropenc, and legacysem under the official frozen setting.
- weakest_claim: Explicit FutureSemanticTraceState is not robustly load-bearing on external V7 and should remain appendix diagnostic/future work.
- main_risk: External SAM2/CoTracker are stronger overall; paper must frame STWM as trace-conditioned semantic belief world model with bounded utility rather than external overall tracker SOTA.
- rebuttal_risk: Reviewers may ask whether STWM is just appearance matching or teacher retrieval; answer with trace_belief_assoc ablations, false-confuser, reacquisition, and counterfactual trace evidence, while disclosing external baseline boundaries.
- next_step_choice: `start_overleaf_draft`
