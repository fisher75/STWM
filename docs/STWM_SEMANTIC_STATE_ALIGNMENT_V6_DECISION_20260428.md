# STWM Semantic-State Alignment V6 Decision

- v5_signal_significant: `unclear`
- listwise_signal_positive: `True`
- predicted_state_load_bearing_robust: `unclear`
- posterior_v6_improves_over_no_predicted_state: `True`
- posterior_v6_improves_over_appearance: `False`
- paper_world_model_claimable: `False`
- semantic_state_branch_status: `appendix_diagnostic`
- recommended_next_step_choice: `improve_with_frozen_vlm_features`

V6 listwise alignment produces positive heldout gains and AUROC CI excludes zero versus no_predicted_state, but AP/top1 CIs still cross zero and appearance-only remains much stronger; keep as appendix diagnostic and improve measurement features before stronger paper claims.
