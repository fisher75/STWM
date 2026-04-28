# STWM Strengthen Reappearance Supervision V1 Audit

- generated_at_utc: `2026-04-28T05:59:22.731022+00:00`
- current_reappearance_gate_definition: `((not endpoint_visible) or obs_occluded) and obs_seen_any`
- current_reappearance_target_definition: `future_visibility_target & reappearance_gate[:, None, :]`
- previous_reappearance_mask_equaled_supervision_mask: `True`
- current_reappearance_mask_policy_after_patch: `at_risk_only`
- current_reappearance_mask_equals_supervision_mask: `False`
- non_risk_slots_as_negatives_before_patch: `True`
- non_risk_slots_as_negatives_after_patch: `False`
- positive_rate_all_slots_previous: `0.017956543262698688`
- head_only_steps_with_positive_estimate_source: `previous training did not record per-step positive batch count; sampling audit added in this pack`
- head_only_signal_negative: `True`
- reappearance_AUROC_pre: `0.6341318182937449`
- reappearance_AUROC_head_only: `0.38254797577394906`
- reappearance_AP_pre: `0.04940293572430614`
- reappearance_AP_head_only: `0.01227077340251688`
- previous_random_head_distribution_available: `False`
- failure_likely_causes_with_evidence: `{"class_imbalance_loss": "auto pos_weight existed but positives were diluted across all slots; at-risk mask and event loss added", "hidden_lacks_signal": "not concluded; must compare headonly_v2 against random-head distribution first", "mask_too_broad": "previous builder used future_reappearance_mask=supervision_mask; all ordinary non-risk slots became negatives", "no_positive_aware_sampling": "previous command did not use reappearance-positive oversample; positive-aware sampler added now", "target_definition": "gate itself is reasonable, but loss mask was not restricted to gate/at-risk slots"}`
## code_evidence
- visibility_target_builder_exists: `True`
- at_risk_mask_present: `True`
- event_target_present: `True`
- event_logit_present: `True`
- event_loss_present: `True`
- positive_sampler_present: `True`
