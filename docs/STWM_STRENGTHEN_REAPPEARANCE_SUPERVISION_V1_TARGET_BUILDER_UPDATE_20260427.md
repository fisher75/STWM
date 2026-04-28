# STWM Strengthened Reappearance Target Builder Update

- generated_at_utc: `2026-04-28T05:59:22.731022+00:00`
- target_builder_file: `code/stwm/tracewm_v2_stage2/utils/visibility_reappearance_targets.py`
- mask_policy_fixed: `True`
- default_reappearance_mask_policy: `at_risk_only`
- reappearance_gate_definition: `((not endpoint_visible) or obs_occluded) and obs_seen_any`
- reappearance_target_definition: `future_visibility_target & reappearance_gate[:, None, :]`
- future_reappearance_risk_mask_definition: `supervision_mask & reappearance_gate[:, None, :]`
- future_reappearance_mask_default: `future_reappearance_risk_mask`
- negative_policy: `future-invisible within at-risk slots only`
- ablation_policy_available: `all_slots`
- new_reporting_fields: `["future_reappearance_risk_slot_ratio", "future_reappearance_risk_entry_ratio", "future_reappearance_positive_rate_all_slots", "future_reappearance_positive_rate_at_risk", "future_reappearance_mask_policy", "future_reappearance_negative_policy"]`
