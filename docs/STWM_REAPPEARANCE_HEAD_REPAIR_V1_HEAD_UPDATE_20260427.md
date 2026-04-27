# STWM Reappearance Head Repair V1 Head Update

- generated_at_utc: `2026-04-27T16:29:49Z`
- updated_file: `code/stwm/tracewm_v2_stage2/models/semantic_trace_world_head.py`
- independent_reappearance_head_added: `True`
- future_reappearance_logit_source: `SemanticTraceStateHead.reappearance_head(future_hidden)`
- visibility_loss_uses: `state.future_visibility_logit`
- reappearance_loss_uses: `state.future_reappearance_logit`
- visibility_logit_used_as_reappearance_fallback: `False`
- missing_reappearance_logit_with_positive_loss_policy: `raises RuntimeError instead of silent fallback`
- auto_pos_weight_added: `True`
- auto_pos_weight_formula: `negatives / positives under future_reappearance_mask, clamped to [1, future_reappearance_pos_weight_max]`
- loss_info_fields: `["future_reappearance_head_available", "future_reappearance_pos_weight", "future_reappearance_positive_rate", "future_reappearance_loss_uses_independent_logit"]`
