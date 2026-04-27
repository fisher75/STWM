# STWM Reappearance Head Repair V1 Schema Update

- generated_at_utc: `2026-04-27T16:29:49Z`
- updated_file: `code/stwm/tracewm_v2_stage2/models/future_semantic_trace_state.py`
- future_reappearance_logit_added: `True`
- field_optional_for_backward_compatibility: `True`
- required_shape_if_present: `[B,H,K]`
- validate_checks_rank_and_prefix_alignment: `True`
- shape_dict_includes_future_reappearance_logit: `True`
- as_tensor_dict_includes_future_reappearance_logit_when_present: `True`
- old_checkpoint_compatibility: `missing field is allowed; eval marks reappearance_head_available=false only if model head absent`
