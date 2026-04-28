# STWM Event-Level Reappearance Head Update

- generated_at_utc: `2026-04-28T05:59:22.731022+00:00`
- schema_file: `code/stwm/tracewm_v2_stage2/models/future_semantic_trace_state.py`
- head_file: `code/stwm/tracewm_v2_stage2/models/semantic_trace_world_head.py`
- trainer_file: `code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py`
- event_target_added: `True`
- event_mask_added: `True`
- event_logit_added: `True`
- event_head_added: `True`
- event_loss_added: `True`
- event_cli_added: `True`
- event_output_shape: `[B,K]`
- event_target_definition: `any_h(future_visibility_target[h,k] & reappearance_gate[k])`
- event_mask_definition: `reappearance_gate[k] & token_mask[k]`
- backward_compatibility: `future_reappearance_event_logit optional; strict=False checkpoint loading remains valid`
