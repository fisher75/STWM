# STWM OSTF V31 V30 Field Compression Audit

- v30_pools_point_tokens_to_object_token: `True`
- v30_temporal_rollout_object_token_time_token: `True`
- v30_per_point_future_decoder_residual: `True`
- v30_main_rollout_state_field_preserving: `False`
- v30_can_be_used_as_object_token_rollout_baseline: `True`
- needs_v31_field_preserving_rollout: `True`
- answers: `{'1_point_tokens_pooled_to_object_token': 'yes: V30 computes point_token [B,M,D], then density_pooler returns pooled [B,D].', '2_rollout_state': 'yes: rollout consumes step_tokens = obj[:,None,:] + time_embed, so temporal state is object/time token.', '3_per_point_future': 'yes: per-point residual decoder adds step_hidden back to each point_token; points are decoded but not rolled out as the primary state.', '4_density_failure_alignment': 'consistent: higher M can change pooled statistics but cannot create independent field rollout states.', '5_field_output_vs_field_state': 'output is field-shaped, but the main rollout state is not field-preserving.', '6_v30_role': 'V30 should be reported as an object-token rollout baseline, not as strict field-level rollout.', '7_v31_need': 'V31 is required to test the original trace-field world-model hypothesis.'}`
