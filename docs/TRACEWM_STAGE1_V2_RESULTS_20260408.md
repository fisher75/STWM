# TRACEWM Stage1-v2 Scientific Revalidation Results

- generated_at_utc: 2026-04-08T10:19:20.178820+00:00
- best_state_variant: stage1_v2_state_multitoken_gru
- best_backbone_variant: stage1_v2_backbone_transformer_debugsmall
- best_loss_variant: stage1_v2_loss_coord_visibility
- final_mainline_model: stage1_v2_mainline::stage1_v2_state_multitoken_gru::stage1_v2_backbone_transformer_debugsmall::stage1_v2_loss_coord_visibility
- final_mainline_parameter_count: 3213066
- final_mainline_target_220m_range_pass: False
- whether_v2_is_scientifically_validated: True
- next_step_choice: do_small_followup_ablation

## Evaluation Policy
- primary: free_rollout_endpoint_l2
- secondary: free_rollout_coord_mean_l2
- tertiary: tapvid_eval.free_rollout_endpoint_l2
- total_loss: reference_only_not_primary
