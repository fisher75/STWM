# TRACEWM Stage1-v2 Scientific Rigor-Fix Results

- generated_at_utc: 2026-04-08T11:02:55.148252+00:00
- best_state_variant: stage1_v2_state_multitoken_gru
- best_backbone_variant: stage1_v2_backbone_transformer_debugsmall
- best_loss_variant: stage1_v2_loss_coord_visibility
- final_mainline_model: stage1_v2_mainline::multitoken::stage1_v2_backbone_transformer_debugsmall::coord_visibility
- final_mainline_parameter_count: 3213066
- final_mainline_target_220m_range_pass: False
- validation_status: scientifically_validated
- whether_v2_is_scientifically_validated: True
- best_small_model: stage1_v2_backbone_transformer_debugsmall
- best_220m_model: stage1_v2_backbone_transformer_prototype220m
- should_promote_220m_now: False
- next_step_choice: run_220m_competitiveness_gap_closure

## Validation Gaps
- none

## Evaluation Policy
- primary: free_rollout_endpoint_l2
- secondary: free_rollout_coord_mean_l2
- tertiary: tapvid_eval.free_rollout_endpoint_l2 or tapvid3d_limited_eval.free_rollout_endpoint_l2
- total_loss: reference_only_not_primary

## Mainline Replay Evidence
- replay_report_json: /home/chen034/workspace/stwm/reports/stage1_v2_mainline_replay_20260408.json
- replay_report_md: /home/chen034/workspace/stwm/docs/STAGE1_V2_MAINLINE_REPLAY_20260408.md
