# STWM OSTF V32 Model Code Audit

- py_compile_all_ok: `True`
- forward_smoke_ok: `True`
- forward_shape_report: `{'point_hypotheses_shape': [2, 16, 4, 7, 2], 'top1_point_pred_shape': [2, 16, 4, 2], 'point_pred_shape': [2, 16, 4, 2], 'visibility_logits_shape': [2, 16, 4], 'semantic_logits_shape': [2, 16, 4, 32], 'recurrent_loop_steps': 4, 'main_rollout_state_is_field': True, 'global_motion_prior_active': True}`
- preserves_point_state: `True`
- has_recurrent_loop: `True`
- feeds_predicted_position_forward: `True`
- has_global_motion_prior_branch: `True`
- has_semantic_broadcast_context: `True`
- semantic_loss_disabled_unless_target_exists: `True`
