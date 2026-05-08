# STWM OSTF V31 Model Code Audit

- py_compile: `{'ok': True, 'returncode': 0, 'stderr': ''}`
- preserves_point_tokens_before_rollout: `True`
- main_rollout_state_shape: `[B,M,H,D]`
- global_token_context_only: `True`
- semantic_token_context_only: `True`
- does_not_pool_to_main_object_token: `False`
- output_contract_shapes: `[{'M': 128, 'H': 32, 'point_hypotheses_shape': [2, 128, 32, 9, 2], 'top1_point_pred_shape': [2, 128, 32, 2], 'point_pred_shape': [2, 128, 32, 2], 'visibility_logits_shape': [2, 128, 32], 'semantic_logits_shape': [2, 128, 32, 32], 'hypothesis_logits_shape': [2, 128, 9], 'main_rollout_state_is_field': True, 'uses_object_token_only_shortcut': False}, {'M': 512, 'H': 32, 'point_hypotheses_shape': [2, 512, 32, 9, 2], 'top1_point_pred_shape': [2, 512, 32, 2], 'point_pred_shape': [2, 512, 32, 2], 'visibility_logits_shape': [2, 512, 32], 'semantic_logits_shape': [2, 512, 32, 32], 'hypothesis_logits_shape': [2, 512, 9], 'main_rollout_state_is_field': True, 'uses_object_token_only_shortcut': False}, {'M': 1024, 'H': 32, 'point_hypotheses_shape': [2, 1024, 32, 9, 2], 'top1_point_pred_shape': [2, 1024, 32, 2], 'point_pred_shape': [2, 1024, 32, 2], 'visibility_logits_shape': [2, 1024, 32], 'semantic_logits_shape': [2, 1024, 32, 32], 'hypothesis_logits_shape': [2, 1024, 9], 'main_rollout_state_is_field': True, 'uses_object_token_only_shortcut': False}]`
- supports_M128_M512_M1024_forward: `True`
- semantic_training_status: `not_tested_not_failed`
