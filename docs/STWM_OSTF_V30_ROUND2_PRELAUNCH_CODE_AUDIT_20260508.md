# STWM OSTF V30 Round-2 Prelaunch Code Audit

- fatal_issue_found: `False`
- fatal_issues: `[]`
- semantic_status: `{'semantic_load_bearing': 'not_tested', 'semantic_not_tested_not_failed': True, 'semantic_id_valid_ratio': 0.0}`
- strongest_prior_naming: `{'h32': 'last_observed_copy', 'h64': 'last_observed_copy', 'tie_status': 'last_observed_copy_or_last_visible_copy_tie', 'report_name_if_tie': 'last_observed_copy_or_last_visible_copy_tie'}`
- bootstrap_pairing_rule: `item_key=uid+H+M+cache_path when present; legacy uid+H+M fallback only for old seed42 rows`
- item_uid_uniqueness: `{'h32_seed42': {'row_count': 169, 'uid_h_m_unique': True, 'item_key_unique': True, 'item_key_present_ratio': 0.0}, 'h64_seed42': {'row_count': 202, 'uid_h_m_unique': True, 'item_key_unique': True, 'item_key_present_ratio': 0.0}}`
- seed42_eval_validation: `{'h32_train_loss_decreased': True, 'h64_train_loss_decreased': True, 'h32_eval_item_row_count': 169, 'h64_eval_item_row_count': 202}`
- missrate32_saturation: `{'h32_motion': 'saturated_or_non_discriminative', 'h64_motion': 'saturated_or_non_discriminative'}`
- threshold_auc_metric_key: `threshold_auc_endpoint_16_32_64_128`
- ready_to_launch_round2: `True`
