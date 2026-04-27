# STWM Future Semantic State Degeneracy Audit 20260427

- source_export: `reports/stwm_future_semantic_state_export_20260427.json`
- item_count: `64`
- raw_embedding_available: `False`
- semantic_state_degenerate: `False`
- safe_for_medium_training: `False`
- exact_failure_reason: `V2 export lacks raw future_semantic_embedding/future_identity_embedding tensors; only norm summaries are available, so unit/horizon embedding degeneracy cannot be ruled out.`

## Key Distributions
- semantic_embedding_norm_distribution: `{'count': 64, 'finite_count': 64, 'nan_inf_ratio': 0.0, 'mean': 9.234694704413414, 'std': 0.045638388955322504, 'min': 9.148828506469727, 'max': 9.354607582092285, 'all_zero_ratio': 0.0, 'constant': False}`
- identity_embedding_norm_distribution: `{'count': 64, 'finite_count': 64, 'nan_inf_ratio': 0.0, 'mean': 9.239005237817764, 'std': 0.04840054727506755, 'min': 9.137425422668457, 'max': 9.326595306396484, 'all_zero_ratio': 0.0, 'constant': False}`
- visibility_probability_distribution: `{'count': 64, 'finite_count': 64, 'nan_inf_ratio': 0.0, 'mean': 0.5076779844239354, 'std': 0.018789687497758435, 'min': 0.46152567863464355, 'max': 0.5613614320755005, 'all_zero_ratio': 0.0, 'constant': False}`
- uncertainty_distribution: `{'count': 64, 'finite_count': 64, 'nan_inf_ratio': 0.0, 'mean': 0.7270932141691446, 'std': 0.03438910143004142, 'min': 0.6411131024360657, 'max': 0.7992051243782043, 'all_zero_ratio': 0.0, 'constant': False}`
