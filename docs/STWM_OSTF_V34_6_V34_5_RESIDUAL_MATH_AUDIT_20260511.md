# STWM OSTF V34.6 V34.5 Residual Math Audit

- residual_composition: `normalize(base + gate * residual)`
- current_delta_target_type: `orthogonal_projection_only`
- current_delta_is_orthogonal_only: `True`
- true_vector_delta_missing: `True`
- anti_base_correction_allowed: `False`
- orthogonality_regularization_misaligned: `True`
- v34_4_residual_init_not_used: `True`
- residual_content_ablation_not_real: `True`
- strict_residual_target_json_missing_but_md_exists: `False`
- recommended_residual_parameterizations: `['standalone_target_residual', 'orthogonal_delta_residual', 'true_vector_delta_residual', 'scaled_tangent_delta_residual', 'mixture_residual']`
- recommended_fix: `Run a residual parameterization/capacity sweep, including true_vector_delta and v34_4 residual initialization, before any learned gate training.`
