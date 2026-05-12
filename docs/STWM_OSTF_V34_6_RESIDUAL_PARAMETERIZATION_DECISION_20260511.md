# STWM OSTF V34.6 Residual Parameterization Decision

- candidate_count: `10`
- residual_parameterization_passed: `True`
- best_residual_parameterization: `standalone_target_residual`
- best_residual_init: `init_from_v34_4_standalone_residual_checkpoint`
- best_checkpoint_path: `outputs/checkpoints/stwm_ostf_v34_6_residual_parameterization_h32_m128/v34_6_standalone_target_residual__init_from_v34_4_standalone_residual_checkpoint_m128_h32_seed42.pt`
- semantic_hard_signal: `{'val': False, 'test': False}`
- changed_semantic_signal: `{'val': True, 'test': False}`
- stable_preservation: `{'val': True, 'test': True}`
- strict_residual_subset_gain: `{'val': 0.027522452175617218, 'test': 0.016661137342453003}`
- delta_vs_v34_4_standalone_gain: `{'val': 0.023282773792743683, 'test': 0.009946078062057495}`
- recommended_next_step: `run_real_residual_content_ablation`
