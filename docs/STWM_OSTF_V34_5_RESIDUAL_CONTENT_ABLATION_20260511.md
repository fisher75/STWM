# STWM OSTF V34.5 Residual Content Ablation

- compared_models: `['v34_4_standalone_target_residual', 'v34_5_delta_residual', 'v34_2_pointwise_no_unit', 'oracle_target_upper_bound', 'random_unit_residual', 'residual_without_unit_memory', 'residual_with_shuffled_unit_assignment']`
- strict_residual_subset_gain: `{'test': -0.005870908498764038, 'val': 0.0062751248478889465}`
- standalone_residual_subset_gain: `{'test': 0.006715059280395508, 'val': 0.004239678382873535}`
- delta_vs_standalone_gain: `{'val': 0.0020354464650154114, 'test': -0.012585967779159546}`
- whether_delta_objective_beats_standalone_objective: `False`
- semantic_hard_gain: `{'test': False, 'val': False}`
- changed_gain: `{'test': False, 'val': False}`
- stable_delta: `{'test': True, 'val': True}`
