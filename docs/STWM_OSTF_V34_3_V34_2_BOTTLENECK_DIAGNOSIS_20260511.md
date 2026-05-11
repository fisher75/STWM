# STWM OSTF V34.3 V34.2 Bottleneck Diagnosis

- v34_2_units_load_bearing: `True`
- v34_2_units_predictively_successful: `False`
- pointwise_no_unit_dominates: `True`
- teacher_top5_delta_vs_pointwise: `{'val': -0.15025690121531626, 'test': -0.15482357371782557}`
- identity_auc_delta_vs_pointwise: `{'val': -0.0009517153988686244, 'test': -0.0029437424466478568}`
- drop_z_dyn_metric_delta: `{'identity_auc': {'val': 0.00013502068084314534, 'test': -0.00012087812332695069}, 'teacher_top5': {'val': -0.0004264267528438781, 'test': -3.842872624143512e-06}}`
- drop_z_sem_metric_delta: `{'identity_auc': {'val': -0.01757732614222829, 'test': 0.00012918207933565462}, 'teacher_top5': {'val': -0.04854157869872949, 'test': -0.04673701685483933}}`
- permute_assignment_metric_delta: `{'identity_auc': {'val': -0.02655066164862141, 'test': -0.01685933878536483}, 'teacher_top5': {'val': -0.08516411159002829, 'test': -0.0697135522745963}}`
- unit_bottleneck_detected: `True`
- recommended_architecture_fix: `Use pointwise no-unit prediction as the preserved base path and restrict trace units to gated residual memory corrections for hard/changed/confuser cases.`
