# V34.8 causal assignment residual target 构建中文报告

- 中文结论: `已构建更严格的 causal assignment targets；正样本需要 assignment、unit purity 和 observed semantic measurement 同时满足阈值。`
- causal_assignment_targets_built: `True`
- causal_assignment_target_ready: `True`
- point_positive_ratio_by_split: `{'train': 0.004346946775494306, 'val': 0.002316063194244123, 'test': 0.002184673086825864}`
- unit_positive_ratio_by_split: `{'train': 0.005859375, 'val': 0.005182291666666667, 'test': 0.004773021449704142}`
- semantic_measurement_required_ratio_by_split: `{'train': 0.004346946775494306, 'val': 0.002316063194244123, 'test': 0.002184673086825864}`
- coverage_loss_vs_v34_7: `{'train': 0.9661152199299339, 'val': 0.961655592469546, 'test': 0.9648063887083295}`
- coverage_loss_vs_v34_5_strict: `{'train': 0.9690263125122311, 'val': 0.9790864477161193, 'test': 0.9842729888237247}`
- thresholds_by_split: `{'train': {'point_assignment_confidence': 0.9921220541000366, 'unit_instance_purity': 0.9522188305854797, 'unit_semantic_purity': 0.7417578101158142, 'semantic_measurement_confidence': 1.0, 'teacher_confidence': 1.0, 'pointwise_error': 0.6677087187767029}, 'val': {'point_assignment_confidence': 0.8761507272720337, 'unit_instance_purity': 0.62, 'unit_semantic_purity': 0.58, 'semantic_measurement_confidence': 1.0, 'teacher_confidence': 1.0, 'pointwise_error': 0.8990535914897919}, 'test': {'point_assignment_confidence': 0.8775403648614883, 'unit_instance_purity': 0.62, 'unit_semantic_purity': 0.58, 'semantic_measurement_confidence': 1.0, 'teacher_confidence': 1.0, 'pointwise_error': 0.8804962635040283}}`
- exact_blockers: `{}`
