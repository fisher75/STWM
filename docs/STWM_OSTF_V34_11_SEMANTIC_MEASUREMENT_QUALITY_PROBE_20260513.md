# V34.11 semantic measurement quality probe 中文报告

- 中文结论: `V34.11 semantic measurement quality probe 已完成；该 probe 只评估 observed measurement 与 future supervision 的关系，不训练主模型。`
- semantic_measurement_quality_probe_done: `True`
- semantic_measurement_quality_passed: `True`
- measurement_beats_random: `True`
- measurement_beats_pointwise_on_hard: `True`
- measurement_beats_pointwise_on_changed: `True`
- measurement_future_alignment_by_subset: `{'val': {'best_measurement_valid': 0.5985951085890598, 'random_valid': 0.27110718266415823, 'pointwise_valid': 0.36945859576314793, 'best_minus_random_valid': 0.41335210637144537, 'best_minus_pointwise_semantic_hard': 0.27888114534991376, 'best_minus_pointwise_changed': 0.21552869839127148, 'best_minus_pointwise_strict_residual': 0.3135182604548075}, 'test': {'best_measurement_valid': 0.5865809059197307, 'random_valid': 0.27028542635956854, 'pointwise_valid': 0.3306925478871466, 'best_minus_random_valid': 0.41066790735691994, 'best_minus_pointwise_semantic_hard': 0.3113444685883701, 'best_minus_pointwise_changed': 0.2482549818153876, 'best_minus_pointwise_strict_residual': 0.33393303492143905}}`
- measurement_temporal_stability: `{'val': 0.8166830621022079, 'test': 0.8225021135961436}`
- measurement_instance_consistency: `{'val': 0.7888407679692071, 'test': 0.7768056178213205}`
- measurement_discriminative_margin: `{'val': 0.4096404298146566, 'test': 0.406563996827814}`
- recommended_interpretation: `measurement 若不赢 hard/changed 的 pointwise base，优先修 measurement bank；若只赢 random，可谨慎跑 local usage probe 但不能训练 learned gate。`
