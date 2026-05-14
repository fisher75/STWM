# V34.15 timestep-supervised selector 决策中文报告

- 中文结论: `V34.15 timestep-supervised selector 已评估；该轮测试用 future target 只生成训练监督的 observed timestep label，forward 仍不读 future target。`
- horizon_timestep_supervised_selector_built: `True`
- selector_was_trained: `True`
- measurement_selector_nonoracle_passed: `False`
- selector_beats_random: `{'val': True, 'test': True}`
- selector_beats_pointwise_on_hard: `{'val': True, 'test': True}`
- selector_beats_pointwise_on_changed: `{'val': False, 'test': False}`
- selector_beats_v34_14_on_oracle_gap: `False`
- oracle_gap_to_selector_hard: `{'val': 0.22524312900545398, 'test': 0.25480853776522094}`
- oracle_gap_to_selector_changed: `{'val': 0.1409466121520368, 'test': 0.17127097666162744}`
- oracle_timestep_top1_hard: `{'val': 0.23566927630322704, 'test': 0.21126026120140004}`
- oracle_timestep_top1_changed: `{'val': 0.2165917231918299, 'test': 0.21052861247616747}`
- selector_entropy: `{'val': 0.8392158548037211, 'test': 0.8165468672911326}`
- selector_max_weight: `{'val': 0.29688774545987445, 'test': 0.3081910014152527}`
- recommended_next_step: `fix_nonoracle_measurement_selector`
