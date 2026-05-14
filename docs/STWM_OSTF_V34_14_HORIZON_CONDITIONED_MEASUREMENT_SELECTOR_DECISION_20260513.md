# V34.14 horizon-conditioned selector 决策中文报告

- 中文结论: `V34.14 horizon-conditioned selector 已评估；它用 future trace hidden 逐 horizon 读取 observed semantic memory，不使用 future teacher 作为输入。`
- horizon_conditioned_selector_built: `True`
- selector_was_trained: `True`
- measurement_selector_nonoracle_passed: `False`
- selector_beats_random: `{'val': True, 'test': True}`
- selector_beats_pointwise_on_hard: `{'val': True, 'test': True}`
- selector_beats_pointwise_on_changed: `{'val': True, 'test': True}`
- selector_beats_v34_13_on_oracle_gap: `True`
- oracle_gap_to_selector_hard: `{'val': 0.2125378471579924, 'test': 0.24685240509561485}`
- oracle_gap_to_selector_changed: `{'val': 0.1318851563690839, 'test': 0.16231445223993368}`
- selector_entropy: `{'val': 0.41540904839833576, 'test': 0.365915114680926}`
- selector_max_weight: `{'val': 0.6693363388379415, 'test': 0.7059178749720255}`
- recommended_next_step: `fix_nonoracle_measurement_selector`
