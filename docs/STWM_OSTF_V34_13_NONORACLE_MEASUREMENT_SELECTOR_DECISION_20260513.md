# V34.13 non-oracle selector 决策中文报告

- 中文结论: `V34.13 训练式 non-oracle selector 已评估；通过要求 observed-only selector 在 val/test 均赢 random、hard/changed pointwise，并且不能离 oracle best 太远。`
- selector_was_trained: `True`
- measurement_selector_nonoracle_passed: `False`
- selector_beats_random: `{'val': True, 'test': True}`
- selector_beats_pointwise_on_hard: `{'val': False, 'test': False}`
- selector_beats_pointwise_on_changed: `{'val': False, 'test': False}`
- oracle_gap_to_selector_hard: `{'val': 0.27253198629120323, 'test': 0.29993541381084626}`
- oracle_gap_to_selector_changed: `{'val': 0.20594463908048166, 'test': 0.2389155737474489}`
- selector_entropy: `{'val': 0.4594583958387375, 'test': 0.4021715819835663}`
- test_confirmation: `False`
- recommended_next_step: `fix_nonoracle_measurement_selector`
