# V34.15 horizon timestep-supervised selector 训练中文报告

- 中文结论: `V34.15 horizon timestep-supervised selector 已完成训练；future target 只用于生成训练监督的 oracle observed timestep label，selector forward 仍只读 observed semantic memory 与 frozen V30 trace query。`
- horizon_timestep_supervised_selector_built: `True`
- selector_was_trained: `True`
- fresh_training_completed: `True`
- checkpoint_path: `outputs/checkpoints/stwm_ostf_v34_15_horizon_timestep_supervised_selector_h32_m128/v34_15_horizon_timestep_supervised_selector_m128_h32_seed42_best.pt`
- init_checkpoint: `{'init_source': 'v34_14_horizon_conditioned', 'checkpoint_path': 'outputs/checkpoints/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_h32_m128/v34_14_horizon_conditioned_measurement_selector_m128_h32_seed42_best.pt', 'missing_key_count': 0, 'unexpected_key_count': 0}`
- train_sample_count: `128`
- steps: `1000`
- v30_backbone_frozen: `True`
- future_teacher_embedding_input_allowed: `False`
- oracle_timestep_label_supervision_only: `True`
