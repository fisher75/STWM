# V34.14 horizon-conditioned selector 训练中文报告

- 中文结论: `V34.14 horizon-conditioned selector 已完成训练；future trace hidden 作为 query 读取 observed semantic memory，输出 [M,H,Tobs] 权重与 [M,H,D] evidence。future teacher embedding 只用于 loss supervision。`
- horizon_conditioned_selector_built: `True`
- selector_was_trained: `True`
- fresh_training_completed: `True`
- checkpoint_path: `outputs/checkpoints/stwm_ostf_v34_14_horizon_conditioned_measurement_selector_h32_m128/v34_14_horizon_conditioned_measurement_selector_m128_h32_seed42_best.pt`
- train_sample_count: `128`
- steps: `1500`
- v30_backbone_frozen: `True`
- future_teacher_embedding_input_allowed: `False`
- measurement_weight_shape: `B,M,H,Tobs`
- selected_evidence_shape: `B,M,H,D`
