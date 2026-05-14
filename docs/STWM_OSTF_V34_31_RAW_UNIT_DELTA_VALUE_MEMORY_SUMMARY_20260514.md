# V34.31 raw unit-delta value memory 训练中文摘要

- 中文结论: `V34.31 raw unit-delta value head 训练完成；冻结 V30、selector、assignment 和原 residual model，只训练 raw value head，不训练 gate。`
- fresh_training_completed: `True`
- checkpoint_path: `outputs/checkpoints/stwm_ostf_v34_31_raw_unit_delta_value_memory_h32_m128/v34_31_raw_unit_delta_value_head_m128_h32_seed42.pt`
- steps: `1000`
- train_sample_count: `128`
- residual_scale: `1.0`
- v30_backbone_frozen: `True`
- assignment_frozen: `True`
- learned_gate_training_ran: `False`
- future_leakage_detected: `False`
- trajectory_degraded: `False`
