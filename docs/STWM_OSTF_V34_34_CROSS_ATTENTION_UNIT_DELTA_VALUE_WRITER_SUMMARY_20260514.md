# V34.34 cross-attention unit-delta value writer 训练中文摘要

- 中文结论: `V34.34 cross-attention unit-delta writer 训练完成；使用 unit/horizon query 读取 point×top-k raw evidence set，显式蒸馏 V34.33 oracle unit_delta 缓存，不训练 learned gate。`
- fresh_training_completed: `True`
- checkpoint_path: `outputs/checkpoints/stwm_ostf_v34_34_cross_attention_unit_delta_value_writer_h32_m128/v34_34_cross_attention_unit_delta_value_writer_m128_h32_seed42_top1.pt`
- target_kind: `top1`
- steps: `1500`
- train_sample_count: `128`
- value_hidden_dim: `256`
- v30_backbone_frozen: `True`
- assignment_frozen: `True`
- learned_gate_training_ran: `False`
- future_leakage_detected: `False`
- trajectory_degraded: `False`
