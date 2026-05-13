# V34.11 local semantic usage oracle probe 训练中文摘要

- 中文结论: `V34.11 local semantic usage oracle probe 已完成训练；usage/assignment loss 改为局部逐点逐 horizon，对照分支 detach，normal path 保持可训练；未训练 learned gate。`
- local_semantic_usage_probe_ran: `True`
- fresh_training_completed: `True`
- checkpoint_path: `outputs/checkpoints/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_h32_m128/v34_11_local_semantic_usage_oracle_probe_m128_h32_seed42_best.pt`
- train_sample_count: `128`
- pooling_variant: `teacher_agreement_weighted_pooling`
- v30_backbone_frozen: `True`
- trace_state_contract_fully_passed: `True`
- local_semantic_usage_loss_active: `True`
- local_assignment_contrast_loss_active: `True`
- semantic_measurement_usage_score_used_in_residual_magnitude: `True`
- teacher_agreement_score_used_in_loss_weight: `True`
- obs_measurement_confidence_used_in_loss_weight: `True`
