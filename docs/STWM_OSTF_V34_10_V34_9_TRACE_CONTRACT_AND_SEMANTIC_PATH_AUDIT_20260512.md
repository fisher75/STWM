# V34.10 对 V34.9 trace contract 与 semantic path 的中文审计

- 中文结论: `V34.9 bank 已保存真实 trace state，但 V34.9 probe 仍复用 V34.8 dataset，实际训练把 semantic confidence 当成 obs_conf，且 usage/assignment contrast loss 未激活。`
- measurement_bank_obs_points_real: `True`
- measurement_bank_obs_vis_real: `True`
- measurement_bank_obs_conf_real: `True`
- train_dataset_uses_real_obs_conf: `False`
- obs_conf_semantic_confidence_substitution_detected: `True`
- trace_state_contract_fully_passed: `False`
- semantic_usage_loss_inactive: `True`
- assignment_contrast_loss_inactive: `True`
- v34_9_measurement_report_json_missing: `False`
- v34_9_target_report_json_missing: `False`
- recommended_fix: `新增 V34.10 dataset，明确读取 zm['obs_conf'] 为 trace_obs_conf；保留 semantic_measurement_confidence 独立字段；usage/assignment contrast loss 只 detach 对照分支，不 detach normal path。`
