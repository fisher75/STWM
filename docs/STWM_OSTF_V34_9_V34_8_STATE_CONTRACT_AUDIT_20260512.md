# V34.9 对 V34.8 trace state contract 的中文审计

- 中文结论: `V34.8 使用的 V34 semantic measurement bank 违反 trace contract：obs_points 为零，obs_vis 来自 semantic measurement mask，训练/评估又把这些字段送入 frozen V30。`
- obs_points_zero_detected: `True`
- obs_vis_from_semantic_mask_detected: `True`
- obs_conf_from_semantic_confidence_detected: `True`
- v34_8_uses_real_trace_input: `False`
- trace_conditioned_contract_broken: `True`
- artifact_packaging_truly_fixed: `True`
- semantic_usage_loss_inactive: `True`
- assignment_contrast_loss_inactive: `True`
- recommended_fix: `重建 trace-preserving semantic measurement bank：obs_points/obs_vis/obs_conf 必须来自真实 V30 external-GT trace sidecar；然后重建 causal targets 并重跑 oracle probe，不训练 learned gate。`
