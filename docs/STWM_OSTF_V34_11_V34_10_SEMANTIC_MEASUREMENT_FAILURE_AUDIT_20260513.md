# V34.11 V34.10 semantic measurement 因果失败审计中文报告

- 中文结论: `V34.10 已修复 trace contract 并激活 usage/assignment loss，但语义 measurement 仍不是因果必要路径；当前 tokenizer 的时间均值 pooling 和全局 scalar usage loss 是主要可疑点。`
- semantic_measurement_bank_teacher_name: `dinov2_base`
- semantic_measurements_have_variance: `True`
- measurement_confidence_degenerate: `False`
- teacher_agreement_used_in_training: `False`
- semantic_pooling_too_global: `True`
- usage_loss_too_global: `True`
- semantic_usage_score_not_used_in_residual: `True`
- semantic_measurement_not_load_bearing_confirmed: `True`
- assignment_not_load_bearing_confirmed: `True`
- recommended_fix: `先运行 semantic measurement quality probe；若 measurement 本身有 hard/changed 预测力，则把 usage loss 改为局部逐点逐 horizon，并把 usage score 接入 residual magnitude；否则重建 measurement bank。`
