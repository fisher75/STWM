# STWM OSTF V36 Causal Unified Semantic/Identity Slice Build

- sample_count: 325
- real_instance_identity_count: 121
- pseudo_identity_count: 204
- future_points_source: v30_predicted_future_trace
- future_teacher_trace_input_allowed: false
- future_leakage_detected: false
- semantic_identity_alignment_passed: True

## 中文总结
V36 causal unified slice 已构建：obs trace 来自 past-only 输入，future_points 来自 frozen V30 预测，teacher trace 只保留为评估 target。
