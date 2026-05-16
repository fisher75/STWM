# STWM OSTF V36.1 Strongest-Prior Downstream Slice Build

- strongest_prior_name: last_observed_copy
- sample_count: 325
- real_instance_identity_count: 121
- pseudo_identity_count: 204
- future_points_source: strongest_analytic_prior:last_observed_copy
- future_teacher_trace_input_allowed: false
- future_leakage_detected: false

## 中文总结
已构建 strongest-prior downstream slice：future_points 使用 last_observed_copy，用于直接比较 V36 V30 causal trace 与 strongest prior 的 semantic/identity downstream utility。
