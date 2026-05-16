# STWM OSTF V36 V30 Past-Only Future Trace Rollout

- v30_checkpoint_loaded: True
- v30_backbone_frozen: True
- future_trace_predicted_from_past_only: True
- sample_count: 325
- ADE_mean: 126.70885446086619
- FDE_mean: 206.29918207485863
- visibility_F1_mean: 0.7228657176105464
- strongest_prior: last_observed_copy
- v30_beats_strongest_prior: False
- trajectory_degraded: False

## 中文总结
V36 已用 frozen V30 从 past-only observed trace 预测 future trace，并与 full-clip teacher trace 仅做 target 对比。
