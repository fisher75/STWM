# STWM OSTF V35.41 Trace-Instance Cue Identity Probe

- v35_41_trace_instance_cue_identity_probe_done: true
- best_trace_instance_method_by_val: measurement_trace_fused_w0.1
- trace_instance_cues_help_identity: False
- best_trace_instance_method_passes_val_test: False
- m128_h32_video_system_benchmark_claim_allowed: false
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: fix_identity_targets_or_video_instance_supervision

## 中文总结
V35.41 显示简单 trace-shape 融合还不足以修复 hard identity；下一步应回到 identity target / video instance supervision，而不是继续调 semantic head。

## Claim boundary
本轮是 identity 输入上界 probe；未训练新模型，不能开放完整系统 claim。
