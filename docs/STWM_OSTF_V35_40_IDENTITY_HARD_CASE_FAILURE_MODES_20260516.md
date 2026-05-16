# STWM OSTF V35.40 Identity Hard-Case Failure Modes

- v35_40_identity_hard_case_failure_audit_done: true
- identity_feature_alignment_ok: True
- hard_vspw_identity_failure_detected: False
- measurement_baseline_dominates_failed_clips: False
- measurement_baseline_also_weak_on_failed_clips: True
- failed_sample_uids: ['VSPW__1176_i0goiI8AhPk', 'VSPW__93_qZmq-lc8lAg', 'VSPW__15_3oTBlynptOo', 'VIPSEG__1041_kIXALP9plU0', 'VSPW__2165_k0X5Y4jbUTY', 'VIPSEG__1044_X1-6N9RWn6g']
- m128_h32_video_system_benchmark_claim_allowed: false
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: fix_identity_inputs_with_trace_instance_cues

## 中文总结
V35.40 显示 identity blocker 集中在少数 VSPW hard/confuser/crossing clips；rerun feature alignment 没问题。下一步应修 identity hard-case 表征或 measurement-preserving head，而不是回到语义 writer/gate。

## Claim boundary
本轮只做 identity hard-case failure attribution；不能 claim full video semantic/identity system。
