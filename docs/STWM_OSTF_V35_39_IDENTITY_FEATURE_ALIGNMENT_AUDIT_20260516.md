# STWM OSTF V35.39 Identity Feature Alignment Audit

- v35_39_identity_feature_alignment_audit_done: true
- feature_delta_mean_abs: 0.0
- feature_delta_max_abs: 0.0
- identity_pair_mask_mismatch_detected: False
- original_selected_val_identity_passed_all_seeds: False
- rerun_selected_val_identity_passed_all_seeds: False
- identity_target_alignment_broken: False
- selected_val_hard_identity_intrinsic: True
- m128_h32_video_system_benchmark_claim_allowed: false
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: fix_identity_val_confuser_category_or_eval_slice

## 中文总结
V35.39 显示 rerun unified identity feature 与原始 cache 完全一致，pair mask 也一致；identity 失败不是 raw-video rerun feature rebuild bug，而是当前 val hard slice 本身暴露了 identity retrieval 在少数 VSPW confuser/crossing clip 上不稳。

## Claim boundary
本轮是 identity feature / target alignment 审计；不开放 video identity field 或完整系统 claim。
