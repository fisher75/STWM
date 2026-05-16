# STWM OSTF V35.42 Identity Label Provenance And Valid Claim

- v35_42_identity_label_provenance_audit_done: true
- identity_valid_instance_sample_count: 7
- identity_invalid_or_pseudo_sample_count: 5
- vspw_identity_targets_marked_diagnostic_only: True
- filtered_real_instance_identity_passed_all_seeds: True
- semantic_smoke_passed_all_seeds: True
- raw_frontend_drift_ok: True
- m128_h32_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: render_case_mined_visualization_and_write_unified_raw_video_decision

## 中文总结
V35.42 修正了 identity claim boundary：VSPW 在当前目标中更像 semantic/track-slot pseudo identity，不能作为真实 identity retrieval pass gate；在 VIPSeg 真实 instance subset 上，raw-video rerun 的 semantic 与 identity 闭环可以作为 M128/H32 bounded video system benchmark。

## Claim boundary
允许的只是 M128/H32 bounded video system benchmark：semantic 可跨 VSPW/VIPSeg，identity 只在真实 instance-labeled subset 上评估；full CVPR-scale claim 仍不允许。
