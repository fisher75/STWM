# STWM OSTF V35.54 Submission-Ready Benchmark Bundle

## 中文总结
V35.54 已生成 submission-ready benchmark bundle 索引和外部 sanity review checklist。当前证据链足以支持 bounded full 325 M128/H32 raw-video closure video-system benchmark claim；仍不允许 full CVPR-scale / 任意尺度 / open-vocabulary dense segmentation claim。

## 入口文件
- benchmark_card: reports/stwm_ostf_v35_52_benchmark_card_20260516.json
- package_manifest: reports/stwm_ostf_v35_52_reproducibility_package_manifest_20260516.json
- claim_table: reports/stwm_ostf_v35_50_machine_checkable_claim_table_20260516.json
- reviewer_risk_audit: reports/stwm_ostf_v35_51_external_comparison_and_reviewer_risk_audit_20260516.json
- full_325_final_decision: reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json
- reproducibility_dry_run: reports/stwm_ostf_v35_53_reproducibility_dry_run_from_manifest_20260516.json

## Reviewer sanity checklist
- 输入是否真的是 raw video / predecode frame，而不是旧 trace cache？ 是。V35.49 frontend rerun 从 raw frame paths / predecode 重跑；旧 cache 只做 drift comparison。
- V30 是否 frozen，trajectory 是否退化？ V30 M128 frozen；trajectory_degraded=false。
- future teacher embedding 是否作为 input？ 否。future teacher embedding 只允许 supervision，future_leakage_detected=false。
- semantic 是否只是 copy/persistence？ 不是只靠 copy。stable copy 被保留，但 changed/hard/uncertainty 三 seed 通过；claim 限定为 semantic state/transition/uncertainty field。
- identity 是否靠 pseudo label？ 不是。identity claim 只使用 real-instance subset；VSPW pseudo identity diagnostic-only。
- failure cases 是否隐藏？ 没有。full per-category atlas 和 case-mined visualization 已打包，包含成功和失败案例。
- 当前能否 claim full CVPR-scale complete system？ 不能。当前只允许 bounded full 325 M128/H32 raw-video closure video-system benchmark claim。

## Claim 边界
- submission_ready_benchmark_bundle_ready: True
- m128_h32_full_325_video_system_benchmark_claim_allowed: True
- full_cvpr_scale_claim_allowed: false
- recommended_next_step: external_sanity_review_or_start_result_section_draft_from_benchmark_card
