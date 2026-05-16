# STWM OSTF V36.6 Occlusion/Reappear Identity Field Repair Decision

## 中文总结
V36.6 发现并修复了 occlusion/reappear identity 的 target/eval contract：旧 mask 来自 predicted future_vis 且为空，因此 V36.3 的 0.0 是空 target 被聚合为 0，不是 real-instance occlusion 全失败。使用 future_trace_teacher_vis 作为 supervision-only/eval-only target 后，现有 V35.29 identity head 在 real-instance occlusion/reappear 上三 seed 通过。

## 关键发现
- original_predicted_future_vis_occ_mask_empty: True
- teacher_future_vis_occ_target_available: True
- previous_v36_3_occlusion_reappear_top1: 0.0
- val_occlusion_reappear_retrieval_top1: 0.9981999672721322
- test_occlusion_reappear_retrieval_top1: 0.9967759269210102
- val_occlusion_reappear_total: 2037.0
- test_occlusion_reappear_total: 1861.0
- occlusion_reappear_identity_three_seed_passed: True
- future_teacher_trace_input_allowed: false
- future_teacher_embedding_input_allowed: false
- future_trace_predicted_from_past_only: true
- full_cvpr_scale_claim_allowed: false

## Claim 边界
- 可以把 V36.3 中 occlusion/reappear=0.0 更正为 target/eval contract 问题，而不是模型在真实遮挡样本上全错。
- 如果引用 V36.6 指标，必须说明遮挡/再出现 target 来自 future_trace_teacher_vis，仅作为 supervision/eval target，不作为模型输入。
- 仍不允许 claim full CVPR-scale complete system、H64/H96、M512/M1024 或 full open-vocabulary semantic segmentation。

## 输出
- audit_report: `reports/stwm_ostf_v36_6_occlusion_reappear_identity_target_contract_audit_20260516.json`
- eval_summary: `reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_eval_summary_20260516.json`
- decision_report: `reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_decision_20260516.json`
- override_root: `outputs/cache/stwm_ostf_v36_6_occlusion_reappear_identity_target_overrides/M128_H32`
- log: `outputs/logs/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_20260516.log`
- recommended_next_step: `update_v36_claim_table_after_occlusion_identity_repair`
