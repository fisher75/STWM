# STWM OSTF V35.21 V35.20 Domain Shift / Target Split Audit

- v35_20_domain_shift_target_split_audit_done: true
- vipseg_to_vspw_target_split_imbalanced: True
- vspw_heldout_changed_sparse: True
- domain_normalized_risk_calibration_required: true
- adapter_training_should_remain_blocked: true
- recommended_fix: build_domain_normalized_per_video_risk_targets_and_dataset_balanced_unseen_protocol

## 中文总结
V35.20 的 mixed-unseen 有正信号，但 VIPSeg→VSPW 仍失败。审计显示 VSPW held-out 的 changed/risk 分布偏稀疏，直接用全局阈值会把数据集风格差异误当语义状态差异；下一步应做 per-video/domain-normalized risk target 和 dataset-balanced unseen 协议，不应训练 adapter。
