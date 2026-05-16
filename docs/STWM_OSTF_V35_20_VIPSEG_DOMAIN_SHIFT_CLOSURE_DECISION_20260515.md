# STWM OSTF V35.20 VIPSeg Domain Shift Closure Decision

- vipseg_processed_split_counts: {'test': 15, 'train': 85, 'val': 21}
- target_sample_count: 325
- mixed_unseen_passed: True
- vspw_to_vipseg_passed: False
- vipseg_to_vspw_stratified_passed: False
- cross_dataset_video_semantic_suite_ready: False
- semantic_adapter_training_ran: false
- full_video_semantic_identity_field_claim_allowed: false
- recommended_next_step: fix_video_semantic_target_split_or_domain_normalization_before_adapter_training

## 中文总结
V35.20 扩大 VIPSeg source 后，mixed-unseen 已通过，但 VIPSeg→VSPW 在 all/stratified 下仍未过；这说明当前 mask-derived semantic target 仍有真实跨数据集域迁移/target split 问题。不能训练 adapter，不能 claim 完整 video semantic field。
