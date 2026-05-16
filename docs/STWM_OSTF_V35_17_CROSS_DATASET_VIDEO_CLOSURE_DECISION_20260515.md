# STWM OSTF V35.17 Cross-Dataset Video Closure Decision

- processed_clip_count: 192
- mixed_unseen_passed: True
- vspw_to_vipseg_passed: True
- vipseg_to_vspw_passed: False
- cross_dataset_video_semantic_suite_ready: False
- semantic_adapter_training_ran: false
- identity_retrieval_training_ran: false
- primary_blocker: vipseg_to_vspw_low_changed_positive_ratio_and_domain_shift
- full_video_semantic_identity_field_claim_allowed: false
- recommended_next_step: fix_vipseg_to_vspw_video_semantic_domain_shift_or_target_split

## 中文总结
V35.17 已扩到 192 clips；mixed_unseen 与 VSPW→VIPSeg 通过，但 VIPSeg→VSPW 失败。因此不能继续训练 cross-dataset adapter，也不能 claim full video semantic/identity field。
