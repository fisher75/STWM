# STWM OSTF V35.24 Balanced Cross-Dataset Changed Predictability Decision

- balanced_cross_dataset_changed_suite_ready: True
- source_only_vipseg_to_vspw_passed: True
- target_val_calibrated_vipseg_to_vspw_passed: True
- mixed_domain_balanced_unseen_passed: True
- semantic_id_shortcut_hurts_cross_dataset: True
- best_target_val_protocol: vipseg_to_vspw_target_val_future_trace_only
- recommended_next_step: run_joint_video_semantic_identity_closure_with_case_mining

## 中文总结
V35.24 是好消息：在禁止 semantic-id shortcut 后，VIPSeg→VSPW source-only、target-val calibrated 和 mixed-domain unseen 都至少有一个 ontology-agnostic feature family 通过。最稳的跨域 changed 信号来自 future/trace-risk，说明原先失败主要是 semantic-id shortcut 与 ontology shift，而不是 STWM trace-conditioned idea 失效。
