# STWM OSTF V35.18 Ontology-Agnostic Video Semantic Target Build

- ontology_agnostic_video_semantic_state_targets_built: True
- sample_count: 289
- video_semantic_target_source: mask_transition / panoptic_instance_transition / visibility_conditioned_risk
- vipseg_source_train_val_expanded: True
- vspw_test_changed_hard_sparse: True
- future_teacher_embedding_input_allowed: false
- recommended_next_step: eval_vipseg_to_vspw_domain_shift_with_stratified_target

## 中文总结
V35.18 已把 semantic change 从跨数据集 ontology 类名依赖，改成同一 video 内 mask transition 和 visibility-conditioned semantic risk；VSPW test 如果 changed/hard 仍稀疏，后续评估必须做分层诊断。
