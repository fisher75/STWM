# STWM OSTF V35.19 Boundary-Risk Video Semantic Target Build

- boundary_risk_video_semantic_state_targets_built: True
- sample_count: 325
- vipseg_source_train_val_expanded: True
- vspw_test_changed_sparse: True
- vspw_test_hard_sparse: True
- future_teacher_embedding_input_allowed: false
- recommended_next_step: eval_boundary_risk_video_semantic_predictability

## 中文总结
V35.19 使用真实 mask 局部边界、trace motion、visibility/confidence 构建 ontology-agnostic hard/risk target；目标是避免 V35.18 纯 label transition 过窄导致 VSPW test changed/hard 不可评估。
