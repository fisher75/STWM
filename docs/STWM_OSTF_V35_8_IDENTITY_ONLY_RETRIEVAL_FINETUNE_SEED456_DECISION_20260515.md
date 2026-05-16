# STWM OSTF V35 Semantic State Head Decision

- semantic_state_head_training_ran: true
- semantic_state_head_passed: True
- semantic_hard_signal: {'val': True, 'test': True}
- changed_semantic_signal: {'val': True, 'test': True}
- stable_preservation: {'val': True, 'test': True}
- unit_memory_load_bearing: False
- semantic_measurement_load_bearing: True
- assignment_load_bearing: False
- identity_retrieval_passed: True
- identity_retrieval_exclude_same_point_top1: val=0.875650972473164 test=0.8353819969742814
- identity_retrieval_same_frame_top1: val=0.875650972473164 test=0.8353819969742814
- identity_retrieval_instance_pooled_top1: val=0.7506775067750677 test=0.7297022378398372
- identity_confuser_separation: val=0.3072691850725325 test=0.2246987242464069
- recommended_next_step: run_v35_identity_seed123_replication

## 中文总结
V35 identity retrieval 评估已加入 pairwise / retrieval gate。只有 exclude-same-point、same-frame、instance-pooled 检索和 confuser separation 同时过，才允许 identity field claim；semantic field success 仍不允许。
