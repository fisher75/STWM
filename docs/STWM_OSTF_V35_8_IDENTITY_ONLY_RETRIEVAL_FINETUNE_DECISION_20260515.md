# STWM OSTF V35 Semantic State Head Decision

- semantic_state_head_training_ran: true
- semantic_state_head_passed: True
- semantic_hard_signal: {'val': True, 'test': True}
- changed_semantic_signal: {'val': True, 'test': True}
- stable_preservation: {'val': True, 'test': True}
- unit_memory_load_bearing: True
- semantic_measurement_load_bearing: True
- assignment_load_bearing: True
- identity_retrieval_passed: True
- identity_retrieval_exclude_same_point_top1: val=0.8713997236688277 test=0.8273449319213313
- identity_retrieval_same_frame_top1: val=0.8713997236688277 test=0.8273449319213313
- identity_retrieval_instance_pooled_top1: val=0.7563060245987075 test=0.726280747179582
- identity_confuser_separation: val=0.2938940285523876 test=0.2151662932981117
- recommended_next_step: run_v35_identity_seed123_replication

## 中文总结
V35 identity retrieval 评估已加入 pairwise / retrieval gate。只有 exclude-same-point、same-frame、instance-pooled 检索和 confuser separation 同时过，才允许 identity field claim；semantic field success 仍不允许。
