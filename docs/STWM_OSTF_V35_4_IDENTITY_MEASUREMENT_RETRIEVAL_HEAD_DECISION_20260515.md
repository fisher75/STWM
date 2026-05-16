# STWM OSTF V35 Semantic State Head Decision

- semantic_state_head_training_ran: true
- semantic_state_head_passed: True
- semantic_hard_signal: {'val': True, 'test': True}
- changed_semantic_signal: {'val': True, 'test': True}
- stable_preservation: {'val': True, 'test': True}
- unit_memory_load_bearing: True
- semantic_measurement_load_bearing: True
- assignment_load_bearing: False
- identity_retrieval_passed: False
- identity_retrieval_exclude_same_point_top1: val=0.7599107237751089 test=0.6984209531013615
- identity_retrieval_same_frame_top1: val=0.7599107237751089 test=0.6984209531013615
- identity_retrieval_instance_pooled_top1: val=0.5502397331665624 test=0.49704087294248195
- identity_confuser_separation: val=0.12357101074965027 test=0.07051363776477393
- recommended_next_step: fix_identity_retrieval_head

## 中文总结
V35 identity retrieval 评估已加入 pairwise / retrieval gate。只有 exclude-same-point、same-frame、instance-pooled 检索和 confuser separation 同时过，才允许 identity field claim；semantic field success 仍不允许。
