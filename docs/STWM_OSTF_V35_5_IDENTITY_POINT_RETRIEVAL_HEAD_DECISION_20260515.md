# STWM OSTF V35 Semantic State Head Decision

- semantic_state_head_training_ran: true
- semantic_state_head_passed: False
- semantic_hard_signal: {'val': True, 'test': True}
- changed_semantic_signal: {'val': True, 'test': True}
- stable_preservation: {'val': False, 'test': False}
- unit_memory_load_bearing: True
- semantic_measurement_load_bearing: True
- assignment_load_bearing: True
- identity_retrieval_passed: False
- identity_retrieval_exclude_same_point_top1: val=0.8492932298862791 test=0.7973714069591528
- identity_retrieval_same_frame_top1: val=0.8492932298862791 test=0.7973714069591528
- identity_retrieval_instance_pooled_top1: val=0.6929330831769857 test=0.6605326428703533
- identity_confuser_separation: val=0.1914080434798966 test=0.08482016136061238
- recommended_next_step: fix_v35_semantic_state_head

## 中文总结
V35 identity retrieval 评估已加入 pairwise / retrieval gate。只有 exclude-same-point、same-frame、instance-pooled 检索和 confuser separation 同时过，才允许 identity field claim；semantic field success 仍不允许。
