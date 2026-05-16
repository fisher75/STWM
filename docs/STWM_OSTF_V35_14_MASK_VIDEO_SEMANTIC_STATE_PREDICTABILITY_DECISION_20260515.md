# STWM OSTF V35.14 Mask Video Semantic State Predictability Decision

- target_predictability_eval_done: true
- video_semantic_target_source: mask_label / panoptic_instance / object_track
- semantic_changed_is_real_video_state: true
- semantic_cluster_transition_passed: False
- semantic_changed_passed: True
- semantic_hard_passed: True
- evidence_anchor_family_passed: False
- uncertainty_target_passed: True
- observed_predictable_video_semantic_state_suite_ready: True
- semantic_state_adapter_training_allowed: True
- recommended_next_step: train_video_semantic_state_adapter

## 中文总结
V35.14 mask-derived video semantic target suite 通过 observed+future-trace 上界，可以进入 video semantic adapter。
