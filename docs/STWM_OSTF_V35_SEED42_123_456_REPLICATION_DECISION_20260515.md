# STWM OSTF V35 Seed42/123/456 Replication Decision

- seed42_123_456_semantic_state_head_passed: True
- semantic_hard_signal_replicated: True
- changed_semantic_signal_replicated: True
- stable_preservation_replicated: True
- semantic_measurement_load_bearing_replicated: True
- unit_memory_load_bearing_replicated: False
- assignment_load_bearing_replicated: False
- integrated_semantic_field_claim_allowed: False
- recommended_next_step: fix_identity_consistency_and_unit_assignment_load_bearing

## 中文总结
V35 semantic state head 在 seed42/123/456 上复现了 stable/changed/hard 语义状态正信号，且 semantic measurement 是 load-bearing；但 identity consistency、unit memory、assignment 仍未立住，因此不能 claim 完整 semantic/identity field success，下一步应修 identity 与 unit/assignment load-bearing。
