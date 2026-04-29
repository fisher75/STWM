# STWM Semantic Trace Field Decoder V2 Trainability Audit

- can_isolate_semantic_branch: `True`
- proposed_semantic_branch_trainable_params: `1054726`
- stage1_trainable_param_count: `0`
- trace_backbone_trainable: `False`
- implemented_controlled_unfreeze_scope: `['future_semantic_state_head', 'semantic_fusion.semantic_proj', 'readout_head']`

Trace-unit semantic internals are kept frozen because current modules do not expose a clean semantic-only trainability boundary.
