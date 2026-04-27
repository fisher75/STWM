# STWM Future Semantic Trace State Export Audit 20260427

- export_path: `reports/stwm_future_semantic_state_export_20260427.json`
- checkpoint: `outputs/checkpoints/stage2_tusb_v3p1_worldmodel_v2_smalltrain_lr1e7_20260427/latest.pt`
- checkpoint_has_future_semantic_state_head: `True`
- future_semantic_trace_field_available: `True`
- full_stage1_stage2_forward_executed: `False`
- forward_scope: `future_semantic_state_head_checkpoint_forward_with_manifest_surrogate_features`
- item_count: `64`
- valid_output_ratio: `1.0`

This exporter is deliberately readout-layer scoped in V2: it consumes a trained FutureSemanticTraceState head checkpoint and materialized item metadata. It does not redefine STWM official metrics.
