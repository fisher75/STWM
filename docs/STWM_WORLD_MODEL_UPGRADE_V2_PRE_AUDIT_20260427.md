# STWM World Model Upgrade V2 Pre Audit 20260427

- current_world_model_level: `['trace_future_state_backbone', 'semantic_conditioned_rollout_adapter']`
- FutureSemanticTraceState optional head only: `True`
- official checkpoint has trained future semantic state: `False`
- future query eval consumes FutureSemanticTraceState: `False`
- free rollout semantic state head connected: `False`
- hard-coded repo root present: `True`

## Potential Bugs
- V1 _expect did not enforce exact rank for required [B,H,K] tensors when suffix was empty
- validate(strict=False) did not return coord_dim/horizon/slot_count
- V1 MultiHypothesisTraceHead emitted hypothesis_count * 2 deltas only, so 3D base coords were not supported
