# STWM World Model No-Drift Guardrail V3 20260427

## Allowed
- FutureSemanticTraceState is the world-model output contract.
- SAM2/CoTracker can only be external consumers or baselines.
- Association/reacquisition is utility, not the model definition.
- Medium training may validate semantic trajectory world-state output only after non-degenerate signal judgement.

## Forbidden
- Calling STWM a SAM2 plugin.
- Calling 16-item smoke a paper-level semantic trajectory world model.
- Using simplified visibility_accuracy=1.0 as calibrated reappearance evidence.
- Moving to 1B before medium semantic-state signal is positive.
- Claiming external hard-case full-model export when export used Stage2 val split.

- visibility_metric_status: `smoke_only_simplified_target until real per-time/per-entity visibility/reappearance targets are implemented`
- current_export_data_source: `Stage2SemanticDataset validation split from checkpoint args for full-model export/eval in this pack`
