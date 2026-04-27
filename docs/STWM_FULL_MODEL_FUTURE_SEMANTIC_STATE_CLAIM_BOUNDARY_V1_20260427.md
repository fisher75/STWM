# STWM Full-Model Future Semantic State Claim Boundary V1 20260427

- engineering_output_claimable: `True`
- paper_world_model_claimable: `False`
- visibility_metric_status: `smoke_only_simplified_target`
- current_export_data_source: `Stage2SemanticDataset validation split from checkpoint args; not external_389_item_manifest`

## Forbidden Claims
- Do not claim calibrated visibility/reappearance from current visibility_accuracy=1.0.
- Do not claim paper-level semantic trajectory world model from 16-item smoke.
- Do not claim external hard-case full-model export unless export actually uses that manifest.
- Do not describe STWM as a SAM2/CoTracker plugin.

## Allowed Engineering Claims
- Full-model teacher-forced export is wired to Stage2 model hidden states.
- Full-model free-rollout export is wired to Stage2 free-rollout hidden states.
- The current output claim is engineering-level only pending medium-scale evaluation.
