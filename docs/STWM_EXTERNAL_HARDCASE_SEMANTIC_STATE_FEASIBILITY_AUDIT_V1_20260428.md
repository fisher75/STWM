# STWM External Hardcase Semantic-State Feasibility Audit V1

- external_389_manifest_exists: true
- item_count: 389
- items_with_observed_target: 389
- items_with_future_candidates: 389
- items_with_gt_candidate_id: 389
- items_with_gt_candidate_id_in_future_candidates: 389
- can_construct_hardcase_event_target: true
- can_construct_strong_slot_aligned_target: false
- current_export_tool_supports_external_manifest_full_model: false
- allowed_to_enter_external_hardcase_semantic_state_eval: false

Blocking reasons:

- export_future_semantic_trace_state_20260427.py full_model modes build Stage2SemanticDataset from checkpoint args and do not consume external manifest items
- external manifest has observed prompt and single future-frame candidates, but no per-horizon [H,K] visibility sequence
- external candidates are not aligned to Stage2 model entity slots/fut_valid tensors

The manifest supports future candidate association labels, but current full-model FutureSemanticTraceState export cannot consume this external manifest. Running it would silently evaluate Stage2 val split, so external hardcase eval is blocked rather than faked.
