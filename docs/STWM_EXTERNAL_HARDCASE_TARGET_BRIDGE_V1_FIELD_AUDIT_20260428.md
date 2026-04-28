# STWM External Hardcase Target Bridge V1 Field Audit

- external_manifest_exists: `true`
- total_items: `389`
- mappable_to_stage2_count: `0`
- full_model_forward_possible_count: `0`
- event_target_possible_count: `389`
- candidate_aligned_target_possible_count: `389`
- per_horizon_target_possible_count: `0`

Hard blockers:
- missing_stage2_dataset_mapping_key: 389
- missing_per_horizon_visibility_sequence: 389
- full_model_forward_not_possible_from_external_raw_payload_currently: 389

The external manifest contains raw-frame candidate association targets, but no verified Stage2SemanticDataset/cache mapping. Therefore full-model FutureSemanticTraceState export cannot run on these items without an additional mapping bridge.
