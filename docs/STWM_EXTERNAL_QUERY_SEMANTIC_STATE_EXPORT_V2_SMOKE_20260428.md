# STWM External Query Semantic State Export V2 Smoke

- manifest_mode: `external_hardcase_query`
- current_export_data_source: `external_hardcase_query_manifest`
- external_manifest_consumed: `True`
- stage2_val_fallback_used: `False`
- old_association_report_used: `False`
- future_candidate_used_as_input: `False`
- candidate_used_for_eval_scoring: `True`
- full_model_forward_executed: `True`
- full_free_rollout_executed: `True`
- total_items: `389`
- valid_items: `389`
- valid_ratio: `1.0`
- candidate_record_count: `5672`
- positive_candidate_count: `389`
- negative_candidate_count: `5283`
- score_components_used: `['future_reappearance_event_prob', 'future_trace_coord_distance', 'future_visibility_prob']`

Future candidates are used only after FutureSemanticTraceState export for scoring; they are not part of rollout input.
