# STWM External Hardcase Semantic-State Bridge V1 Smoke Summary

- external_manifest_exists: true
- original_item_count: 389
- semantic_state_manifest_item_count: 389
- usable_for_full_model_export_count: 0
- usable_for_event_eval_count: 389
- usable_for_candidate_eval_count: 389
- exported_valid_items: 0
- valid_output_ratio: 0.0
- metric_eligible_items: 0
- stage2_val_fallback_used: false
- old_association_report_used: false
- external_hardcase_eval_available: false
- external_hardcase_signal_positive: unclear
- exact_block_reason: semantic-state manifest items are external raw payloads without verified Stage2SemanticDataset/cache mapping; strict mode forbids fallback

The bridge successfully materialized candidate-aligned external hard-case targets, but strict export produced zero valid FutureSemanticTraceState outputs because no external-to-Stage2 full-model input mapping exists.
