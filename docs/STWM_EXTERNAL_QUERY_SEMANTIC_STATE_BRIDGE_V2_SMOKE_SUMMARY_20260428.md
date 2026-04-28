# STWM External Query Semantic State Bridge V2 Smoke Summary

- original_item_count: `389`
- candidate_record_count: `5672`
- positive_candidate_count: `389`
- negative_candidate_count: `5283`
- valid_output_items: `389`
- valid_output_ratio: `1.0`
- future_candidate_used_as_input: `False`
- stage2_val_fallback_used: `False`
- old_association_report_used: `False`
- full_model_forward_executed: `True`
- full_free_rollout_executed: `True`
- candidate_top1: `0.13212435233160622`
- candidate_MRR: `0.32914047588490813`
- candidate_AP: `0.06995069572340755`
- candidate_AUROC: `0.4843919640571625`
- score_components_used: `['future_reappearance_event_prob', 'future_trace_coord_distance', 'future_visibility_prob']`
- external_query_eval_available: `True`
- external_query_signal_positive: `False`

Interpretation: the strict external query bridge is now technically available, but the candidate-expanded semantic-state signal does not transfer positively on this smoke/full 389-item diagnostic set.
