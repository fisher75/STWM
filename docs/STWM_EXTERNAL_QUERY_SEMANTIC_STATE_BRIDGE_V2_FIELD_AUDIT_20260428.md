# STWM External Query Semantic State Bridge V2 Field Audit

- total_items: `389`
- frame_paths_exist_items: `389`
- observed_target_prompt_available_items: `389`
- gt_candidate_id_in_future_candidates_items: `389`
- candidate_expanded_records_possible: `5672`
- positive_candidate_records: `389`
- negative_candidate_records: `5283`
- can_construct_query_only_model_input_without_future_target_leakage: `True`

## Minimal Batch Schema
- K=1 observed-target query slot.
- Candidate boxes/masks are held out of rollout input and used only for scoring.
- Future target state is dummy/zero for forward only; evaluation uses candidate labels after export.
