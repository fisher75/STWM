# STWM Real Visibility/Reappearance V1 Current Target Audit 20260427

- can_construct_B_H_K_target: `True`
- most_reliable_target_source: `fut_valid_slot_aligned with obs_valid endpoint/occlusion gate for reappearance`
- minimum_viable_target_definition: future_visibility_target=fut_valid[B,H,K]; future_visibility_mask=token_mask broadcast to [B,H,K]; future_reappearance_target=fut_valid & ((not visible at observed endpoint) or observed-window occlusion) for each slot; target_source=fut_valid_slot_aligned; target_quality=strong_slot_aligned.

## Why Previous Accuracy Was Not Calibrated
The previous exported target was sample-level and almost always positive when any future entity existed; it did not test per-horizon invisibility, reappearance, or calibrated negative entries.

## Available Fields
- fut_valid: available [B,H,K], entity-slot aligned future visibility/presence mask
- obs_valid: available [B,Obs,K], observed visibility/presence mask used for reappearance gate
- semantic_instance_valid: available [B,K,T] but temporal window may be shorter than obs+future and is dataset-instance-source dependent
- entity_valid: represented by token_mask [B,K] plus obs_valid/fut_valid
- candidate_valid_mask: not part of Stage2SemanticDataset training batch; external manifest only
- observed_future_instance_id: semantic_entity_dominant_instance_id and point_ids available, but not needed for minimum slot-aligned fut_valid target
- gt_candidate_id: not available in Stage2SemanticDataset validation split
- trace_valid_mask: tf_out valid_mask available as fut_valid & token_mask
