# STWM External Baseline Item Audit 20260426

- total_items_found: `389`
- runnable_items: `0`
- exact_blocking_reason: `existing STWM reports expose per-item rankings/subset tags but not raw frame paths, observed prompt masks/boxes, future frame indices, or future candidate masks/boxes required by VOS/tracking baselines`

| baseline | runnable_items | skipped_items |
|---|---:|---:|
| cutie | 0 | 389 |
| sam2 | 0 | 389 |
| cotracker | 0 | 389 |

## Top skipped reasons

- `missing_frame_paths+missing_observed_frame_indices+missing_future_frame_index+missing_observed_target_mask_or_box+missing_future_candidate_masks_or_boxes`: 778
- `missing_frame_paths+missing_observed_frame_indices+missing_future_frame_index+missing_observed_target_mask_or_box_or_point+missing_future_candidate_masks_or_boxes`: 389
