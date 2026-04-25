# STWM Belief Mechanism Source Audit 20260425

## Source Reports

| report | exists | valid_json | panels | per-item rows |
|---|---:|---:|---|---:|
| `trace_belief_eval` | True | True | densified_200_context_preserving, heldout_burst_heavy_context_preserving, heldout_scene_category_video_context_preserving | 31320 |
| `belief_final_eval` | True | True | densified_200_context_preserving, legacy_85_context_preserving, protocol_v3_extended_600_context_preserving | 14040 |
| `belief_strict_bootstrap` | True | True | densified_200_context_preserving, protocol_v3_extended_600_context_preserving | 0 |


## Per-Item Schema

Observed keys: `clip_id, dataset, future_mask_iou_at_top1, item_split, method_name, mrr, panel_name, protocol_eval_context_entity_count, protocol_item_id, query_future_hit_rate, query_future_localization_error, query_future_top1_acc, scoring_mode, seed, subset_tags, target_rank, top1_candidate_id, top5_hit`

Paired teacher/belief rows: `3480`

## Confidence Policy

Raw candidate score maps or calibrated probabilities are not present in the source per-item rows. Reliability is therefore computed using `rank_confidence_proxy = MRR = 1 / target_rank`. This is a rank-confidence proxy, not a calibrated model probability.

Subset tags are not used to construct confidence. They are used only for the requested hard-case group breakdowns.

## Method Mapping

- Teacher baseline: `TUSB-v3.1::best_semantic_hard.pt` with `frozen_external_teacher_only`.
- Belief: `TUSB-v3.1::best_semantic_hard.pt` with `trace_belief_assoc`.

Leakage check passed: `True`.
