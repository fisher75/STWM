# STWM V4.2 Hard Query Protocol

## Goal

Increase query difficulty from data/query construction only, while keeping model and loss unchanged.

Original query protocol is kept unchanged for side-by-side comparison.

## Inputs

- source manifest:
  - `manifests/minisplits/stwm_week2_minival_v2.json`
- eventful mining report:
  - `reports/stwm_v4_2_eventful_protocol_v1.json`

## Hard Query Construction

For selected clips, augment `text_labels` and metadata tags with three hardness dimensions:

1. same-category distractor queries
   - add target/distractor label tags from selected primary/secondary labels
2. spatial disambiguation tags
   - add spatial tags based on target centroid region (`left/right/center`, `top/middle/bottom`)
3. reappearing object queries
   - add explicit reappearance query tag on clips with reappearance reason

## Outputs

- manifest:
  - `manifests/minisplits/stwm_v4_2_hard_query_minival_v1.json`
- clip ids:
  - `manifests/minisplits/stwm_v4_2_hard_query_clip_ids_v1.json`
- report:
  - `reports/stwm_v4_2_hard_query_protocol_v1.json`

## Coverage Summary

From `stwm_v4_2_hard_query_protocol_v1.json`:

- selected_count: `18`
- coverage_insufficient: `False`

Hard query type counts:

- same_category_distractor: `18`
- spatial_disambiguation: `18`
- reappearing_object_query: `9`

## Protocol Boundary

- no architecture modification
- no module expansion
- no loss modification
- no replacement of original query protocol (new protocol is additive)
