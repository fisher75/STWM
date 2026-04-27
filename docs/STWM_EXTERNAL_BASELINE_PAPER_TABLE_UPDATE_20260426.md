# STWM External Baseline Paper Table Update 20260426

## Main Item-Aligned Table
| method | overall top1 | MRR | false confuser | occlusion top1 | long-gap top1 | crossing top1 | OOD hard top1 | placement |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| STWM trace_belief_assoc (official) | 0.509 | 0.660 | 0.491 | 0.552 | 0.548 (numeric best; not significant vs SAM2) | 0.472 | 0.518 | main_item_aligned |
| SAM2 | 0.671 | 0.775 | 0.329 | 0.575 | 0.538 | 0.689 | 0.662 | main_item_aligned |
| CoTracker | 0.602 | 0.729 | 0.398 | 0.417 | 0.327 | 0.659 | 0.603 | main_item_aligned |
| legacysem | 0.147 | 0.358 | 0.853 | 0.178 | 0.221 | 0.131 | 0.150 | main_internal_item_aligned |
| Cutie | 0.488 | 0.626 | 0.512 | 0.394 | 0.462 | 0.490 | 0.503 | appendix_or_small_row_item_aligned |

## Footnote
Item-aligned diagnostic protocol: 389 hard cases, 16-frame local window, max-side 384, single future-frame candidate association. This is not a full-video tracking benchmark. frozen_external_teacher_only is a 1038-row reacquisition_v2 aggregate and must not be mixed as an item-aligned external row without a footnote.

## Non Item-Aligned Reference Rows
| method | count | source | placement |
|---|---:|---|---|
| frozen_external_teacher_only | 1038 | stwm_reacquisition_v2_eval_20260425 aggregate; not item-aligned to external manifest | appendix_non_item_aligned_reference |
