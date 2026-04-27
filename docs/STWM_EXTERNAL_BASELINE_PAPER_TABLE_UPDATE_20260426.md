# STWM External Baseline Paper Table Update 20260426

| method | overall top1 | MRR | false-confuser | occlusion top1 | long-gap top1 | OOD top1 | note |
|---|---:|---:|---:|---:|---:|---:|---|
| STWM trace_belief_assoc (official) | 0.509 | 0.660 | 0.491 | 0.552 | 0.548 | 0.518 | official matched per-item mean |
| SAM2 | 0.671 | 0.775 | 0.329 | 0.575 | 0.538 | 0.662 | external manifest item-aligned |
| CoTracker | 0.602 | 0.729 | 0.398 | 0.417 | 0.327 | 0.603 | external manifest item-aligned |
| frozen_external_teacher_only | 0.520 | 0.695 | 0.480 | 0.520 | 0.590 | NA | stwm_reacquisition_v2_eval_20260425 aggregate; not item-aligned to external manifest |
| legacysem | 0.147 | 0.358 | 0.853 | 0.178 | 0.221 | 0.150 | official matched per-item mean |
| Cutie | 0.488 | 0.626 | 0.512 | 0.394 | 0.462 | 0.503 | external manifest item-aligned |

- best_overall_item_aligned: `sam2`
- best_long_gap_item_aligned: `stwm_trace_belief_assoc`
- best_occlusion_item_aligned: `sam2`
- SAM2 and CoTracker should be in main paper; Cutie can be appendix or a smaller row.
