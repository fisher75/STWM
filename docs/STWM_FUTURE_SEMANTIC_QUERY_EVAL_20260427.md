# STWM Future Semantic Query Eval 20260427

This is a read-only utility evaluation over existing per-item reports. It does not train, rerun inference, or modify official results.

## Overall
- source_report: `/home/chen034/workspace/stwm/reports/stwm_trace_belief_eval_20260424.json`
- future_semantic_trace_field_available: `False`
- exact_blocking_reason_for_visibility_auroc: `current official per-item reports do not contain explicit future_visibility_logit or visibility labels emitted by a FutureSemanticTraceState head`
- exact_blocking_reason_for_uncertainty_ece: `current official per-item reports do not contain calibrated future_uncertainty/confidence fields from a FutureSemanticTraceState head`

## Panels
| panel | rows | top1 | MRR | false confuser | visibility metric | uncertainty metric |
|---|---:|---:|---:|---:|---|---|
| densified_200_context_preserving | 894 | 0.463 | 0.628 | 0.537 | unavailable_no_future_visibility_field | unavailable_no_confidence_or_uncertainty_field |
| heldout_burst_heavy_context_preserving | 1350 | 0.566 | 0.724 | 0.434 | unavailable_no_future_visibility_field | unavailable_no_confidence_or_uncertainty_field |
| heldout_scene_category_video_context_preserving | 1236 | 0.456 | 0.606 | 0.544 | unavailable_no_future_visibility_field | unavailable_no_confidence_or_uncertainty_field |
