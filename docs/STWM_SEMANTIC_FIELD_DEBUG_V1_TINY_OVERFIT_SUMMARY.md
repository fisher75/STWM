# STWM Semantic Field Debug V1 Tiny Overfit Summary

- Tiny overfit uses 24 train items and 8 heldout tiny-val items from the same cache-aligned subset.
- The objective is semantic prototype CE only; no candidate scorer, teacher loss, rescue loss, or trace-unit auxiliary loss is enabled.

- audit_name: `stwm_semantic_field_debug_v1_tiny_overfit_summary`
- tiny_overfit_started: `True`
- tiny_overfit_success: `False`
- item_count: `32`
- train_item_count: `24`
- tiny_val_item_count: `8`
- batch_size: `4`
- steps_per_prototype_count: `1000`
- loss_mode: `semantic_proto_ce_only`
- best_prototype_count: `32`
- best_train_top5: `0.6590724177628247`
- best_tiny_val_top5: `0.3137755129600362`
- best_overfit_gap_top5: `0.34529690480278846`
- model_can_memorize_semantic_field_when_target_aligned: `False`
