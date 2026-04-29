# STWM Semantic Field Tiny Recipe Ablation V1

- All variants keep Stage1 and dynamic trace paths frozen.
- The ablation optimizes semantic prototype objectives only; no candidate scorer is used.

- audit_name: `stwm_semantic_field_tiny_recipe_ablation_v1`
- item_count: `32`
- train_item_count: `24`
- val_item_count: `8`
- steps_per_variant: `300`
- best_variant: `semantic_fusion_gate_norm_c32`
- best_variant_train_top5: `0.6883645141580611`
- best_variant_val_top5: `0.34438775494998813`
- any_variant_train_top5_gt_0p8: `False`
- any_variant_val_top5_beats_frequency: `False`
- recipe_ablation_finds_success: `False`
- trace_regression_detected: `False`
