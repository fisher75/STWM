# STWM Semantic Field Debug V1 Decision

- One-batch overfit is the gate before any medium semantic-field training.
- If this gate fails with clean alignment/input/gradients, the semantic target space or optimization recipe must be redesigned before scaling.

- audit_name: `stwm_semantic_field_debug_v1_decision`
- target_alignment_ok: `True`
- semantic_input_valid: `True`
- proto_loss_grad_reaches_tusb_semantic: `True`
- stage1_grad_detected: `False`
- dynamic_grad_detected: `False`
- one_batch_overfit_success: `True`
- one_batch_overfit_c32_success: `True`
- one_batch_overfit_c64_success: `True`
- tiny_overfit_success: `False`
- root_cause: `model_capacity_or_training_recipe`
- recommended_next_step_choice: `improve_prototype_targets`
- target_alignment_audit: `reports/stwm_semantic_field_debug_v1_target_alignment_audit_20260428.json`
- semantic_input_audit: `reports/stwm_semantic_field_debug_v1_semantic_input_audit_20260428.json`
- gradient_audit: `reports/stwm_semantic_field_debug_v1_gradient_audit_20260428.json`
- one_batch_overfit_c32: `reports/stwm_semantic_field_debug_v1_one_batch_overfit_c32.json`
- one_batch_overfit_c64: `reports/stwm_semantic_field_debug_v1_one_batch_overfit_c64.json`
- tiny_overfit_summary: `reports/stwm_semantic_field_debug_v1_tiny_overfit_summary_20260428.json`
