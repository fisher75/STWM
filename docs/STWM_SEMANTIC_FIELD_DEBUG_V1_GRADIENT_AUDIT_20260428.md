# STWM Semantic Field Debug V1 Gradient Audit

- The backward pass optimizes only semantic prototype CE on a fixed batch.
- Stage1 and TUSB dynamic paths should have zero gradient.

- audit_name: `stwm_semantic_field_debug_v1_gradient_audit`
- proto_loss_grad_reaches_future_head: `True`
- proto_loss_grad_reaches_tusb_semantic: `True`
- proto_loss_grad_reaches_tusb_dynamic: `False`
- stage1_grad_detected: `False`
- dynamic_grad_detected: `False`
