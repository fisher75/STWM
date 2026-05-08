# STWM OSTF V31 Semantic Broadcasting Design

- semantic_identity_role: `context/broadcast/rendering signal for each point token, not a replacement for point-field state`
- future_semantic_logits_contract: `[B,M,H,C]`
- current_pointodyssey_semantic_class_labels_available: `False`
- semantic_status: `not_tested_not_failed`
- recommended_future_targets: `['instance identity labels when available', 'teacher crop embeddings bound to object/point tokens', 'FSTF semantic prototype targets transferred to OSTF objects', 'reacquisition/false-confuser identity utility on semantic hard cases']`
- no_future_semantic_leakage_rule: `future semantic targets may supervise loss but must not enter observed model input`
