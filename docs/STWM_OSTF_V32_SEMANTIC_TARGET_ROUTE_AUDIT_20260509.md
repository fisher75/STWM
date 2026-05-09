# STWM OSTF V32 Semantic Target Route Audit

- semantic_field_target_available_now: `False`
- semantic_training_run_this_round: `False`
- semantic_status: `not_tested_not_failed`
- semantic_broadcasting_contract: `{'semantic_is_context_only': True, 'semantic_must_not_compress_physical_field': True, 'future_semantic_logits_shape': '[B,M,H,C]', 'semantic_loss_disabled_until_targets_exist': True}`
- candidate_future_semantic_targets: `['PointOdyssey instance identity if reliable and split-safe', 'crop teacher embeddings from observed/video-derived object crops', 'SAM2/DINO/CLIP teacher per object or per point', 'FSTF/TUSB semantic prototype transfer as observed semantic object token and future prototype target when aligned']`
- no_future_leakage_rule: `future semantic target may be used only as supervision, never as model input`
- recommended_semantic_next_step: `construct explicit semantic/identity targets after trajectory field dynamics are validated`
