# STWM Semantic Field Debug V1 Target Alignment Audit

- Alignment passes only if cache hits are complete, shape is compatible, valid targets never supervise invalid future slots, and -1 labels only appear under mask=false.
- Exact target_mask == fut_valid is reported separately because feature-target validity can be a subset of future visibility.

- audit_name: `stwm_semantic_field_debug_v1_target_alignment_audit`
- checked_item_count: `32`
- cache_path: `outputs/cache/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428/prototype_targets.npz`
- prototype_count: `64`
- cache_hit_ratio: `1.0`
- target_valid_ratio: `0.9424418604651162`
- target_mask_subset_of_fut_valid_all: `True`
- target_mask_exact_match_ratio: `1.0`
- proto_target_minus_one_only_when_mask_false_all: `True`
- H_consistent_all: `True`
- K_consistent_all: `True`
- slot_order_audit_method: `point_ids proxy plus cache/sample key equality; visual crop trace examples recorded`
- label_nonzero_count: `43`
- alignment_ok: `True`
