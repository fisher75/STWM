# STWM Papergrade Protocol V2 Dataset Coverage Audit

VIPSeg is present in raw/future target caches but absent from eligible semantic-memory splits because observed semantic memory coverage is zero for VIPSeg.

This is a dataset/protocol limitation, not evidence that the semantic trace world model fails on VIPSeg.

- audit_name: `stwm_papergrade_protocol_v2_dataset_coverage_audit`
- why_final_splits_show_only_vspw: `VIPSeg has future semantic targets but zero valid observed semantic memory targets in the current observed feature cache, so eligibility requires observed memory and filters VIPSeg out.`
- vipseg_raw_entries_available_count: `3149`
- vipseg_future_feature_target_count: `2791`
- vipseg_observed_target_count: `0`
- vipseg_observed_future_overlap_count: `0`
- vipseg_materializable_count: `0`
- vipseg_filtered_by_item_key_mismatch: `False`
- vipseg_filtered_by_missing_crops: `True`
- vipseg_filtered_by_target_cache: `False`
- vipseg_filtered_by_timeout: `False`
- vipseg_filtered_by_code_bug: `unclear`
- vipseg_blocker: `observed semantic memory cache reports direct_cache_item_hits only for VSPW; VIPSeg observed_feature_mask is all false. The likely blocker is missing/zero VIPSeg observed predecode semantic crops or missing VIPSeg teacher/predecode cache entries, not missing future targets.`
- vipseg_can_be_included_in_v2: `False`
- observed_feature_fast_path: `predecode_crop_clip`
- predecode_cache_path: `data/processed/stage2_tusb_v3_predecode_cache_20260418`
- direct_cache_item_hits: `1522`
- fullscale_v1_train_items: `1065`
- fullscale_v1_val_items: `228`
- fullscale_v1_test_items: `229`
