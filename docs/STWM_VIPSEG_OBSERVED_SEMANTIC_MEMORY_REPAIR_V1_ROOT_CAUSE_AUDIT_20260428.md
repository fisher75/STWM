# STWM VIPSeg Observed Semantic Memory Repair V1 Root-Cause Audit

- VIPSeg failed because observed-memory lookup used exact dataset-name casing.
- Alias repair recovers the locally available VIPSeg predecode subset, but the local predecode cache is still incomplete.

- audit_name: `stwm_vipseg_observed_semantic_memory_repair_v1_root_cause_audit`
- vipseg_raw_item_count: `3149`
- vipseg_future_target_count: `2791`
- vipseg_observed_crop_availability_before_repair: `0`
- vipseg_observed_crop_availability_after_alias_repair: `284`
- vipseg_observed_semantic_crop_feature_availability: `284`
- vipseg_predecode_cache_entries_case_sensitive_upper: `0`
- vipseg_predecode_cache_entries_alias_vipseg: `284`
- vipseg_predecode_cache_entries_used: `284`
- vipseg_predecode_missing_count: `2865`
- vipseg_item_keys_match_future_target_cache: `True`
- vipseg_observed_feature_mask_was_all_false_reason: `dataset-name casing mismatch: item keys use VIPSEG while local predecode files use VIPSeg; old lookup did not try aliases.`
- root_cause: `dataset_name_casing_bug_plus_partial_vipseg_predecode_cache`
- missing_crops: `True`
- item_key_mismatch: `False`
- dataset_name_mismatch: `True`
- cache_path_mismatch: `False`
- code_filter_bug: `True`
- can_rebuild_from_raw_vipseg_observed_frames: `not completed in this protocol-hardening pass; requires a dedicated raw VIPSeg crop materialization job because predecode covers only 284/3149 VIPSeg items.`
- repair_strategy: `fixed dataset-name aliases and rebuilt observed features from local predecode crops; remaining blocker is missing VIPSeg predecode/raw observed crop materialization.`
- no_future_leakage: `True`
- no_candidate_scorer: `True`
