# STWM VIPSeg Raw Observed Memory V2 Root-Cause Audit

- audit_name: `stwm_vipseg_raw_observed_memory_v2_root_cause_audit`
- stage2semanticdataset_can_load_vipseg_samples: `True`
- vipseg_samples_have_semantic_rgb_crop: `raw fast path reconstructs observed crops from frames/masks/bboxes instead of Stage2 __getitem__ semantic_rgb_crop tensor`
- vipseg_samples_have_semantic_rgb_crop_temporal: `not required by raw fast path; observed last-frame crop is used`
- vipseg_samples_have_obs_valid: `True`
- vipseg_samples_have_semantic_crop_valid: `derived from observed valid mask and raw mask-derived boxes`
- why_vipseg_crops_did_not_enter_observed_cache: `partial predecode was accepted with observed_min_coverage=0 and exact VIPSEG casing failed before alias repair`
- build_or_load_accepted_partial_predecode_cache_reason: `observed_min_coverage default was 0.0 and no dataset-specific rejection was enforced`
- observed_min_coverage_current: `0.0`
- force_raw_stage2dataset_reconstruction_needed: `True`
- vipseg_raw_rebuild_can_avoid_predecode_cache: `True`
- v2_raw_feature_report: `reports/stwm_vipseg_raw_observed_semantic_features_v2_20260428.json`
